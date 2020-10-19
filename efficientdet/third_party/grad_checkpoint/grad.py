# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a re-distribution based on the copyright in LICENSE.txt
# ==============================================================================
"""Recompute tensorflow graph reducing GPU memory usage."""
# pylint: disable=g-direct-tensorflow-import,g-doc-args,g-doc-return-or-yield
# pylint: disable=g-complex-comprehension,g-doc-exception
# pylint: disable=protected-access

import contextlib
import functools
import re
import sys
import time
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

# save original gradients since tf.gradient could be monkey-patched.
import third_party.graph_edit as ge
tf_gradient_function = tf.gradients
sys.setrecursionlimit(10000)
# getting rid of "WARNING:tensorflow:VARIABLES collection name is deprecated"
setattr(tf.GraphKeys, "VARIABLES", "variables")


def toposort(nodes):
  """Sort topologically nodes based on https://pypi.org/project/toposort/."""
  # Special case empty input.
  if not nodes:
    return

  # Copy the input so as to leave it unmodified.
  nodes = nodes.copy()
  # Ignore self dependencies.
  for k, v in nodes.items():
    v.discard(k)
  # Find all items that don't depend on anything.
  isolated_items = functools.reduce(set.union, nodes.values()) - set(
      nodes.keys())
  # Add empty dependences where needed.
  nodes.update({item: set() for item in isolated_items})
  while True:
    ordered = set(item for item, dep in nodes.items() if not dep)
    if not ordered:
      break
    yield ordered
    nodes = {
        item: (dep - ordered)
        for item, dep in nodes.items()
        if item not in ordered
    }
  if nodes:
    raise ValueError("expected empty, but got {}".format(nodes))


# ISSUE: https://github.com/cybertronai/gradient-checkpointing/issues/38
def tf_gradients(ys, *args, **kwargs):
  """Decorate tf.gradients calls with explicit device placement.

  Decorate tf.gradients calls with explicit device placement to avoid memory
    leaks when splitting model across multiple GPUs
  """
  source = ys[0] if isinstance(ys, (list, tuple)) else ys
  device = source.op.node_def.device if isinstance(source, tf.Tensor) else None
  with tf.device(device):
    return tf_gradient_function(ys, *args, **kwargs)


MIN_CHECKPOINT_NODE_SIZE = 1024  # use lower value during testing


# specific versions we can use to do process-wide replacement of tf.gradients
def gradients_speed(ys, xs, grad_ys=None, **kwargs):
  """Optimize for speed."""
  return gradients(ys, xs, grad_ys, checkpoints="speed", **kwargs)


def gradients_memory(ys, xs, grad_ys=None, **kwargs):
  """Optimize for memory."""
  return gradients(ys, xs, grad_ys, checkpoints="memory", **kwargs)


def gradients_collection(ys, xs, grad_ys=None, **kwargs):
  """Use user-provided collection."""
  return gradients(ys, xs, grad_ys, checkpoints="collection", **kwargs)


def gradients(ys, xs, grad_ys=None, checkpoints="collection", **kwargs):
  """Recompute gradients.

  Authors: Tim Salimans & Yaroslav Bulatov
  Modified by: Nikolay Zakirov

  memory efficient gradient implementation inspired by
  "Training Deep Nets with Sublinear Memory Cost"
  by Chen et al. 2016 (https://arxiv.org/abs/1604.06174)

  ys,xs,grad_ys,kwargs are the arguments to standard tensorflow tf.gradients
  (https://www.tensorflow.org/versions/r0.12/api_docs/python/train.html#gradients)

  'checkpoints' can either be
      - a list consisting of tensors from the forward pass of the neural net
        that we should re-use when calculating the gradients in the
        backward pass
        all other tensors that do not appear in this list will be re-computed
      - a string or list specifying how this list should be determined.
      currently we support
          - 'speed':  checkpoint all outputs of convolutions and matmuls.
          these ops are usually the most expensive,
                      so checkpointing them maximizes the running speed
                      (this is a good option if nonlinearities, concats,
                      batchnorms, etc are taking up a lot of memory)
          - 'memory': try to minimize the memory usage
                      (currently using a very simple strategy that
                      identifies a number of bottleneck tensors in the
                      graph to checkpoint)
          - 'collection': look for a tensorflow collection named
          'checkpoints', which holds the tensors to checkpoint
          - a list: a list of strings to be matched in the names of
          the tensors
  """
  #    print("Calling memsaving gradients with", checkpoints)
  if not isinstance(ys, list):
    ys = [ys]
  if not isinstance(xs, list):
    xs = [xs]

  bwd_ops = ge.get_backward_walk_ops([y.op for y in ys], inclusive=True)

  logging.debug("bwd_ops: %s", len(bwd_ops))

  # forward ops are all ops that are candidates for recomputation
  fwd_ops = ge.get_forward_walk_ops([x.op for x in xs],
                                    inclusive=True,
                                    within_ops=bwd_ops)
  logging.debug("fwd_ops: %s", len(fwd_ops))

  # exclude ops with no inputs
  fwd_ops = [op for op in fwd_ops if op.inputs]

  logging.debug("fwd_ops: %s", len(fwd_ops))
  # don't recompute xs, remove variables
  xs_ops = _to_ops(xs)
  fwd_ops = [op for op in fwd_ops if op not in xs_ops]
  fwd_ops = [op for op in fwd_ops if "/assign" not in op.name]
  fwd_ops = [op for op in fwd_ops if "/Assign" not in op.name]
  fwd_ops = [op for op in fwd_ops if "/read" not in op.name]
  logging.debug("fwd_ops: %s", len(fwd_ops))
  ts_all = ge.filter_ts(fwd_ops, True)  # get the tensors
  logging.debug("ts_all: %s", len(ts_all))

  ts_all = [t for t in ts_all if "/read" not in t.name]
  ts_all = set(ts_all) - set(xs) - set(ys)
  logging.debug("ts_all: %s", len(ts_all))

  # construct list of tensors to checkpoint during forward pass, if not
  # given as input
  if not isinstance(checkpoints, list):
    if checkpoints == "collection":
      checkpoints = tf.get_collection("checkpoints")

    elif checkpoints == "speed":
      # checkpoint all expensive ops to maximize running speed
      checkpoints = ge.filter_ts_from_regex(fwd_ops, "conv2d|Conv|MatMul")

    elif checkpoints == "memory":

      # remove very small tensors and some weird ops
      def fixdims(
          t
      ):  # tf.Dimension values are not compatible with int, convert manually
        try:
          return [int(e if e is not None else 64) for e in t.as_list()]
        except ValueError as e:
          logging.exception("%s", e)
          logging.exception("unknown shape %s", t)
          return [0]  # unknown shape

      ts_all = [
          t for t in ts_all
          if np.prod(fixdims(t.shape)) > MIN_CHECKPOINT_NODE_SIZE
          # if (tf.size(t) > MIN_CHECKPOINT_NODE_SIZE)
      ]
      logging.debug("ts_all: %s", len(ts_all))
      ts_all = [t for t in ts_all if "L2Loss" not in t.name]
      ts_all = [t for t in ts_all if "entropy" not in t.name]
      ts_all = [t for t in ts_all if "FusedBatchNorm" not in t.name]
      ts_all = [t for t in ts_all if "Switch" not in t.name]
      ts_all = [t for t in ts_all if "dropout" not in t.name]
      # DV: FP16_FIX - need to add 'Cast' layer here to make it work for FP16
      ts_all = [t for t in ts_all if "Cast" not in t.name]
      logging.debug("ts_all: %s", len(ts_all))

      # filter out all tensors that are inputs of the backward graph
      with capture_ops() as bwd_ops:
        tf_gradients(ys, xs, grad_ys, **kwargs)

      bwd_inputs = [t for op in bwd_ops for t in op.inputs]
      # list of tensors in forward graph that is in input to bwd graph
      ts_filtered = list(set(bwd_inputs).intersection(ts_all))
      debug_print("Using tensors %s", ts_filtered)

      # try two slightly different ways of getting bottlenecks tensors
      # to checkpoint
      logging.debug("len(ts_filtered): %s", len(ts_filtered))
      logging.debug("len(ts_all) %s", len(ts_all))
      for ts in [ts_filtered, ts_all]:

        # get all bottlenecks in the graph
        bottleneck_ts = []
        for t in ts:

          b = set(
              ge.get_backward_walk_ops(
                  t.op, inclusive=True, within_ops=fwd_ops))
          f = set(
              ge.get_forward_walk_ops(
                  t.op, inclusive=False, within_ops=fwd_ops))
          # check that there are no shortcuts
          b_inp = {inp for op in b for inp in op.inputs}.intersection(ts_all)
          f_inp = {inp for op in f for inp in op.inputs}.intersection(ts_all)
          if not set(b_inp).intersection(
              f_inp) and len(b_inp) + len(f_inp) >= len(ts_all):
            bottleneck_ts.append(t)  # we have a bottleneck!
          else:
            logging.debug("Rejected bottleneck candidate and ops %s %d", [t],
                          len(b_inp) + len(f_inp) - len(ts_all))

        # success? or try again without filtering?
        if len(bottleneck_ts) >= np.sqrt(
            len(ts_filtered)):  # yes, enough bottlenecks found!
          break
      # bottleneck_ts = [t for t in ts_all if 'Add' in t.name]
      # logging.debug("Add only ts_all: %s", len(bottleneck_ts))

      if not bottleneck_ts:
        raise Exception(
            "unable to find bottleneck tensors! please provide checkpoint "
            'nodes manually, or use checkpoints="speed" or a list of strings.')
      logging.debug("len(bottleneck_ts): %s", len(bottleneck_ts))

      # sort the bottlenecks
      bottlenecks_sorted_lists = tf_toposort(bottleneck_ts, within_ops=fwd_ops)
      sorted_bottlenecks = [t for ts in bottlenecks_sorted_lists for t in ts]

      # save an approximately optimal number ~ sqrt(N)
      n_filtered = len(ts_filtered)
      if len(bottleneck_ts) <= np.ceil(np.sqrt(n_filtered)):
        checkpoints = sorted_bottlenecks
      else:
        step = int(np.ceil(len(bottleneck_ts) / np.sqrt(n_filtered)))
        checkpoints = sorted_bottlenecks[step::step]

    else:
      raise Exception('%s is unsupported input for "checkpoints"' %
                      (checkpoints,))
  else:
    # exclude some layers as was done in the original bottleneck searching
    # algorithm
    for excl_layer in [
        "L2Loss", "entropy", "FusedBatchNorm", "Switch", "dropout", "Cast"
    ]:
      ts_all = [t for t in ts_all if excl_layer not in t.name]
      logging.info("Excluding %s from ts_all: %d", excl_layer, len(ts_all))

    # leave only layers that match strings in checkpoints list
    matchers = {c: re.compile('.*' + c + '.*') for c in checkpoints}
    ts_set = set()
    for c, matcher in matchers.items():
      ts_match = [t for t in ts_all if matcher.match(t.name)]
      ts_set.update(ts_match)
    ts_all = list(ts_set)
    logging.info("Leaving only %s in ts_all: %d", checkpoints, len(ts_all))
    checkpoints = ts_all.copy()

  checkpoints = list(set(checkpoints).intersection(ts_all))

  # at this point selection happened and checkpoints is list of nodes
  # assert isinstance(checkpoints, list)

  # TODO(nikzak): implement multithreading in graph recomputation
  logging.info("Checkpoint nodes used: %s", len(checkpoints))
  # better error handling of special cases
  # xs are already handled as checkpoint nodes, so no need to include them
  xs_intersect_checkpoints = set(xs).intersection(set(checkpoints))
  if xs_intersect_checkpoints:
    debug_print("Warning, some input nodes are also checkpoint nodes: %s",
                xs_intersect_checkpoints)
  ys_intersect_checkpoints = set(ys).intersection(set(checkpoints))
  debug_print("ys: %s, checkpoints: %s, intersect: %s", ys, checkpoints,
              ys_intersect_checkpoints)
  # saving an output node (ys) gives no benefit in memory while creating
  # new edge cases, exclude them
  if ys_intersect_checkpoints:
    debug_print("Warning, some output nodes are also checkpoints nodes: %s",
                format_ops(ys_intersect_checkpoints))

  # remove initial and terminal nodes from checkpoints list if present

  checkpoints = list(set(checkpoints) - set(ys) - set(xs))
  logging.info("Pruned initial and terminal nodes. Leaving %d",
               len(checkpoints))

  # check that we have some nodes to checkpoint
  if not checkpoints:
    raise Exception("no checkpoints nodes found or given as input! ")

  # disconnect dependencies between checkpointed tensors
  checkpoints_disconnected = {}
  for x in checkpoints:
    if x.op and x.op.name is not None:
      grad_node = tf.stop_gradient(x, name=x.op.name + "_sg")
    else:
      grad_node = tf.stop_gradient(x)
    grad_node.op._set_device(x.op.node_def.device)
    checkpoints_disconnected[x] = grad_node

  # partial derivatives to the checkpointed tensors and xs
  ops_to_copy = fast_backward_ops(
      seed_ops=[y.op for y in ys], stop_at_ts=checkpoints, within_ops=fwd_ops)
  debug_print("Found %s ops to copy within fwd_ops %s, seed %s, stop_at %s",
              len(ops_to_copy), fwd_ops, [r.op for r in ys], checkpoints)
  debug_print("ops_to_copy = %s", ops_to_copy)
  debug_print("Processing list %s", ys)
  _, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
  for origin_op, op in info._transformed_ops.items():
    op._set_device(origin_op.node_def.device)
  copied_ops = info._transformed_ops.values()
  debug_print("Copied %s to %s", ops_to_copy, copied_ops)
  ge.reroute_ts(
      checkpoints_disconnected.values(),
      checkpoints_disconnected.keys(),
      can_modify=copied_ops)
  debug_print("Rewired %s in place of %s restricted to %s",
              checkpoints_disconnected.values(),
              checkpoints_disconnected.keys(), copied_ops)

  # get gradients with respect to current boundary + original x's
  copied_ys = [info._transformed_ops[y.op]._outputs[0] for y in ys]
  boundary = list(checkpoints_disconnected.values())
  dv = tf_gradients(ys=copied_ys, xs=boundary + xs, grad_ys=grad_ys, **kwargs)
  debug_print("Got gradients %s", dv)
  debug_print("for %s", copied_ys)
  debug_print("with respect to %s", boundary + xs)

  inputs_to_do_before = [y.op for y in ys]
  if grad_ys is not None:
    inputs_to_do_before += grad_ys
  wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
  my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

  # partial derivatives to the checkpointed nodes
  # dictionary of "node: backprop" for nodes in the boundary
  d_checkpoints = dict(
      zip(checkpoints_disconnected.keys(), dv[:len(checkpoints_disconnected)]))

  # partial derivatives to xs (usually the params of the neural net)
  d_xs = dv[len(checkpoints_disconnected):]

  # incorporate derivatives flowing through the checkpointed nodes
  logging.info("Sorting nodes topologically")
  checkpoints_sorted_lists = tf_toposort(checkpoints, within_ops=fwd_ops)
  logging.info("Rebuilding graph with %d checkpoints",
               len(checkpoints_sorted_lists))
  for index, ts in enumerate(checkpoints_sorted_lists[::-1]):
    if index % 50 == 0:
      logging.info("Processed %d nodes", index)
    debug_print("Processing list %s", ts)
    checkpoints_other = [r for r in checkpoints if r not in ts]
    checkpoints_disconnected_other = [
        checkpoints_disconnected[r] for r in checkpoints_other
    ]

    # copy part of the graph below current checkpoint node, stopping at
    # other checkpoints nodes
    ops_to_copy = fast_backward_ops(
        within_ops=fwd_ops,
        seed_ops=[r.op for r in ts],
        stop_at_ts=checkpoints_other)
    debug_print("Found %s ops to copy within %s, seed %s, stop_at %s",
                len(ops_to_copy), fwd_ops, [r.op for r in ts],
                checkpoints_other)
    debug_print("ops_to_copy = %s", ops_to_copy)
    if not ops_to_copy:  # we're done!
      break
    _, info = ge.copy_with_input_replacements(ge.sgv(ops_to_copy), {})
    for origin_op, op in info._transformed_ops.items():
      op._set_device(origin_op.node_def.device)
    copied_ops = info._transformed_ops.values()
    debug_print("Copied %s to %s", ops_to_copy, copied_ops)
    ge.reroute_ts(
        checkpoints_disconnected_other,
        checkpoints_other,
        can_modify=copied_ops)
    debug_print("Rewired %s in place of %s restricted to %s",
                checkpoints_disconnected_other, checkpoints_other, copied_ops)

    # gradient flowing through the checkpointed node
    boundary = [info._transformed_ops[r.op]._outputs[0] for r in ts]
    substitute_backprops = [d_checkpoints[r] for r in ts]
    dv = tf_gradients(
        boundary,
        checkpoints_disconnected_other + xs,
        grad_ys=substitute_backprops,
        **kwargs)
    debug_print("Got gradients %s", dv)
    debug_print("for %s", boundary)
    debug_print("with respect to %s", checkpoints_disconnected_other + xs)
    debug_print("with boundary backprop substitutions %s", substitute_backprops)

    inputs_to_do_before = [d_checkpoints[r].op for r in ts]
    wait_to_do_ops = list(copied_ops) + [g.op for g in dv if g is not None]
    my_add_control_inputs(wait_to_do_ops, inputs_to_do_before)

    # partial derivatives to the checkpointed nodes
    for r, dr in zip(checkpoints_other, dv[:len(checkpoints_other)]):
      if dr is not None:
        if d_checkpoints[r] is None:
          d_checkpoints[r] = dr
        else:
          d_checkpoints[r] += dr

    def _unsparsify(x):
      if not isinstance(x, tf.IndexedSlices):
        return x
      if x.dense_shape is None:
        raise ValueError(
            "memory_saving_gradients has sparse gradients of unknown shape.")

      indices = x.indices
      while indices.shape.ndims < x.values.shape.ndims:
        indices = tf.expand_dims(indices, -1)
      return tf.scatter_nd(indices, x.values, x.dense_shape)

    # partial derivatives to xs (usually the params of the neural net)
    d_xs_new = dv[len(checkpoints_other):]
    for j in range(len(xs)):
      if d_xs_new[j] is not None:
        if d_xs[j] is None:
          d_xs[j] = _unsparsify(d_xs_new[j])
        else:
          d_xs[j] += _unsparsify(d_xs_new[j])

  return d_xs


def tf_toposort(ts, within_ops=None):
  """Sort computation graph topologically."""
  all_ops = ge.get_forward_walk_ops([x.op for x in ts], within_ops=within_ops)

  deps = {}
  for op in all_ops:
    for o in op.outputs:
      deps[o] = set(op.inputs)
  sorted_ts = toposort(deps)

  # only keep the tensors from our original list
  ts_sorted_lists = []
  for l in sorted_ts:
    keep = list(set(l).intersection(ts))
    if keep:
      ts_sorted_lists.append(keep)

  return ts_sorted_lists


def fast_backward_ops(within_ops, seed_ops, stop_at_ts):
  """Get backward ops."""
  bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
  ops = bwd_ops.intersection(within_ops).difference([t.op for t in stop_at_ts])
  return list(ops)


@contextlib.contextmanager
def capture_ops():
  """Capture ops.

  Decorator to capture ops created in the block.
  with capture_ops() as ops:
    # create some ops
  print(ops) # => prints ops created.
  """
  micros = int(time.time() * 10**6)
  scope_name = str(micros)
  op_list = []
  with tf.name_scope(scope_name):
    yield op_list

  g = tf.get_default_graph()
  op_list.extend(ge.select_ops(scope_name + "/.*", graph=g))


def _to_op(tensor_or_op):
  if hasattr(tensor_or_op, "op"):
    return tensor_or_op.op
  return tensor_or_op


def _to_ops(iterable):
  if not _is_iterable(iterable):
    return iterable
  return [_to_op(i) for i in iterable]


def _is_iterable(o):
  try:
    _ = iter(o)
  except TypeError:
    return False
  return True


DEBUG_LOGGING = False


def debug_print(s, *args):
  """Print logs while debugging.

  Like logger.log, but also replaces all TensorFlow ops/tensors with their
  names. Sensitive to value of DEBUG_LOGGING, see enable_debug/disable_debug

  Usage:
    debug_print("see tensors %s for %s", tensorlist, [1,2,3])
  """
  if DEBUG_LOGGING:
    formatted_args = [format_ops(arg) for arg in args]
    print("DEBUG " + s % tuple(formatted_args))


def format_ops(ops, sort_outputs=True):
  """Convert Tensor/Operation op to op.name, rest to str(op).

  Helper method for printing ops. Converts Tensor/Operation op to op.name,
  rest to str(op).
  """
  if hasattr(ops, "__iter__") and not isinstance(ops, str):
    l = [(op.name if hasattr(op, "name") else str(op)) for op in ops]
    if sort_outputs:
      return sorted(l)
    return l
  return ops.name if hasattr(ops, "name") else str(ops)


def my_add_control_inputs(wait_to_do_ops, inputs_to_do_before):
  """Add control inputs."""
  for op in wait_to_do_ops:
    ci = [
        i for i in inputs_to_do_before
        if op.control_inputs is None or i not in op.control_inputs
    ]
    ge.add_control_inputs(op, ci)
