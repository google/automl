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
# ==============================================================================
"""Utility functions for the graph_editor."""
# pylint: disable=g-direct-tensorflow-import

import re
from six import iteritems
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import array_ops as tf_array_ops
from tensorflow.python.util.compat import collections_abc

__all__ = [
    "make_list_of_op",
    "get_tensors",
    "make_list_of_t",
    "get_generating_ops",
    "get_consuming_ops",
    "ControlOutputs",
    "placeholder_name",
    "make_placeholder_from_tensor",
    "make_placeholder_from_dtype_and_shape",
]

# The graph editor sometimes need to create placeholders, they are named
# "geph_*". "geph" stands for Graph-Editor PlaceHolder.
_DEFAULT_PLACEHOLDER_PREFIX = "geph"


def concatenate_unique(la, lb):
  """Add all the elements of `lb` to `la` if they are not there already.

  The elements added to `la` maintain ordering with respect to `lb`.

  Args:
    la: List of Python objects.
    lb: List of Python objects.
  Returns:
    `la`: The list `la` with missing elements from `lb`.
  """
  la_set = set(la)
  for l in lb:
    if l not in la_set:
      la.append(l)
      la_set.add(l)
  return la


# TODO(fkp): very generic code, it should be moved in a more generic place.
class ListView():
  """Immutable list wrapper.

  This class is strongly inspired by the one in tf.Operation.
  """

  def __init__(self, list_):
    if not isinstance(list_, list):
      raise TypeError("Expected a list, got: {}.".format(type(list_)))
    self._list = list_

  def __iter__(self):
    return iter(self._list)

  def __len__(self):
    return len(self._list)

  def __bool__(self):
    return bool(self._list)

  # Python 3 wants __bool__, Python 2.7 wants __nonzero__
  __nonzero__ = __bool__

  def __getitem__(self, i):
    return self._list[i]

  def __add__(self, other):
    if not isinstance(other, list):
      other = list(other)
    return list(self) + other


# TODO(fkp): very generic code, it should be moved in a more generic place.
def is_iterable(obj):
  """Return true if the object is iterable."""
  if isinstance(obj, tf_ops.Tensor):
    return False
  try:
    _ = iter(obj)
  except Exception:  # pylint: disable=broad-except
    return False
  return True


def flatten_tree(tree, leaves=None):
  """Flatten a tree into a list.

  Args:
    tree: iterable or not. If iterable, its elements (child) can also be
      iterable or not.
    leaves: list to which the tree leaves are appended (None by default).
  Returns:
    A list of all the leaves in the tree.
  """
  if leaves is None:
    leaves = []
  if isinstance(tree, dict):
    for _, child in iteritems(tree):
      flatten_tree(child, leaves)
  elif is_iterable(tree):
    for child in tree:
      flatten_tree(child, leaves)
  else:
    leaves.append(tree)
  return leaves


def transform_tree(tree, fn, iterable_type=tuple):
  """Transform all the nodes of a tree.

  Args:
    tree: iterable or not. If iterable, its elements (child) can also be
      iterable or not.
    fn: function to apply to each leaves.
    iterable_type: type use to construct the resulting tree for unknown
      iterable, typically `list` or `tuple`.
  Returns:
    A tree whose leaves has been transformed by `fn`.
    The hierarchy of the output tree mimics the one of the input tree.
  """
  if is_iterable(tree):
    if isinstance(tree, dict):
      res = tree.__new__(type(tree))
      res.__init__(
          (k, transform_tree(child, fn)) for k, child in iteritems(tree))
      return res
    if isinstance(tree, tuple):
      # NamedTuple?
      if hasattr(tree, "_asdict"):
        res = tree.__new__(type(tree), **transform_tree(tree._asdict(), fn))
      else:
        res = tree.__new__(
            type(tree), (transform_tree(child, fn) for child in tree))
      return res
    if isinstance(tree, collections_abc.Sequence):
      res = tree.__new__(type(tree))
      res.__init__(transform_tree(child, fn) for child in tree)
      return res
    return iterable_type(transform_tree(child, fn) for child in tree)
  return fn(tree)


def check_graphs(*args):
  """Check that all the element in args belong to the same graph.

  Args:
    *args: a list of object with a obj.graph property.
  Raises:
    ValueError: if all the elements do not belong to the same graph.
  """
  graph = None
  for i, sgv in enumerate(args):
    if graph is None and sgv.graph is not None:
      graph = sgv.graph
    elif sgv.graph is not None and sgv.graph is not graph:
      raise ValueError("Argument[{}]: Wrong graph!".format(i))


def get_unique_graph(tops, check_types=None, none_if_empty=False):
  """Return the unique graph used by the all the elements in tops.

  Args:
    tops: list of elements to check (usually a list of tf.Operation and/or
      tf.Tensor). Or a tf.Graph.
    check_types: check that the element in tops are of given type(s). If None,
      the types (tf.Operation, tf.Tensor) are used.
    none_if_empty: don't raise an error if tops is an empty list, just return
      None.
  Returns:
    The unique graph used by all the tops.
  Raises:
    TypeError: if tops is not a iterable of tf.Operation.
    ValueError: if the graph is not unique.
  """
  if isinstance(tops, tf_ops.Graph):
    return tops
  if not is_iterable(tops):
    raise TypeError("{} is not iterable".format(type(tops)))
  if check_types is None:
    check_types = (tf_ops.Operation, tf_ops.Tensor)
  elif not is_iterable(check_types):
    check_types = (check_types,)
  g = None
  for op in tops:
    if not isinstance(op, check_types):
      raise TypeError("Expected a type in ({}), got: {}".format(
          ", ".join([str(t) for t in check_types]), type(op)))
    if g is None:
      g = op.graph
    elif g is not op.graph:
      raise ValueError("Operation {} does not belong to given graph".format(op))
  if g is None and not none_if_empty:
    raise ValueError("Can't find the unique graph of an empty list")
  return g


def make_list_of_op(ops, check_graph=True, allow_graph=True, ignore_ts=False):
  """Convert ops to a list of `tf.Operation`.

  Args:
    ops: can be an iterable of `tf.Operation`, a `tf.Graph` or a single
      operation.
    check_graph: if `True` check if all the operations belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ts: if True, silently ignore `tf.Tensor`.
  Returns:
    A newly created list of `tf.Operation`.
  Raises:
    TypeError: if ops cannot be converted to a list of `tf.Operation` or,
     if `check_graph` is `True`, if all the ops do not belong to the
     same graph.
  """
  if isinstance(ops, tf_ops.Graph):
    if allow_graph:
      return ops.get_operations()
    raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  if not is_iterable(ops):
    ops = [ops]
  if not ops:
    return []
  if check_graph:
    check_types = None if ignore_ts else tf_ops.Operation
    get_unique_graph(ops, check_types=check_types)
  return [op for op in ops if isinstance(op, tf_ops.Operation)]


# TODO(fkp): move this function in tf.Graph?
def get_tensors(graph):
  """Get all the tensors which are input or output of an op in the graph.

  Args:
    graph: a `tf.Graph`.
  Returns:
    A list of `tf.Tensor`.
  Raises:
    TypeError: if graph is not a `tf.Graph`.
  """
  if not isinstance(graph, tf_ops.Graph):
    raise TypeError("Expected a graph, got: {}".format(type(graph)))
  ts = []
  for op in graph.get_operations():
    ts += op.outputs
  return ts


def make_list_of_t(ts, check_graph=True, allow_graph=True, ignore_ops=False):
  """Convert ts to a list of `tf.Tensor`.

  Args:
    ts: can be an iterable of `tf.Tensor`, a `tf.Graph` or a single tensor.
    check_graph: if `True` check if all the tensors belong to the same graph.
    allow_graph: if `False` a `tf.Graph` cannot be converted.
    ignore_ops: if `True`, silently ignore `tf.Operation`.
  Returns:
    A newly created list of `tf.Tensor`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor` or,
     if `check_graph` is `True`, if all the ops do not belong to the same graph.
  """
  if isinstance(ts, tf_ops.Graph):
    if allow_graph:
      return get_tensors(ts)
    raise TypeError("allow_graph is False: cannot convert a tf.Graph.")
  if not is_iterable(ts):
    ts = [ts]
  if not ts:
    return []
  if check_graph:
    check_types = None if ignore_ops else tf_ops.Tensor
    get_unique_graph(ts, check_types=check_types)
  return [t for t in ts if isinstance(t, tf_ops.Tensor)]


def get_generating_ops(ts):
  """Return all the generating ops of the tensors in `ts`.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the generating `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if `ts` cannot be converted to a list of `tf.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  return [t.op for t in ts]


def get_consuming_ops(ts):
  """Return all the consuming ops of the tensors in ts.

  Args:
    ts: a list of `tf.Tensor`
  Returns:
    A list of all the consuming `tf.Operation` of the tensors in `ts`.
  Raises:
    TypeError: if ts cannot be converted to a list of `tf.Tensor`.
  """
  ts = make_list_of_t(ts, allow_graph=False)
  ops = []
  for t in ts:
    for op in t.consumers():
      if op not in ops:
        ops.append(op)
  return ops


class ControlOutputs():
  """The control outputs topology."""

  def __init__(self, graph):
    """Create a dictionary of control-output dependencies.

    Args:
      graph: a `tf.Graph`.
    Returns:
      A dictionary where a key is a `tf.Operation` instance and the
         corresponding value is a list of all the ops which have the key
         as one of their control-input dependencies.
    Raises:
      TypeError: graph is not a `tf.Graph`.
    """
    if not isinstance(graph, tf_ops.Graph):
      raise TypeError("Expected a tf.Graph, got: {}".format(type(graph)))
    self._control_outputs = {}
    self._graph = graph
    self._version = None
    self._build()

  def update(self):
    """Update the control outputs if the graph has changed."""
    if self._version != self._graph.version:
      self._build()
    return self

  def _build(self):
    """Build the control outputs dictionary."""
    self._control_outputs.clear()
    ops = self._graph.get_operations()
    for op in ops:
      for control_input in op.control_inputs:
        if control_input not in self._control_outputs:
          self._control_outputs[control_input] = []
        if op not in self._control_outputs[control_input]:
          self._control_outputs[control_input].append(op)
    self._version = self._graph.version

  def get_all(self):
    """Get control outputs."""
    return self._control_outputs

  def get(self, op):
    """Return the control outputs of op."""
    if op in self._control_outputs:
      return self._control_outputs[op]
    return ()

  @property
  def graph(self):
    """Return graph."""
    return self._graph


def scope_finalize(scope):
  if scope and scope[-1] != "/":
    scope += "/"
  return scope


def scope_dirname(scope):
  slash = scope.rfind("/")
  if slash == -1:
    return ""
  return scope[:slash + 1]


def scope_basename(scope):
  slash = scope.rfind("/")
  if slash == -1:
    return scope
  return scope[slash + 1:]


def placeholder_name(t=None, scope=None, prefix=_DEFAULT_PLACEHOLDER_PREFIX):
  """Create placeholder name for the graph editor.

  Args:
    t: optional tensor on which the placeholder operation's name will be based
      on
    scope: absolute scope with which to prefix the placeholder's name. None
      means that the scope of t is preserved. "" means the root scope.
    prefix: placeholder name prefix.
  Returns:
    A new placeholder name prefixed by "geph". Note that "geph" stands for
      Graph Editor PlaceHolder. This convention allows to quickly identify the
      placeholder generated by the Graph Editor.
  Raises:
    TypeError: if t is not None or a tf.Tensor.
  """
  if scope is not None:
    scope = scope_finalize(scope)
  if t is not None:
    if not isinstance(t, tf_ops.Tensor):
      raise TypeError("Expected a tf.Tenfor, got: {}".format(type(t)))
    op_dirname = scope_dirname(t.op.name)
    op_basename = scope_basename(t.op.name)
    if scope is None:
      scope = op_dirname

    if op_basename.startswith("{}__".format(prefix)):
      ph_name = op_basename
    else:
      ph_name = "{}__{}_{}".format(prefix, op_basename, t.value_index)

    return scope + ph_name
  if scope is None:
    scope = ""
  return "{}{}".format(scope, prefix)


def make_placeholder_from_tensor(t,
                                 scope=None,
                                 prefix=_DEFAULT_PLACEHOLDER_PREFIX):
  """Create a `tf.compat.v1.placeholder` for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.

  Args:
    t: a `tf.Tensor` whose name will be used to create the placeholder (see
      function placeholder_name).
    scope: absolute scope within which to create the placeholder. None means
      that the scope of `t` is preserved. `""` means the root scope.
    prefix: placeholder name prefix.

  Returns:
    A newly created `tf.compat.v1.placeholder`.
  Raises:
    TypeError: if `t` is not `None` or a `tf.Tensor`.
  """
  return tf_array_ops.placeholder(
      dtype=t.dtype,
      shape=t.get_shape(),
      name=placeholder_name(t, scope=scope, prefix=prefix))


def make_placeholder_from_dtype_and_shape(dtype,
                                          shape=None,
                                          scope=None,
                                          prefix=_DEFAULT_PLACEHOLDER_PREFIX):
  """Create a tf.compat.v1.placeholder for the Graph Editor.

  Note that the correct graph scope must be set by the calling function.
  The placeholder is named using the function placeholder_name (with no
  tensor argument).

  Args:
    dtype: the tensor type.
    shape: the tensor shape (optional).
    scope: absolute scope within which to create the placeholder. None means
      that the scope of t is preserved. "" means the root scope.
    prefix: placeholder name prefix.

  Returns:
    A newly created tf.placeholder.
  """
  return tf_array_ops.placeholder(
      dtype=dtype,
      shape=shape,
      name=placeholder_name(scope=scope, prefix=prefix))


_INTERNAL_VARIABLE_RE = re.compile(r"^__\w+__$")


def get_predefined_collection_names():
  """Return all the predefined collection names."""
  return [
      getattr(tf_ops.GraphKeys, key)
      for key in dir(tf_ops.GraphKeys)
      if not _INTERNAL_VARIABLE_RE.match(key)
  ]


def find_corresponding_elem(target, dst_graph, dst_scope="", src_scope=""):
  """Find corresponding op/tensor in a different graph.

  Args:
    target: A `tf.Tensor` or a `tf.Operation` belonging to the original graph.
    dst_graph: The graph in which the corresponding graph element must be found.
    dst_scope: A scope which is prepended to the name to look for.
    src_scope: A scope which is removed from the original of `target` name.

  Returns:
    The corresponding tf.Tensor` or a `tf.Operation`.

  Raises:
    ValueError: if `src_name` does not start with `src_scope`.
    TypeError: if `target` is not a `tf.Tensor` or a `tf.Operation`
    KeyError: If the corresponding graph element cannot be found.
  """
  src_name = target.name
  if src_scope:
    src_scope = scope_finalize(src_scope)
    if not src_name.startswidth(src_scope):
      raise ValueError("{} does not start with {}".format(src_name, src_scope))
    src_name = src_name[len(src_scope):]

  dst_name = src_name
  if dst_scope:
    dst_scope = scope_finalize(dst_scope)
    dst_name = dst_scope + dst_name

  if isinstance(target, tf_ops.Tensor):
    return dst_graph.get_tensor_by_name(dst_name)
  if isinstance(target, tf_ops.Operation):
    return dst_graph.get_operation_by_name(dst_name)
  raise TypeError("Expected tf.Tensor or tf.Operation, got: %s" % type(target))


def find_corresponding(targets, dst_graph, dst_scope="", src_scope=""):
  """Find corresponding ops/tensors in a different graph.

  `targets` is a Python tree, that is, a nested structure of iterable
  (list, tupple, dictionary) whose leaves are instances of
  `tf.Tensor` or `tf.Operation`

  Args:
    targets: A Python tree containing `tf.Tensor` or `tf.Operation`
      belonging to the original graph.
    dst_graph: The graph in which the corresponding graph element must be found.
    dst_scope: A scope which is prepended to the name to look for.
    src_scope: A scope which is removed from the original of `top` name.

  Returns:
    A Python tree containin the corresponding tf.Tensor` or a `tf.Operation`.

  Raises:
    ValueError: if `src_name` does not start with `src_scope`.
    TypeError: if `top` is not a `tf.Tensor` or a `tf.Operation`
    KeyError: If the corresponding graph element cannot be found.
  """

  def func(top):
    return find_corresponding_elem(top, dst_graph, dst_scope, src_scope)

  return transform_tree(targets, func)
