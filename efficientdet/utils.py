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
"""Common utils."""

from __future__ import absolute_import
from __future__ import division
# gtype import
from __future__ import print_function

import os
import re
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from typing import Text, Tuple, Union

from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import
# pylint: disable=logging-format-interpolation


def activation_fn(features: tf.Tensor, act_type: Text):
  """Customized non-linear activation type."""
  if act_type == 'swish':
    return tf.nn.swish(features)
  elif act_type == 'swish_native':
    return features * tf.sigmoid(features)
  elif act_type == 'relu':
    return tf.nn.relu(features)
  elif act_type == 'relu6':
    return tf.nn.relu6(features)
  else:
    raise ValueError('Unsupported act_type {}'.format(act_type))


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
  """Wrap keras DepthwiseConv2D to tf.layers."""

  pass


def get_ema_vars():
  """Get all exponential moving average (ema) variables."""
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  return list(set(ema_vars))


def get_ckpt_var_map(ckpt_path, ckpt_scope, var_scope, var_exclude_expr=None):
  """Get a var map for restoring from pretrained checkpoints.

  Args:
    ckpt_path: string. A pretrained checkpoint path.
    ckpt_scope: string. Scope name for checkpoint variables.
    var_scope: string. Scope name for model variables.
    var_exclude_expr: string. A regex for excluding variables.
      This is useful for finetuning with different classes, where
      var_exclude_expr='.*class-predict.*' can be used.

  Returns:
    var_map: a dictionary from checkpoint name to model variables.
  """
  logging.info('Init model from checkpoint {}'.format(ckpt_path))
  if not ckpt_scope.endswith('/') or not var_scope.endswith('/'):
    raise ValueError('Please specific scope name ending with /')
  if ckpt_scope.startswith('/'):
    ckpt_scope = ckpt_scope[1:]
  if var_scope.startswith('/'):
    var_scope = var_scope[1:]

  var_map = {}
  # Get the list of vars to restore.
  model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)
  reader = tf.train.load_checkpoint(ckpt_path)
  ckpt_var_names = set(reader.get_variable_to_shape_map().keys())

  exclude_matcher = re.compile(var_exclude_expr) if var_exclude_expr else None
  for v in model_vars:
    if exclude_matcher and exclude_matcher.match(v.op.name):
      logging.info(
          'skip {} -- excluded by {}'.format(v.op.name, var_exclude_expr))
      continue

    if not v.op.name.startswith(var_scope):
      logging.info('skip {} -- does not match scope {}'.format(
          v.op.name, var_scope))
    ckpt_var = ckpt_scope + v.op.name[len(var_scope):]
    if ckpt_var not in ckpt_var_names:
      if v.op.name.endswith('/ExponentialMovingAverage'):
        ckpt_var = ckpt_scope + v.op.name[:-len('/ExponentialMovingAverage')]
      if ckpt_var not in ckpt_var_names:
        logging.info('skip {} ({}) -- not in ckpt'.format(v.op.name, ckpt_var))
        continue

    logging.info('Init {} from ckpt var {}'.format(v.op.name, ckpt_var))
    var_map[ckpt_var] = v
  return var_map


def get_ckpt_var_map_ema(ckpt_path, ckpt_scope, var_scope, var_exclude_expr):
  """Get a ema var map for restoring from pretrained checkpoints.

  Args:
    ckpt_path: string. A pretrained checkpoint path.
    ckpt_scope: string. Scope name for checkpoint variables.
    var_scope: string. Scope name for model variables.
    var_exclude_expr: string. A regex for excluding variables.
      This is useful for finetuning with different classes, where
      var_exclude_expr='.*class-predict.*' can be used.

  Returns:
    var_map: a dictionary from checkpoint name to model variables.
  """
  logging.info('Init model from checkpoint {}'.format(ckpt_path))
  if not ckpt_scope.endswith('/') or not var_scope.endswith('/'):
    raise ValueError('Please specific scope name ending with /')
  if ckpt_scope.startswith('/'):
    ckpt_scope = ckpt_scope[1:]
  if var_scope.startswith('/'):
    var_scope = var_scope[1:]

  var_map = {}
  # Get the list of vars to restore.
  model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)
  reader = tf.train.load_checkpoint(ckpt_path)
  ckpt_var_names = set(reader.get_variable_to_shape_map().keys())
  exclude_matcher = re.compile(var_exclude_expr) if var_exclude_expr else None
  for v in model_vars:
    if exclude_matcher and exclude_matcher.match(v.op.name):
      logging.info(
          'skip {} -- excluded by {}'.format(v.op.name, var_exclude_expr))
      continue

    if not v.op.name.startswith(var_scope):
      logging.info('skip {} -- does not match scope {}'.format(
          v.op.name, var_scope))

    if v.op.name.endswith('/ExponentialMovingAverage'):
      logging.info('skip ema var {}'.format(v.op.name))
      continue

    ckpt_var = ckpt_scope + v.op.name[len(var_scope):]
    ckpt_var_ema = ckpt_var + '/ExponentialMovingAverage'
    if ckpt_var_ema in ckpt_var_names:
      var_map[ckpt_var_ema] = v
      logging.info('Init {} from ckpt var {}'.format(v.op.name, ckpt_var_ema))
    elif ckpt_var in ckpt_var_names:
      var_map[ckpt_var] = v
      logging.info('Init {} from ckpt var {}'.format(v.op.name, ckpt_var))
    else:
      logging.info('skip {} ({}) -- not in ckpt'.format(v.op.name, ckpt_var))
  return var_map


class TpuBatchNormalization(tf.layers.BatchNormalization):
  # class TpuBatchNormalization(tf.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t, num_shards_per_group):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
      if num_shards % num_shards_per_group != 0:
        raise ValueError(
            'num_shards: %d mod shards_per_group: %d, should be 0' %
            (num_shards, num_shards_per_group))
      num_groups = num_shards // num_shards_per_group
      group_assignment = [[
          x for x in range(num_shards) if x // num_shards_per_group == y
      ] for y in range(num_groups)]
    return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
      num_shards_per_group = 1
    else:
      num_shards_per_group = max(8, num_shards // 8)
    logging.info('TpuBatchNormalization with num_shards_per_group {}'.format(
        num_shards_per_group))
    if num_shards_per_group > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = self._cross_replica_average(shard_mean, num_shards_per_group)
      group_mean_of_square = self._cross_replica_average(
          shard_mean_of_square, num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


class BatchNormalization(tf.layers.BatchNormalization):
  """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super(BatchNormalization, self).__init__(**kwargs)


def batch_norm_class(is_training, use_tpu=False,):
  if is_training and use_tpu:
    return TpuBatchNormalization
  else:
    return BatchNormalization


def tpu_batch_normalization(inputs, training=False, use_tpu=False, **kwargs):
  """A wrapper for TpuBatchNormalization."""
  layer = batch_norm_class(training, use_tpu)(**kwargs)
  return layer.apply(inputs, training=training)


def batch_norm_act(inputs,
                   is_training_bn: bool,
                   act_type: Union[Text, None],
                   init_zero: bool = False,
                   data_format: Text = 'channels_last',
                   momentum: float = 0.99,
                   epsilon: float = 1e-3,
                   use_tpu: bool = False,
                   name: Text = None):
  """Performs a batch normalization followed by a non-linear activation.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    act_type: non-linear relu function type. If None, omits the relu operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    use_tpu: `bool`, whether to use tpu version of batch norm.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tpu_batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      training=is_training_bn,
      use_tpu=use_tpu,
      gamma_initializer=gamma_initializer,
      name=name)

  if act_type:
    inputs = activation_fn(inputs, act_type)
  return inputs


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = tf.div(inputs, survival_prob) * binary_tensor
  return output


def num_params_flops(readable_format=True):
  """Return number of parameters and flops."""
  nparams = np.sum(
      [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
  options = tf.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.profiler.profile(
      tf.get_default_graph(), options=options).total_float_ops
  # We use flops to denote multiply-adds, which is counted as 2 ops in tfprof.
  flops = flops // 2
  if readable_format:
    nparams = float(nparams) * 1e-6
    flops = float(flops) * 1e-9
  return nparams, flops


conv_kernel_initializer = tf.initializers.variance_scaling()
dense_kernel_initializer = tf.initializers.variance_scaling()


def scalar(name, tensor):
  """Stores a (name, Tensor) tuple in a custom collection."""
  logging.info('Adding summary {}'.format((name, tensor)))
  tf.add_to_collection('edsummaries', (name, tf.reduce_mean(tensor)))


def get_scalar_summaries():
  """Returns the list of (name, Tensor) summaries recorded by scalar()."""
  return tf.get_collection('edsummaries')


def get_tpu_host_call(global_step, params):
  """Get TPU host call for summaries."""
  summaries = get_scalar_summaries()
  if not summaries:
    # No summaries to write.
    return None

  model_dir = params['model_dir']
  iterations_per_loop = params.get('iterations_per_loop', 100)

  def host_call_fn(global_step, *args):
    """Training host call. Creates scalar summaries for training metrics."""
    gs = global_step[0]
    with tf2.summary.create_file_writer(
        model_dir, max_queue=iterations_per_loop).as_default():
      with tf2.summary.record_if(True):
        for i in range(len(summaries)):
          name = summaries[i][0]
          tensor = args[i][0]
          tf2.summary.scalar(name, tensor, step=gs)
        return tf.summary.all_v2_summary_ops()

  reshaped_tensors = [tf.reshape(t, [1]) for _, t in summaries]
  global_step_t = tf.reshape(global_step, [1])
  return host_call_fn, [global_step_t] + reshaped_tensors


def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
  """Archive a checkpoint if the metric is better."""
  ckpt_dir, ckpt_name = os.path.split(ckpt_path)

  saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')
  saved_objective = float('-inf')
  if tf.io.gfile.exists(saved_objective_path):
    with tf.io.gfile.GFile(saved_objective_path, 'r') as f:
      saved_objective = float(f.read())
  if saved_objective > ckpt_objective:
    logging.info('Ckpt {} is worse than {}'.format(ckpt_objective,
                                                   saved_objective))
    return False

  filenames = tf.io.gfile.glob(ckpt_path + '.*')
  if filenames is None:
    logging.info('No files to copy for checkpoint {}'.format(ckpt_path))
    return False

  # clear up the backup folder.
  backup_dir = os.path.join(ckpt_dir, 'backup')
  if tf.io.gfile.exists(backup_dir):
    tf.io.gfile.rmtree(backup_dir)

  # rename the old checkpoints to backup folder.
  dst_dir = os.path.join(ckpt_dir, 'archive')
  if tf.io.gfile.exists(dst_dir):
    logging.info('mv {} to {}'.format(dst_dir, backup_dir))
    tf.io.gfile.rename(dst_dir, backup_dir)

  # Write checkpoints.
  tf.io.gfile.makedirs(dst_dir)
  for f in filenames:
    dest = os.path.join(dst_dir, os.path.basename(f))
    tf.io.gfile.copy(f, dest, overwrite=True)
  ckpt_state = tf.train.generate_checkpoint_state_proto(
      dst_dir,
      model_checkpoint_path=os.path.join(dst_dir, ckpt_name))
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
    f.write(str(ckpt_state))
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
    f.write('%s' % ckpt_eval)

  # Update the best objective.
  with tf.io.gfile.GFile(saved_objective_path, 'w') as f:
    f.write('%f' % ckpt_objective)

  logging.info('Copying checkpoint {} to {}'.format(ckpt_path, dst_dir))
  return True


def get_feat_sizes(image_size: Union[int, Tuple[int, int]], max_level: int):
  """Get feat widths and heights for all levels."""
  if isinstance(image_size, int):
    image_size = (image_size, image_size)
  feat_sizes = [{'height': image_size[0], 'width': image_size[1]}]
  feat_size = image_size
  for _ in range(1, max_level + 1):
    feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
    feat_sizes.append({'height': feat_size[0], 'width': feat_size[1]})
  return feat_sizes
