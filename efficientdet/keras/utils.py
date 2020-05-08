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

import contextlib
import os
import re
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from typing import Text, Tuple, Union

from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import
# pylint: disable=logging-format-interpolation

class Activation_fn(tf.keras.layers.Layer):
  def __init__(self, act_type: Text, name='activation_fn'):

    super(Activation_fn, self).__init__(name=name)

    if act_type == 'swish':
      self.act = tf.keras.layers.Lambda(lambda x: tf.nn.swish(x))
    elif act_type == 'swish_native':
      self.act = tf.keras.layers.Lambda(lambda x: x * tf.sigmoid(x))
    elif act_type == 'relu':
      self.act = tf.keras.layers.Lambda(lambda x: tf.nn.relu(x))
    elif act_type == 'relu6':
      self.act = tf.keras.layers.Lambda(lambda x: tf.nn.relu6(x))
    else:
      raise ValueError('Unsupported act_type {}'.format(act_type))

    def call(self, features: tf.Tensor):
      return self.act(features)

class TpuBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
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

  def call(self, *args, **kwargs):
    outputs = super(TpuBatchNormalization, self).call(*args, **kwargs)
    # A temporary hack for tf1 compatibility with keras batch norm.
    for u in self.updates:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    return outputs


class BatchNormalization(tf.keras.layers.BatchNormalization):
  """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super(BatchNormalization, self).__init__(**kwargs)

  def call(self, *args, **kwargs):
    outputs = super(BatchNormalization, self).call(*args, **kwargs)
    # A temporary hack for tf1 compatibility with keras batch norm.
    for u in self.updates:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    return outputs


class Batch_norm_act(tf.keras.layers.Layer):
  def __init__(self,
               is_training_bn: bool,
               act_type: Union[Text, None],
               init_zero: bool = False,
               data_format: Text = 'channels_last',
               momentum: float = 0.99,
               epsilon: float = 1e-3,
               use_tpu: bool = False,
               name: Text = None
               ):

    super(Batch_norm_act, self).__init__(name='batch_norm_act')

    self.act_type = act_type
    self.training = is_training_bn

    if init_zero:
      self.gamma_initializer = tf.zeros_initializer()
    else:
      self.gamma_initializer = tf.ones_initializer()

    if data_format == 'channels_first':
      self.axis = 1
    else:
      self.axis = 3

    if is_training_bn and use_tpu:
      self.layer = TpuBatchNormalization(axis=self.axis,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         center=True,
                                         scale=True,
                                         gamma_initializer=self.gamma_initializer,
                                         name=name)
    else:
      self.layer = BatchNormalization(axis=self.axis,
                                      momentum=momentum,
                                      epsilon=epsilon,
                                      center=True,
                                      scale=True,
                                      gamma_initializer=self.gamma_initializer,
                                      name=name)

    if self.act_type:
      self.act = Activation_fn(act_type)

  def call(self, inputs, **kwargs):
    x = self.layer.apply(inputs, training=self.training)
    if self.act_type:
      x = self.act.call(x)
    return x


class Drop_connect(tf.keras.layers.Layer):
  def __init__(self, survival_prob, name='drop_connect'):

    super(Drop_connect, self).__init__(name=name)
    self.survival_prob = survival_prob


    def call(self, inputs: tf.Tensor):
      # Compute tensor.
      batch_size = tf.shape(inputs)[0]
      random_tensor = self.survival_prob
      random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
      binary_tensor = tf.floor(random_tensor)
      # Unlike conventional way that multiply survival_prob at test time, here we
      # divide survival_prob at training time, such that no addition compute is
      # needed at test time.
      output = tf.div(inputs, self.survival_prob) * binary_tensor
      return output