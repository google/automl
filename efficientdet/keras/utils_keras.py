# Lint as: python3
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
"""Common keras utils."""
# gtype import

from typing import Text, Union
import tensorflow as tf

import utils


class ActivationFn(tf.keras.layers.Layer):
  """Activation function."""

  def __init__(self, act_type: Text, name='activation_fn', **kwargs):

    super(ActivationFn, self).__init__()

    self.act_type = act_type

    if act_type == 'swish':
      self.act = tf.nn.swish
    elif act_type == 'swish_native':
      self.act = lambda x: x * tf.sigmoid(x)
    elif act_type == 'relu':
      self.act = tf.nn.relu
    elif act_type == 'relu6':
      self.act = tf.nn.relu6
    else:
      raise ValueError('Unsupported act_type {}'.format(act_type))

  def call(self, inputs, **kwargs):
    # return features
    return self.act(inputs)

  def get_config(self):
    base_config = super(ActivationFn, self).get_config()

    return {**base_config, 'act_type': self.act_type}


class BatchNormAct(tf.keras.layers.Layer):
  """A layer for batch norm and activation."""

  def __init__(self,
               is_training_bn: bool,
               act_type: Union[Text, None],
               init_zero: bool = False,
               data_format: Text = 'channels_last',
               momentum: float = 0.99,
               epsilon: float = 1e-3,
               use_tpu: bool = False,
               name: Text = None):

    super(BatchNormAct, self).__init__()

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
      self.layer = utils.TpuBatchNormalization(
          axis=self.axis,
          momentum=momentum,
          epsilon=epsilon,
          center=True,
          scale=True,
          gamma_initializer=self.gamma_initializer,
          name=f'{name}')
    else:
      self.layer = utils.BatchNormalization(
          axis=self.axis,
          momentum=momentum,
          epsilon=epsilon,
          center=True,
          scale=True,
          gamma_initializer=self.gamma_initializer,
          name=f'{name}')

    self.act = ActivationFn(act_type)

  def call(self, inputs, **kwargs):
    x = self.layer(inputs, training=self.training)
    x = self.act(x)
    return x


class DropConnect(tf.keras.layers.Layer):
  """Drop connect for stocastic depth."""

  def __init__(self, survival_prob, name='drop_connect'):
    super(DropConnect, self).__init__(name=name)
    self.survival_prob = survival_prob

  def call(self, inputs, **kwargs):
    # Compute tensor.
    batch_size = tf.shape(inputs)[0]
    random_tensor = self.survival_prob
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1],
                                       dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    # Unlike conventional way that multiply survival_prob at test time, here we
    # divide survival_prob at training time, such that no addition compute is
    # needed at test time.
    output = tf.math.divide(inputs, self.survival_prob) * binary_tensor
    return output
