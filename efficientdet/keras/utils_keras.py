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

    super(BatchNormAct, self).__init__(name=None)

    self.act_type = act_type
    self.training = is_training_bn

    self.init_zero = init_zero
    if self.init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()

    self.data_format = data_format
    if self.data_format == 'channels_first':
      axis = 1
    else:
      axis = 3

    self.use_tpu = use_tpu

    self.layer = utils.batch_norm_class(self.training, self.use_tpu)(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        gamma_initializer=gamma_initializer,
        name=f'{name}')

  def call(self, inputs, **kwargs):
    x = self.layer(inputs, training=self.training)
    if self.act_type:
      x = utils.activation_fn(x, self.act_type)
    return x

  def get_config(self):
    base_config = super(BatchNormAct, self).get_config()

    return {
        **base_config,
        'act_type': self.act_type,
        'init_zero': self.init_zero,
        'data_format': self.data_format,
        'is_training_bn': self.training,
        'use_tpu': self.use_tpu,
    }
