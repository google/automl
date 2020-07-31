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
from typing import Text
import tensorflow as tf

import utils


def build_batch_norm(is_training_bn: bool,
                     beta_initializer: Text = 'zeros',
                     gamma_initializer: Text = 'ones',
                     data_format: Text = 'channels_last',
                     momentum: float = 0.99,
                     epsilon: float = 1e-3,
                     strategy: Text = None,
                     name: Text = 'tpu_batch_normalization'):
  """Build a batch normalization layer.

  Args:
    is_training_bn: `bool` for whether the model is training.
    beta_initializer: `str`, beta initializer.
    gamma_initializer: `str`, gamma initializer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    strategy: `str`, whether to use tpu, horovod or other version of batch norm.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  axis = 1 if data_format == 'channels_first' else -1
  if is_training_bn and strategy in ('gpus',):
    batch_norm_class = tf.keras.layers.experimental.SyncBatchNormalization
  elif is_training_bn and strategy in ('tpu',):
    # TODO(tanmingxing): compare them on TPU.
    batch_norm_class = utils.batch_norm_class(is_training_bn, strategy)
  else:
    batch_norm_class = tf.keras.layers.BatchNormalization

  bn_layer = batch_norm_class(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      name=name)

  return bn_layer
