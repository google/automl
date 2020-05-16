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
"""Tests for utils_keras."""
import tensorflow as tf

import utils
from keras import utils_keras


class EfficientDetKerasActivationTest(tf.test.TestCase):

  def test_activation_compatibility(self):

    for act_type in ['swish', 'swish_native', 'relu', 'relu6']:
      act = utils_keras.ActivationFn(act_type)
      for i in range(-2, 2):
        i = float(i)
        self.assertEqual(
            utils.activation_fn(i, act_type).numpy(),
            act.call(i).numpy())


class EfficientDetKerasBatchNormTest(tf.test.TestCase):

  def test_batchnorm_compatibility(self):
    x = tf.Variable(tf.ones((4, 1, 1, 1)) * [[1.0], [2.0], [4.0], [8.0]])
    for act_type in ['swish', 'swish_native', 'relu', 'relu6']:
      bna = utils_keras.BatchNormAct(is_training_bn=False, act_type=act_type)
      self.assertEqual(
          tf.reduce_sum(
              utils.batch_norm_act(x, is_training_bn=False,
                                   act_type=act_type).numpy()),
          tf.reduce_sum(bna.call(x).numpy()))


if __name__ == '__main__':
  tf.test.main()
