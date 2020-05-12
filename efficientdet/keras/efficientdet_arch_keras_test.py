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
"""Tests for efficientdet_arch."""

from __future__ import absolute_import, division, print_function

import tensorflow.compat.v1 as tf

import efficientdet_arch
import keras.efficientdet_arch_keras as keras_arch


class EfficientDetVariablesNamesTest(tf.test.TestCase):
  """ Test to see if keras does not break variables names"""

  def build_model(self, keras=False):
    """

    :param keras: If using keras model
    :return: all variables names of the model
    """
    tf.compat.v1.reset_default_graph()
    inputs_shape = [1, 512, 512, 3]
    inputs = tf.ones(shape=inputs_shape, name='input', dtype=tf.float32)
    if not keras:
      efficientdet_arch.efficientdet(
          inputs,
          model_name='efficientdet-d0',
          is_training_bn=False,
          image_size=512)
    else:
      keras_arch.efficientdet(
          inputs,
          model_name='efficientdet-d0',
          is_training_bn=False,
          image_size=512)
    return [n.name for n in tf.global_variables()]

  def test_graph_variables_name_compatibility(self):
    """
    Assert that both variables set are substet of each other (ie keras & tf)
    """
    legacy_names = self.build_model(False)
    keras_names = self.build_model(True)

    self.assertContainsSubset(keras_names, legacy_names)
    self.assertContainsSubset(legacy_names, keras_names)


if __name__ == '__main__':
  tf.disable_eager_execution()
  tf.test.main()
