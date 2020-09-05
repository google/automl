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
"""Tests for efficientnet_builder."""
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from backbone import efficientnet_builder


class EfficientnetBuilderTest(tf.test.TestCase):

  def _test_model_params(self,
                         model_name,
                         input_size,
                         expected_params,
                         override_params=None,
                         features_only=False,
                         pooled_features_only=False):
    images = tf.zeros((1, input_size, input_size, 3), dtype=tf.float32)
    efficientnet_builder.build_model(
        images,
        model_name=model_name,
        override_params=override_params,
        training=False,
        features_only=features_only,
        pooled_features_only=pooled_features_only)
    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    self.assertEqual(num_params, expected_params)

  def test_efficientnet_b0(self):
    self._test_model_params('efficientnet-b0', 224, expected_params=5288548)

  def test_efficientnet_b1(self):
    self._test_model_params('efficientnet-b1', 240, expected_params=7794184)

  def test_efficientnet_b2(self):
    self._test_model_params('efficientnet-b2', 260, expected_params=9109994)

  def test_efficientnet_b3(self):
    self._test_model_params('efficientnet-b3', 300, expected_params=12233232)

  def test_efficientnet_b4(self):
    self._test_model_params('efficientnet-b4', 380, expected_params=19341616)

  def test_efficientnet_b5(self):
    self._test_model_params('efficientnet-b5', 456, expected_params=30389784)

  def test_efficientnet_b6(self):
    self._test_model_params('efficientnet-b6', 528, expected_params=43040704)

  def test_efficientnet_b7(self):
    self._test_model_params('efficientnet-b7', 600, expected_params=66347960)

  def test_efficientnet_b0_with_customized_num_classes(self):
    self._test_model_params(
        'efficientnet-b0',
        224,
        expected_params=4135648,
        override_params={'num_classes': 100})

  def test_efficientnet_b0_with_features_only(self):
    self._test_model_params(
        'efficientnet-b0', 224, features_only=True, expected_params=3595388)

  def test_efficientnet_b0_with_pooled_features_only(self):
    self._test_model_params(
        'efficientnet-b0',
        224,
        pooled_features_only=True,
        expected_params=4007548)

  def test_efficientnet_b0_fails_if_both_features_requested(self):
    with self.assertRaises(AssertionError):
      efficientnet_builder.build_model(
          None,
          model_name='efficientnet-b0',
          training=False,
          features_only=True,
          pooled_features_only=True)

  def test_efficientnet_b0_base(self):
    # Creates a base model using the model configuration.
    images = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    _, endpoints = efficientnet_builder.build_model_base(
        images, model_name='efficientnet-b0', training=False)

    # reduction_1 to reduction_5 should be in endpoints
    self.assertEqual(len(endpoints), 5)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  # Disable eager to allow tf.profile works for #params/#flops.
  tf.disable_eager_execution()
  tf.test.main()
