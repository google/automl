# Copyright 2021 Google Research. All Rights Reserved.
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
"""Tests for effnetv2_model."""
from absl import logging
from absl.testing import parameterized
import tensorflow.compat.v1 as tf
import effnetv2_model


class EffNetV2ModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('v1_b0', 'efficientnet-b0', 5330564),
                                  ('v1_b1', 'efficientnet-b1', 7856232),
                                  ('v1_b2', 'efficientnet-b2', 9177562),
                                  ('v1_b3', 'efficientnet-b3', 12314268),
                                  ('v1_b4', 'efficientnet-b4', 19466816),
                                  ('v1_b5', 'efficientnet-b5', 30562520),
                                  ('v1_b6', 'efficientnet-b6', 43265136))
  def test_effnetv1(self, model_name, expected_params):
    images = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    model = effnetv2_model.EffNetV2Model(model_name)
    _ = model(images)
    self.assertEqual(model.count_params(), expected_params)

  @parameterized.named_parameters(('v1-b0', 'efficientnetv2-b0', 7200312),
                                  ('v1-b1', 'efficientnetv2-b1', 8212124),
                                  ('v1-b2', 'efficientnetv2-b2', 10178374),
                                  ('v1-b3', 'efficientnetv2-b3', 14467622),
                                  ('s', 'efficientnetv2-s', 21612360),
                                  ('m', 'efficientnetv2-m', 54431388),
                                  ('l', 'efficientnetv2-l', 119027848),
                                  ('xl', 'efficientnetv2-xl', 208896832))
  def test_effnetv2(self, model_name, expected_params):
    images = tf.zeros((10, 224, 224, 3), dtype=tf.float32)
    model = effnetv2_model.EffNetV2Model(model_name)
    _ = model(images)
    self.assertEqual(model.count_params(), expected_params)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
