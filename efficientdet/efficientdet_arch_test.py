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

import tensorflow.compat.v1 as tf

from efficientdet_arch import nearest_upsampling

class EfficientnetBuilderTest(tf.test.TestCase):
  def test_nearest_upsampling(self):
    data = tf.random.uniform([1, 416, 416, 3])
    height_scale = 6
    weight_scale = 4
    data_t = tf.transpose(data, [0, 3, 1, 2])
    c_l = nearest_upsampling(data, height_scale, weight_scale, 'channels_last')
    c_f = nearest_upsampling(data_t, height_scale, weight_scale, 'channels_first')
    self.assertAllClose(c_l, tf.transpose(c_f, [0, 2, 3, 1]))


if __name__ == '__main__':
  tf.test.main()
