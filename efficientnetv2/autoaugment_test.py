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
"""Tests for autoaugment."""
import tensorflow.compat.v1 as tf
import autoaugment


class AutoaugmentTest(tf.test.TestCase):

  def test_autoaugment(self):
    """Smoke test to be sure no syntax errors."""
    image = tf.zeros((224, 224, 3), dtype=tf.uint8)
    aug_image = autoaugment.distort_image_with_autoaugment(image, 'v0')
    self.assertEqual((224, 224, 3), aug_image.shape)

  def test_randaug(self):
    """Smoke test to be sure no syntax errors."""
    num_layers = 2
    magnitude = 15
    image = tf.zeros((224, 224, 3), dtype=tf.uint8)
    aug_image = autoaugment.distort_image_with_randaugment(
        image, num_layers, magnitude)
    self.assertEqual((224, 224, 3), aug_image.shape)


if __name__ == '__main__':
  tf.test.main()
