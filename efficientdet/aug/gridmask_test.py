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
"""GridMask Augmentation simple test."""
from absl import logging
import tensorflow.compat.v1 as tf

from aug import gridmask


class GridMaskTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_random_seed(111111)

  def test_gridmask_images(self):
    """Verify transformed image shape is valid and syntax check."""
    images = tf.random.uniform(
        shape=(512, 512, 3), minval=0, maxval=255, dtype=tf.float32)
    bboxes = tf.random.uniform(
        shape=(2, 4), minval=1, maxval=511, dtype=tf.int32)
    transform_images, _ = gridmask.gridmask(images, bboxes)
    self.assertEqual(images.shape[1], transform_images.shape[1])

  def test_gridmask_tiny_images(self):
    """Verify  transform image shape on very tiny image."""
    images = tf.zeros(shape=(4, 4, 3))
    bboxes = tf.random.uniform(
        shape=(2, 4), minval=1, maxval=511, dtype=tf.int32)
    transform_images, _ = gridmask.gridmask(images, bboxes)
    self.assertEqual(images.shape[1], transform_images.shape[1])

  def test_rectangle_image_shape(self):
    """Verify transform image shape on rectangle image."""
    images = tf.zeros(shape=(1028, 512, 3))
    bboxes = tf.random.uniform(
        shape=(2, 4), minval=1, maxval=511, dtype=tf.int32)
    transform_images, _ = gridmask.gridmask(images, bboxes)
    self.assertEqual(images.shape[1], transform_images.shape[1])


if __name__ == "__main__":
  logging.set_verbosity(logging.WARNING)
  tf.disable_eager_execution()
  tf.test.main()
