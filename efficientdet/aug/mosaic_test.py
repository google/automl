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
"""Mosaic Augmentation simple test."""
from absl import logging
import tensorflow.compat.v1 as tf

from aug import mosaic


class MosaicTest(tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.output_size = (512, 512)
    self.mosaic = mosaic.Mosaic(out_size=self.output_size)
    tf.random.set_random_seed(111111)

  def test_mosaic_boxes(self):
    """Verify num of boxes are valid and syntax check random four images."""
    images = tf.random.uniform(
        shape=(4, 512, 512, 3), minval=0, maxval=255, dtype=tf.float32)
    bboxes = tf.random.uniform(
        shape=(4, 2, 4), minval=1, maxval=511, dtype=tf.int32)
    _, mosaic_boxes = self.mosaic(images, bboxes)
    self.assertEqual(bboxes.shape[0], len(mosaic_boxes))

  def test_mosaic_tiny_images(self):
    images = tf.zeros(shape=(4, 4, 4, 3))
    bboxes = tf.random.uniform(
        shape=(4, 2, 4), minval=1, maxval=511, dtype=tf.int32)
    _, mosaic_boxes = self.mosaic(images, bboxes)
    self.assertEqual(bboxes.shape[0], len(mosaic_boxes))


if __name__ == "__main__":
  logging.set_verbosity(logging.WARNING)
  tf.disable_eager_execution()
  tf.test.main()
