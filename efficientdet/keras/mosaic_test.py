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
# =============================================================================
"""Mosaic data augmentation test case"""
from keras.mosaic import mosaic
import tensorflow as tf


class MosaicTest(tf.test.TestCase):

  def test_generate_image(self):
    images = [tf.ones([512, 512, 3])] * 4
    boxes = [tf.constant([[211, 263, 339, 324], [264, 165, 372, 253]])] * 4
    classes = [[1, 2]] * 4
    size = [512, 512]
    image, boxes, classes = mosaic(images, boxes, classes, size)
    self.assertEqual(image.shape, [512, 512, 3])
    self.assertEqual(boxes.shape, [8, 4])
    self.assertEqual(classes.shape, [8])


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
