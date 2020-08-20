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
"""
Mosaic Augmentation simple test.
"""
from absl import logging
import tensorflow as tf

from aug.mosaic import Mosaic


class MosaicTest(tf.test.TestCase):
    def test_mosaic_image(self):
        # A very simple test to verify moisac image is excepted with output_size.
        tf.random.set_seed(111111)
        output_size = (512, 512)
        mosaic = Mosaic(out_size=output_size)
        # random four images
        images = tf.random.uniform(
            shape=(4, 512, 512, 3), minval=0, maxval=255, dtype=tf.float32
        )
        bboxes = tf.random.uniform(
            shape=(4, 5, 4), minval=1, maxval=511, dtype=tf.int32
        )
        mosaic_image, mosaic_boxes = mosaic(images, bboxes)
        mosaic_height = mosaic_image.shape[0]
        mosaic_width= mosaic_image.shape[1]
        self.assertEqual(mosaic_height,output_size[0])
        self.assertEqual(mosaic_width,output_size[1])



if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
