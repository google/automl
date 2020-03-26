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
"""Tests for Autoaugment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from aug import autoaugment


class AutoaugmentTest(tf.test.TestCase):

  def test_autoaugment_policy(self):
    # A very simple test to verify no syntax error.
    image = tf.placeholder(tf.uint8, shape=[640, 640, 3])
    bboxes = tf.placeholder(tf.float32, shape=[4, 4])
    autoaugment.distort_image_with_autoaugment(image, bboxes, 'test')


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()

