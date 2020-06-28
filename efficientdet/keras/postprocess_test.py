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
"""Test for postprocess"""
from absl import logging
import tensorflow as tf

import inference
from keras import postprocess


class PostprocessTest(tf.test.TestCase):
  """A test for postprocess."""

  def setUp(self):
    self.params = {
        'min_level': 1,
        'max_level': 1,
        'aspect_ratios': [(1.0, 1.0)],
        'num_scales': 1,
        'anchor_scale': 1,
        'image_size': 8,
        'num_classes': 2,
        'data_format': 'channels_last',
        'max_detection_points': 10,
        'nms_configs': {
            'max_output_size': 2,
        }
    }

  def test_postprocess_global(self):
    """Test the postprocess with global nms."""
    tf.random.set_seed(1111)
    cls_outputs = {1: tf.random.normal([2, 4, 4, 2])}
    box_outputs = {1: tf.random.normal([2, 4, 4, 4])}
    scales = [1.0, 2.0]

    self.params['max_detection_points'] = 10
    boxes, scores, classes = postprocess.postprocess_global(
        self.params, cls_outputs, box_outputs, scales)

    self.params['disable_pyfun'] = True
    score_thresh = 0.5
    self.params['batch_size'] = len(scales)
    max_output_size = self.params['nms_configs']['max_output_size']
    legacy_detections = inference.det_post_process(self.params, cls_outputs,
                                                   box_outputs, scales,
                                                   score_thresh,
                                                   max_output_size)
    legacy_boxes = legacy_detections[:, :, 1:5]
    legacy_scores = legacy_detections[:, :, 5]
    legacy_classes = legacy_detections[:, :, 6]
    self.assertAllClose(boxes, legacy_boxes)
    self.assertAllClose(scores, legacy_scores)
    self.assertAllClose(classes, legacy_classes)

  def test_postprocess_per_class(self):
    """Test postprocess with per class nms."""
    tf.random.set_seed(1111)
    cls_outputs = {1: tf.random.normal([2, 4, 4, 2])}
    box_outputs = {1: tf.random.normal([2, 4, 4, 4])}
    scales = [1.0, 2.0]
    ids = [0, 1]

    self.params['max_detection_points'] = 10
    outputs = postprocess.generate_detections(self.params, cls_outputs,
                                              box_outputs, scales, ids)

    self.params['disable_pyfun'] = False
    score_thresh = 0.5
    max_output_size = self.params['nms_configs']['max_output_size']
    self.params['batch_size'] = len(scales)
    legacy_outputs = inference.det_post_process(self.params, cls_outputs,
                                                box_outputs, scales,
                                                score_thresh, max_output_size)
    self.assertAllClose(outputs, legacy_outputs)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
