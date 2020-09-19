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
"""Test for postprocess."""
from absl import logging
import tensorflow as tf

from keras import postprocess


class PostprocessTest(tf.test.TestCase):
  """A test for postprocess."""

  def setUp(self):
    super().setUp()
    self.params = {
        'min_level': 1,
        'max_level': 2,
        'aspect_ratios': [1.0],
        'num_scales': 1,
        'anchor_scale': 1,
        'image_size': 8,
        'num_classes': 2,
        'data_format': 'channels_last',
        'max_detection_points': 10,
        'nms_configs': {
            'method': 'hard',
            'iou_thresh': None,
            'score_thresh': None,
            'sigma': None,
            'max_nms_inputs': 0,
            'max_output_size': 2,
        }
    }

  def test_postprocess_global(self):
    """Test the postprocess with global nms."""
    tf.random.set_seed(1111)
    cls_outputs = {
        1: tf.random.normal([2, 4, 4, 2]),
        2: tf.random.normal([2, 2, 2, 2])
    }
    box_outputs = {
        1: tf.random.normal([2, 4, 4, 4]),
        2: tf.random.normal([2, 2, 2, 4])
    }
    cls_outputs_list = [cls_outputs[1], cls_outputs[2]]
    box_outputs_list = [box_outputs[1], box_outputs[2]]
    scales = [1.0, 2.0]

    self.params['max_detection_points'] = 10
    _, scores, classes, valid_len = postprocess.postprocess_global(
        self.params, cls_outputs_list, box_outputs_list, scales)
    self.assertAllClose(valid_len, [2, 2])
    self.assertAllClose(classes.numpy(), [[2., 1.], [1., 2.]])
    self.assertAllClose(scores.numpy(),
                        [[0.90157586, 0.88812476], [0.88454413, 0.8158828]])

  def test_postprocess_per_class_numpy_nms(self):
    """Test postprocess with per class nms using the numpy nms."""
    tf.random.set_seed(1111)
    cls_outputs = {
        1: tf.random.normal([2, 4, 4, 2]),
        2: tf.random.normal([2, 2, 2, 2])
    }
    box_outputs = {
        1: tf.random.normal([2, 4, 4, 4]),
        2: tf.random.normal([2, 2, 2, 4])
    }
    cls_outputs_list = [cls_outputs[1], cls_outputs[2]]
    box_outputs_list = [box_outputs[1], box_outputs[2]]
    scales = [1.0, 2.0]
    ids = [0, 1]

    self.params['max_detection_points'] = 10
    outputs = postprocess.generate_detections(self.params, cls_outputs_list,
                                              box_outputs_list, scales, ids)
    self.assertAllClose(
        outputs.numpy(),
        [[[0., -1.177383, 1.793507, 8.340945, 4.418388, 0.901576, 2.],
          [0., 5.676410, 6.102146, 7.785691, 8.537168, 0.888125, 1.]],
         [[1., 5.885427, 13.529362, 11.410081, 14.154047, 0.884544, 1.],
          [1., 8.145872, -9.660868, 14.173973, 10.41237, 0.815883, 2.]]])

    outputs_flipped = postprocess.generate_detections(self.params,
                                                      cls_outputs_list,
                                                      box_outputs_list, scales,
                                                      ids, True)
    self.assertAllClose(
        outputs_flipped.numpy(),
        [[[0., -0.340945, 1.793507, 9.177383, 4.418388, 0.901576, 2.],
          [0., 0.214309, 6.102146, 2.32359, 8.537168, 0.888125, 1.]],
         [[1., 4.589919, 13.529362, 10.114573, 14.154047, 0.884544, 1.],
          [1., 1.826027, -9.660868, 7.854128, 10.41237, 0.815883, 2.]]])

  def test_postprocess_per_class_tf_nms(self):
    """Test postprocess with per class nms using the tensorflow nms."""
    tf.random.set_seed(1111)
    cls_outputs = {
        1: tf.random.normal([2, 4, 4, 2]),
        2: tf.random.normal([2, 2, 2, 2])
    }
    box_outputs = {
        1: tf.random.normal([2, 4, 4, 4]),
        2: tf.random.normal([2, 2, 2, 4])
    }
    cls_outputs_list = [cls_outputs[1], cls_outputs[2]]
    box_outputs_list = [box_outputs[1], box_outputs[2]]
    scales = [1.0, 2.0]
    ids = [0, 1]

    self.params['max_detection_points'] = 10
    self.params['nms_configs']['pyfunc'] = False
    outputs = postprocess.generate_detections(self.params, cls_outputs_list,
                                              box_outputs_list, scales, ids)
    self.assertAllClose(
        outputs.numpy(),
        [[[0., -1.177383, 1.793507, 8.340945, 4.418388, 0.901576, 2.],
          [0., 5.676410, 6.102146, 7.785691, 8.537168, 0.888125, 1.]],
         [[1., 5.885427, 13.529362, 11.410081, 14.154047, 0.884544, 1.],
          [1., 8.145872, -9.660868, 14.173973, 10.41237, 0.815883, 2.]]])

    outputs_flipped = postprocess.generate_detections(self.params,
                                                      cls_outputs_list,
                                                      box_outputs_list, scales,
                                                      ids, True)
    self.assertAllClose(
        outputs_flipped.numpy(),
        [[[0., -0.340945, 1.793507, 9.177383, 4.418388, 0.901576, 2.],
          [0., 0.214309, 6.102146, 2.32359, 8.537168, 0.888125, 1.]],
         [[1., 4.589919, 13.529362, 10.114573, 14.154047, 0.884544, 1.],
          [1., 1.826027, -9.660868, 7.854128, 10.41237, 0.815883, 2.]]])

  def test_transform_detections(self):
    corners = tf.constant(
        [[[0., -1.177383, 1.793507, 8.340945, 4.418388, 0.901576, 2.],
          [0., 5.676410, 6.102146, 7.785691, 8.537168, 0.888125, 1.]],
         [[1., 5.885427, 13.529362, 11.410081, 14.154047, 0.884544, 1.],
          [1., 8.145872, -9.660868, 14.173973, 10.41237, 0.815883, 2.]]])

    corner_plus_area = postprocess.transform_detections(corners)

    self.assertAllClose(
        corner_plus_area.numpy(),
        [[[0., -1.177383, 1.793507, 9.518328, 2.624881, 0.901576, 2.],
          [0., 5.676410, 6.102146, 2.109282, 2.435021, 0.888125, 1.]],
         [[1., 5.885427, 13.529362, 5.524654, 0.624685, 0.884544, 1.],
          [1., 8.145872, -9.660868, 6.028101, 20.073238, 0.815883, 2.]]])

  def test_postprocess_combined(self):
    """Test postprocess with per class nms."""
    tf.random.set_seed(1111)
    cls_outputs = {
        1: tf.random.normal([2, 4, 4, 2]),
        2: tf.random.normal([2, 2, 2, 2])
    }
    box_outputs = {
        1: tf.random.normal([2, 4, 4, 4]),
        2: tf.random.normal([2, 2, 2, 4])
    }
    cls_outputs_list = [cls_outputs[1], cls_outputs[2]]
    box_outputs_list = [box_outputs[1], box_outputs[2]]
    scales = [1.0, 2.0]

    self.params['max_detection_points'] = 10
    _, scores, classes, valid_len = postprocess.postprocess_combined(
        self.params, cls_outputs_list, box_outputs_list, scales)
    self.assertAllClose(valid_len, [2, 2])
    self.assertAllClose(classes.numpy(), [[2., 1.], [1., 2.]])
    self.assertAllClose(scores.numpy(),
                        [[0.90157586, 0.88812476], [0.88454413, 0.8158828]])


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
