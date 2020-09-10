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
"""Tests for coco_metric."""

from absl import logging
import tensorflow.compat.v1 as tf
import coco_metric


class CocoMetricTest(tf.test.TestCase):

  def setUp(self):
    super(CocoMetricTest, self).setUp()
    # [y1, x1, y2, x2, is_crowd, area, class], in image coords.
    self.groundtruth_data = tf.constant([[
        [10.0, 10.0, 20.0, 20.0, 0.0, 100.0, 1],
        [10.0, 10.0, 30.0, 15.0, 0.0, 100.0, 2],
        [30.0, 30.0, 40.0, 50.0, 0.0, 100.0, 3]
    ]], dtype=tf.float32)
    # [image_id, x, y, width, height, score, class]
    self.detections = tf.constant([[
        [1.0, 10.0, 10.0, 10.0, 10.0, 0.6, 1],
        [1.0, 10.0, 10.0, 5.0, 20.0, 0.5, 2]
    ]], dtype=tf.float32)
    self.class_labels = {1: 'car', 2: 'truck', 3: 'bicycle'}

  def test_mAP(self):

    eval_metric = coco_metric.EvaluationMetric(label_map=self.class_labels)
    coco_metrics = eval_metric.estimator_metric_fn(self.detections,
                                                   self.groundtruth_data)
    self.assertEqual(len(coco_metrics.keys()), 15)
    self.assertAllClose(coco_metrics['AP'][0], 2.0/3.0)
    self.assertAllClose(coco_metrics['AP_/car'][0], 1.0)
    self.assertAllClose(coco_metrics['AP_/truck'][0], 1.0)
    self.assertAllClose(coco_metrics['AP_/bicycle'][0], 0.0)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
