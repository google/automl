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
"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import os
from absl import flags
from absl import logging

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class EvaluationMetric(object):
  """COCO evaluation metric class."""

  def __init__(self, filename=None, testdev_dir=None):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruth and runs COCO evaluation.

    Args:
      filename: Ground truth JSON file name. If filename is None, use
        groundtruth data passed from the dataloader for evaluation. filename is
        ignored if testdev_dir is not None.
      testdev_dir: folder name for testdev data. If None, run eval without
        groundtruth, and filename will be ignored.
    """
    if filename:
      self.coco_gt = COCO(filename)
    self.filename = filename
    self.testdev_dir = testdev_dir
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                         'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    self._reset()

  def _reset(self):
    """Reset COCO API object."""
    if self.filename is None:
      self.coco_gt = COCO()
    self.detections = []
    self.dataset = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    self.image_id = 1
    self.annotation_id = 1
    self.category_ids = []

  def estimator_metric_fn(self, detections, groundtruth_data):
    """Constructs the metric function for tf.TPUEstimator.

    For each metric, we return the evaluation op and an update op; the update op
    is shared across all metrics and simply appends the set of detections to the
    `self.detections` list. The metric op is invoked after all examples have
    been seen and computes the aggregate COCO metrics. Please find details API
    in: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/MetricSpec
    Args:
      detections: Detection results in a tensor with each row representing
        [image_id, x, y, width, height, score, class]
      groundtruth_data: Groundtruth annotations in a tensor with each row
        representing [y1, x1, y2, x2, is_crowd, area, class].
    Returns:
      metrics_dict: A dictionary mapping from evaluation name to a tuple of
        operations (`metric_op`, `update_op`). `update_op` appends the
        detections for the metric to the `self.detections` list.
    """

    def _evaluate():
      """Evaluates with detections from all images with COCO API.

      Returns:
        coco_metric: float numpy array with shape [12] representing the
          coco-style evaluation metrics.
      """
      if self.filename is None:
        self.coco_gt.dataset = self.dataset
        self.coco_gt.createIndex()

      if self.testdev_dir:
        # Run on test-dev dataset.
        box_result_list = []
        for det in self.detections:
          box_result_list.append({
              'image_id': int(det[0]),
              'category_id': int(det[6]),
              'bbox': np.around(
                  det[1:5].astype(np.float64), decimals=2).tolist(),
              'score': float(np.around(det[5], decimals=3)),
          })
        json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
        output_path = os.path.join(self.testdev_dir,
                                   'detections_test-dev2017_test_results.json')
        logging.info('Writing output json file to: %s', output_path)
        with tf.io.gfile.GFile(output_path, 'w') as fid:
          json.dump(box_result_list, fid)

        self._reset()
        return np.array([0.], dtype=np.float32)
      else:
        # Run on validation dataset.
        detections = np.array(self.detections)
        image_ids = list(set(detections[:, 0]))
        coco_dt = self.coco_gt.loadRes(detections)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats
        # clean self.detections after evaluation is done.
        # this makes sure the next evaluation will start with an empty list of
        # self.detections.
        self._reset()
        return np.array(coco_metrics, dtype=np.float32)

    def _update_op(detections, groundtruth_data):
      """Update detection results and groundtruth data.

      Append detection results to self.detections to aggregate results from
      all validation set. The groundtruth_data is parsed and added into a
      dictionary with the same format as COCO dataset, which can be used for
      evaluation.

      Args:
       detections: Detection results in a tensor with each row representing
         [image_id, x, y, width, height, score, class].
       groundtruth_data: Groundtruth annotations in a tensor with each row
         representing [y1, x1, y2, x2, is_crowd, area, class].
      """
      for i in range(len(detections)):
        # Filter out detections with predicted class label = -1.
        indices = np.where(detections[i, :, -1] > -1)[0]
        detections[i] = detections[i, indices]
        if detections[i].shape[0] == 0:
          continue
        # Append groundtruth annotations to create COCO dataset object.
        # Add images.
        image_id = detections[i][0, 0]
        if image_id == -1:
          image_id = self.image_id
        detections[i][:, 0] = image_id
        self.detections.extend(detections[i])

        if self.testdev_dir:
          # Skip annotation for test-dev case.
          self.image_id += 1
          continue

        self.dataset['images'].append({
            'id': int(image_id),
        })

        # Add annotations.
        indices = np.where(groundtruth_data[i, :, -1] > -1)[0]
        for data in groundtruth_data[i, indices]:
          box = data[0:4]
          is_crowd = data[4]
          area = data[5]
          category_id = data[6]
          if category_id < 0:
            break
          if area == -1:
            area = (box[3] - box[1]) * (box[2] - box[0])
          self.dataset['annotations'].append({
              'id': int(self.annotation_id),
              'image_id': int(image_id),
              'category_id': int(category_id),
              'bbox': [box[1], box[0], box[3] - box[1], box[2] - box[0]],
              'area': area,
              'iscrowd': int(is_crowd)
          })
          self.annotation_id += 1
          self.category_ids.append(category_id)
        self.image_id += 1
      self.category_ids = list(set(self.category_ids))
      self.dataset['categories'] = [
          {'id': int(category_id)} for category_id in self.category_ids
      ]

    with tf.name_scope('coco_metric'):
      if self.testdev_dir:
        update_op = tf.py_func(_update_op, [detections, groundtruth_data], [])
        metrics = tf.py_func(_evaluate, [], tf.float32)
        metrics_dict = {'AP': (metrics, update_op)}
        return metrics_dict
      else:
        update_op = tf.py_func(_update_op, [detections, groundtruth_data], [])
        metrics = tf.py_func(_evaluate, [], tf.float32)
        metrics_dict = {}
        for i, name in enumerate(self.metric_names):
          metrics_dict[name] = (metrics[i], update_op)
        return metrics_dict
