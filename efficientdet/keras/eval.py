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
# ==============================================================================
"""Eval libraries."""
from absl import app
from absl import logging
import tensorflow as tf

import coco_metric
import dataloader
import hparams_config

from keras import anchors
from keras import efficientdet_keras
from keras import postprocess

def main(_):
  config = hparams_config.get_efficientdet_config('efficientdet-d0')
  config.batch_size = 8
  config.val_json_file = 'tmp/coco/annotations/instances_val2017.json'

  # dataset
  input_files = 'tmp/coco/val-00000-of-00032.tfrecord'
  is_training = False
  ds = dataloader.InputReader(
      input_files,
      is_training=is_training,
      use_fake_data=False,
      max_instances_per_image=config.max_instances_per_image)(
          config)

  # Network
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((config.batch_size, 512, 512, 3))
  model.load_weights('tmp/efficientdet-d0/model')

  evaluator = coco_metric.EvaluationMetric(
      filename=config.val_json_file)
  # compute stats for all batches.
  for images, labels in ds:
    cls_outputs, box_outputs = model(images, training=False)
    config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
    detections = postprocess.generate_detections(config, cls_outputs, box_outputs,
                                                labels['image_scales'],
                                                labels['source_ids'])
    evaluator.update_state(labels['groundtruth_data'].numpy(), detections.numpy())

  # compute the final eval results.
  metric_values = evaluator.result()
  metric_dict = {}
  for i, metric_value in enumerate(metric_values):
    metric_dict[evaluator.metric_names[i]] = metric_value
  print(metric_dict)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  app.run(main)