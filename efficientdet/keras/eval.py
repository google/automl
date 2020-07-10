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
from absl import flags
from absl import logging
import tensorflow as tf

import coco_metric
import dataloader
import hparams_config

from keras import anchors
from keras import efficientdet_keras
from keras import postprocess

flags.DEFINE_string('validation_file_pattern', None,
                    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file', None,
    'COCO validation JSON containing golden bounding boxes.')
flags.DEFINE_string('model_name', 'efficientdet-d0',
                    'Model name: the efficientdet model to use.')
flags.DEFINE_string('checkpoint', None, 'Location of the checkpoint to evaluate.')

FLAGS = flags.FLAGS

def main(_):
  if FLAGS.validation_file_pattern is None:
    raise RuntimeError('You must specify --validation_file_pattern '
                        'for evaluation.')
  if FLAGS.val_json_file is None:
    raise RuntimeError('You must specify --val_json_file '
                        'for evaluation.')
  if FLAGS.checkpoint is None:
    raise RuntimeError('You must specify --checkpoint '
                        'for evaluation.')

  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.batch_size = 8
  config.val_json_file = FLAGS.val_json_file

  # dataset
  is_training = False
  ds = dataloader.InputReader(
      FLAGS.validation_file_pattern,
      is_training=is_training,
      use_fake_data=False,
      max_instances_per_image=config.max_instances_per_image)(
          config)

  # Network
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((config.batch_size, 512, 512, 3))
  model.load_weights(FLAGS.checkpoint)

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