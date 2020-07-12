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

import coco_metric
import dataloader
import hparams_config

from keras import anchors
from keras import efficientdet_keras
from keras import postprocess

import tensorflow as tf

flags.DEFINE_string('val_file_pattern', None,
                    'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_string(
    'val_json_file', None,
    'Groudtruth file, e.g. annotations/instances_val2017.json.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.55


def detection_iou(d1, d2) -> float:
  # [id, x, y, w, h, score, class]
  A = [d1[1], d1[2], d1[1] + d1[3], d1[2] + d1[4]]
  B = [d2[1], d2[2], d2[1] + d2[3], d2[2] + d2[4]]
  xA = max(A[0], B[0])
  yA = max(A[1], B[1])
  xB = min(A[2], B[2])
  yB = min(A[3], B[3])

  # compute the area of intersection rectangle
  interArea = max(0, xB - xA) * max(0, yB - yA)

  if interArea == 0:
    return 0.0

  # compute the area of both the prediction and ground-truth rectangles
  boxAArea = (A[2] - A[0]) * (A[3] - A[1])
  boxBArea = (B[2] - B[0]) * (B[3] - B[1])

  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def find_matching_cluster(clusters, box):
  best_iou = IOU_THRESHOLD
  best_index = -1
  for i, c in enumerate(clusters):
    if box[6] != c[6]:
      continue
    iou = detection_iou(c, box)
    if iou > best_iou:
      best_index = i
      best_iou = iou

  return best_index


def average_detections(detections):
  detections = tf.stack(detections)
  return [
      detections[0][0],
      tf.math.reduce_mean(detections[:, 1]),
      tf.math.reduce_mean(detections[:, 2]),
      tf.math.reduce_mean(detections[:, 3]),
      tf.math.reduce_mean(detections[:, 4]),
      tf.math.reduce_mean(detections[:, 5]),
      detections[0][6],
  ]


def ensemble_boxes(detections):
  # [id, x, y, w, h, score, class]
  clusters = []
  cluster_averages = []

  # cluster the detections
  for d in detections:
    cluster_index = find_matching_cluster(cluster_averages, d)
    if cluster_index == -1:
      clusters.append([d])
      cluster_averages.append(d)
    else:
      clusters[cluster_index].append(d)
      cluster_averages[cluster_index] = average_detections(
          clusters[cluster_index])

  filtered_clusters = []
  for c, ca in zip(clusters, cluster_averages):
    #   if not len(c) == 1:
    filtered_clusters.append(ca)
  filtered_clusters.sort(reverse=True, key=lambda d: d[5])
  return tf.stack(filtered_clusters)


def main(_):
  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  config.batch_size = FLAGS.batch_size
  config.val_json_file = FLAGS.val_json_file

  # dataset
  ds = dataloader.InputReader(
      FLAGS.val_file_pattern,
      is_training=False,
      use_fake_data=False,
      max_instances_per_image=config.max_instances_per_image)(
          config)

  # Network
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((config.batch_size, 512, 512, 3))
  model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))

  evaluator = coco_metric.EvaluationMetric(filename=config.val_json_file)

  # compute stats for all batches.
  for images, labels in ds:
    config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS

    cls_outputs, box_outputs = model(images, training=False)
    detections = postprocess.generate_detections(config, cls_outputs,
                                                 box_outputs,
                                                 labels['image_scales'],
                                                 labels['source_ids'], False)
    images_flipped = tf.image.flip_left_right(images)
    cls_outputs_flipped, box_outputs_flipped = model(
        images_flipped, training=False)
    detections_flipped = postprocess.generate_detections(
        config, cls_outputs_flipped, box_outputs_flipped,
        labels['image_scales'], labels['source_ids'], True)

    for d, df in zip(detections, detections_flipped):
      combined_detections = ensemble_boxes(tf.concat([d, df], 0))
      combined_detections = tf.stack([combined_detections])
      evaluator.update_state(labels['groundtruth_data'].numpy(),
                             combined_detections.numpy())

    # print(len(detections[0]))
    # print()

    # for d in detections[0][:10]:
    #   print(d[5].numpy(), d[6].numpy())

    # print()
    # break

  # compute the final eval results.
  metric_values = evaluator.result()
  metric_dict = {}
  for i, metric_value in enumerate(metric_values):
    metric_dict[evaluator.metric_names[i]] = metric_value
  print(metric_dict)


if __name__ == '__main__':
  flags.mark_flag_as_required('val_file_pattern')
  flags.mark_flag_as_required('val_json_file')
  flags.mark_flag_as_required('model_dir')
  logging.set_verbosity(logging.WARNING)
  app.run(main)
