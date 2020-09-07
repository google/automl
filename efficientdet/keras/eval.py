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
import utils

from keras import anchors
from keras import efficientdet_keras
from keras import label_util
from keras import postprocess
from keras import wbf

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_enum('strategy', None, ['tpu', 'gpus', ''],
                  'Training: gpus for multi-gpu, if None, use TF default.')

flags.DEFINE_string('val_file_pattern', None,
                    'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_integer('eval_samples', None, 'The number of samples for '
                     'evaluation.')
flags.DEFINE_string(
    'val_json_file', None,
    'Groudtruth file, e.g. annotations/instances_val2017.json.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
flags.DEFINE_boolean('enable_tta', False,
                     'Use test time augmentation (slower, but more accurate).')
FLAGS = flags.FLAGS


def main(_):
  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  config.batch_size = FLAGS.batch_size
  config.val_json_file = FLAGS.val_json_file
  config.nms_configs.max_nms_inputs = anchors.MAX_DETECTION_POINTS
  base_height, base_width = utils.parse_image_size(config['image_size'])

  if FLAGS.strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif FLAGS.strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
    else:
      ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  # in format (height, width, flip)
  augmentations = []
  if FLAGS.enable_tta:
    for size_offset in (0, 128, 256):
      for flip in (False, True):
        augmentations.append(
            (base_height + size_offset, base_width + size_offset, flip))
  else:
    augmentations.append((base_height, base_width, False))

  all_detections = []
  all_labels = []
  with ds_strategy.scope():
    # Network
    model = efficientdet_keras.EfficientDetNet(config=config)
    model.build((config.batch_size, base_height, base_width, 3))
    model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))

    first_loop = True
    for height, width, flip in augmentations:
      config.image_size = (height, width)
      # dataset
      ds = dataloader.InputReader(
          FLAGS.val_file_pattern,
          is_training=False,
          use_fake_data=False,
          max_instances_per_image=config.max_instances_per_image)(
              config)
      if FLAGS.eval_samples:
        ds = ds.take(FLAGS.eval_samples // config.batch_size)

      # create the function once per augmentation, since it closes over the
      # value of config, which gets updated with the new image size
      @tf.function
      def f(images, labels):
        cls_outputs, box_outputs = model(images, training=False)
        return postprocess.generate_detections(config, cls_outputs, box_outputs,
                                               labels['image_scales'],
                                               labels['source_ids'], flip)

      # inference
      for images, labels in ds:
        if flip:
          images = tf.image.flip_left_right(images)
        detections = f(images, labels)

        all_detections.append(detections)
        if first_loop:
          all_labels.append(labels)

      first_loop = False

  # collect the giant list of detections into a map from image id to
  # detections
  detections_per_source = dict()
  for batch in all_detections:
    for d in batch:
      img_id = d[0][0]
      if img_id.numpy() in detections_per_source:
        detections_per_source[img_id.numpy()] = tf.concat(
            [d, detections_per_source[img_id.numpy()]], 0)
      else:
        detections_per_source[img_id.numpy()] = d

  # collect the groundtruth per image id
  groundtruth_per_source = dict()
  for batch in all_labels:
    for img_id, groundtruth in zip(batch['source_ids'],
                                   batch['groundtruth_data']):
      groundtruth_per_source[img_id.numpy()] = groundtruth

  # calucate the AP scores for all the images
  evaluator = coco_metric.EvaluationMetric(filename=config.val_json_file)
  for img_id, d in detections_per_source.items():
    if FLAGS.enable_tta:
      d = wbf.ensemble_detections(config, d, len(augmentations))
    evaluator.update_state(
        tf.stack([groundtruth_per_source[img_id]]).numpy(),
        postprocess.transform_detections(tf.stack([d])).numpy())

  # compute the final eval results.
  if evaluator:
    metrics = evaluator.result()
    metric_dict = {}
    for i, name in enumerate(evaluator.metric_names):
      metric_dict[name] = metrics[i]

    label_map = label_util.get_label_map(config.label_map)
    if label_map:
      for i, cid in enumerate(sorted(label_map.keys())):
        name = 'AP_/%s' % label_map[cid]
        metric_dict[name] = metrics[i - len(evaluator.metric_names)]
    print(metric_dict)


if __name__ == '__main__':
  flags.mark_flag_as_required('val_file_pattern')
  flags.mark_flag_as_required('model_dir')
  logging.set_verbosity(logging.WARNING)
  app.run(main)
