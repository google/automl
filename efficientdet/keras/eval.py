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

flags.DEFINE_string('val_file_pattern', None,
                    'Glob for eval tfrecords, e.g. coco/val-*.tfrecord.')
flags.DEFINE_string(
    'val_json_file', None,
    'Groudtruth file, e.g. annotations/instances_val2017.json.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('eval_samples', 8, 'Batch size.')
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

  # Network
  model = efficientdet_keras.EfficientDetNet(config=config)
  model.build((config.batch_size, base_height, base_width, 3))
  model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))

  @tf.function
  def f(imgs, labels, flip):
    cls_outputs, box_outputs = model(imgs, training=False)
    return postprocess.generate_detections(config, cls_outputs, box_outputs,
                                           labels['image_scales'],
                                           labels['source_ids'], flip)

  # in format (height, width, flip)
  augmentations = []
  if FLAGS.enable_tta:
    for size_offset in (0, 128, 256):
      for flip in (False, True):
        augmentations.append(
            (base_height + size_offset, base_width + size_offset, flip))
  else:
    augmentations.append((base_height, base_width, False))

  evaluator = None
  detections_per_source = dict()
  for height, width, flip in augmentations:
    config.image_size = (height, width)
    # dataset
    ds = dataloader.InputReader(
        FLAGS.val_file_pattern,
        is_training=False,
        use_fake_data=False,
        max_instances_per_image=config.max_instances_per_image)(
            config)

    # compute stats for all batches.
    total_steps = FLAGS.eval_samples // FLAGS.batch_size
    progress = tf.keras.utils.Progbar(total_steps)
    for i, (images, labels) in enumerate(ds):
      progress.update(i, values=None)
      if i > total_steps:
        break

      if flip:
        images = tf.image.flip_left_right(images)
      detections = f(images, labels, flip)

      for img_id, d in zip(labels['source_ids'], detections):
        if img_id.numpy() in detections_per_source:
          detections_per_source[img_id.numpy()] = tf.concat(
              [d, detections_per_source[img_id.numpy()]], 0)
        else:
          detections_per_source[img_id.numpy()] = d

      evaluator = coco_metric.EvaluationMetric(filename=config.val_json_file)
      for d in detections_per_source.values():
        if FLAGS.enable_tta:
          d = wbf.ensemble_detections(config, d, len(augmentations))
        evaluator.update_state(
            labels['groundtruth_data'].numpy(),
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
  flags.mark_flag_as_required('val_json_file')
  flags.mark_flag_as_required('model_dir')
  logging.set_verbosity(logging.WARNING)
  app.run(main)
