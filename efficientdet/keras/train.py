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
"""The main training script."""
import os
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import dataloader
import hparams_config
import utils
from keras import train_lib
from keras import util_keras

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

# Model specific paramenters
flags.DEFINE_string(
    'eval_master',
    default='',
    help='GRPC URL of the eval master. Set to an appropriate value when running'
    ' on CPU/GPU')
flags.DEFINE_string('eval_name', default=None, help='Eval job name')
flags.DEFINE_enum('strategy', None, ['tpu', 'gpus', ''],
                  'Training: gpus for multi-gpu, if None, use TF default.')

flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')

flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')
flags.DEFINE_bool(
    'use_xla', False,
    'Use XLA even if strategy is not tpu. If strategy is tpu, always use XLA, '
    'and this flag has no effect.')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')

flags.DEFINE_string('pretrained_ckpt', None,
                    'Start training from this EfficientDet checkpoint.')

flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                     'evaluation.')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string('validation_file_pattern', None,
                    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file', None,
    'COCO validation JSON containing golden bounding boxes. If None, use the '
    'ground truth from the dataloader. Ignored if testdev_dir is not None.')
flags.DEFINE_string('testdev_dir', None,
                    'COCO testdev dir. If not None, ignorer val_json_file.')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', None, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_string('model_name', 'efficientdet-d1', 'Model name.')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('debug', False, 'Enable debug mode')
flags.DEFINE_bool('profile', False, 'Enable profile mode')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

FLAGS = flags.FLAGS


def main(_):
  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  if FLAGS.num_epochs:  # NOTE: remove this flag after updating all docs.
    config.num_epochs = FLAGS.num_epochs

  # Parse image size in case it is in string format.
  config.image_size = utils.parse_image_size(config.image_size)

  if FLAGS.use_xla and FLAGS.strategy != 'tpu':
    tf.config.optimizer.set_jit(True)
    for gpu in tf.config.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)

  if FLAGS.debug:
    tf.config.experimental_run_functions_eagerly(True)
    tf.debugging.set_log_device_placement(True)
    tf.random.set_seed(111111)
    logging.set_verbosity(logging.DEBUG)

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

  # Check data path
  if FLAGS.mode in ('train',
                    'train_and_eval') and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if FLAGS.validation_file_pattern is None:
      raise RuntimeError('You must specify --validation_file_pattern '
                         'for evaluation.')

  steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.batch_size
  params = dict(
      config.as_dict(),
      model_name=FLAGS.model_name,
      iterations_per_loop=FLAGS.iterations_per_loop,
      model_dir=FLAGS.model_dir,
      steps_per_epoch=steps_per_epoch,
      strategy=FLAGS.strategy,
      batch_size=FLAGS.batch_size // ds_strategy.num_replicas_in_sync,
      num_shards=ds_strategy.num_replicas_in_sync,
      val_json_file=FLAGS.val_json_file,
      testdev_dir=FLAGS.testdev_dir,
      mode=FLAGS.mode)

  # set mixed precision policy by keras api.
  precision = utils.get_precision(params['strategy'], params['mixed_precision'])
  policy = tf.keras.mixed_precision.experimental.Policy(precision)
  tf.keras.mixed_precision.experimental.set_policy(policy)

  def get_dataset(is_training, params):
    file_pattern = (
        FLAGS.training_file_pattern
        if is_training else FLAGS.validation_file_pattern)
    if not file_pattern:
      raise ValueError('No matching files.')

    return dataloader.InputReader(
        file_pattern,
        is_training=is_training,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=config.max_instances_per_image)(
            params)

  with ds_strategy.scope():
    model = train_lib.EfficientDetNetTrain(params['model_name'], config)
    model.compile(
        optimizer=train_lib.get_optimizer(params),
        loss={
            'box_loss':
                train_lib.BoxLoss(
                    params['delta'], reduction=tf.keras.losses.Reduction.NONE),
            'box_iou_loss':
                train_lib.BoxIouLoss(
                    params['iou_loss_type'],
                    params['min_level'],
                    params['max_level'],
                    params['num_scales'],
                    params['aspect_ratios'],
                    params['anchor_scale'],
                    params['image_size'],
                    reduction=tf.keras.losses.Reduction.NONE),
            'class_loss':
                train_lib.FocalLoss(
                    params['alpha'],
                    params['gamma'],
                    label_smoothing=params['label_smoothing'],
                    reduction=tf.keras.losses.Reduction.NONE),
            'seg_loss':
                tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE)
        })

    if FLAGS.pretrained_ckpt:
      ckpt_path = tf.train.latest_checkpoint(FLAGS.pretrained_ckpt)
      util_keras.restore_ckpt(model, ckpt_path, params['moving_average_decay'])
    tf.io.gfile.makedirs(FLAGS.model_dir)
    model.fit(
        get_dataset(True, params=params),
        epochs=params['num_epochs'],
        steps_per_epoch=steps_per_epoch,
        callbacks=train_lib.get_callbacks(params, FLAGS.profile),
        validation_data=get_dataset(False, params=params),
        validation_steps=(FLAGS.eval_samples // FLAGS.batch_size))
  model.save_weights(os.path.join(FLAGS.model_dir, 'ckpt-final'))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
