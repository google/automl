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
import numpy as np

import dataloader
import det_model_fn
import hparams_config
import utils

flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string('eval_name', default=None, help='Eval job name')
flags.DEFINE_enum('strategy', None, ['tpu', 'horovod', ''],
                  'Training: horovod for multi-gpu, if None, use TF default.')

flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')
flags.DEFINE_bool(
    'use_xla', False,
    'Use XLA even if strategy is not tpu. If strategy is tpu, always use XLA, '
    'and this flag has no effect.')
flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('backbone_ckpt', '',
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string('ckpt', None,
                    'Start training from this EfficientDet checkpoint.')

flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_bool('use_spatial_partition', False, 'Use spatial partition.')
flags.DEFINE_integer(
    'num_cores_per_replica', default=8, help='Number of TPU cores per'
    'replica when using spatial partition.')
flags.DEFINE_multi_integer(
    'input_partition_dims', [1, 4, 2, 1],
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
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
flags.DEFINE_integer(
    'tf_random_seed', None, 'Sets the TF graph seed for deterministic execution'
    ' across runs (for debugging).')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

FLAGS = flags.FLAGS


def main(_):

  if FLAGS.strategy == 'horovod':
    import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
    logging.info('Use horovod with multi gpus')
    hvd.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
  import tensorflow.compat.v1 as tf  # pylint: disable=g-import-not-at-top
  tf.enable_v2_tensorshape()
  tf.disable_eager_execution()

  if FLAGS.strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    tf.Session.reset(tpu_grpc_url)
  else:
    tpu_cluster_resolver = None

  # Check data path
  if FLAGS.mode in ('train',
                    'train_and_eval') and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if FLAGS.validation_file_pattern is None:
      raise RuntimeError('You must specify --validation_file_pattern '
                         'for evaluation.')

  # Parse and override hparams
  config = hparams_config.get_detection_config(FLAGS.model_name)
  config.override(FLAGS.hparams)
  if FLAGS.num_epochs:  # NOTE: remove this flag after updating all docs.
    config.num_epochs = FLAGS.num_epochs

  # Parse image size in case it is in string format.
  config.image_size = utils.parse_image_size(config.image_size)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # In the TPUEstimator context, the meaning of `shard` and `replica` is the
  # same; follwing the API, here has mixed use of both.
  if FLAGS.use_spatial_partition:
    # Checks input_partition_dims agrees with num_cores_per_replica.
    if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
      raise RuntimeError('--num_cores_per_replica must be a product of array'
                         'elements in --input_partition_dims.')

    labels_partition_dims = {
        'mean_num_positives': None,
        'source_ids': None,
        'groundtruth_data': None,
        'image_scales': None,
    }
    # The Input Partition Logic: We partition only the partition-able tensors.
    # Spatial partition requires that the to-be-partitioned tensors must have a
    # dimension that is a multiple of `partition_dims`. Depending on the
    # `partition_dims` and the `image_size` and the `max_level` in config, some
    # high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
    # be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
    # size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
    # [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
    # case, the level-8 and level-9 target tensors are not partition-able, and
    # the highest partition-able level is 7.
    feat_sizes = utils.get_feat_sizes(
        config.get('image_size'), config.get('max_level'))
    for level in range(config.get('min_level'), config.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

      spatial_dim = feat_sizes[level]
      if _can_partition(spatial_dim['height']) and _can_partition(
          spatial_dim['width']):
        labels_partition_dims['box_targets_%d' %
                              level] = FLAGS.input_partition_dims
        labels_partition_dims['cls_targets_%d' %
                              level] = FLAGS.input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['cls_targets_%d' % level] = None
    num_cores_per_replica = FLAGS.num_cores_per_replica
    input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores

  params = dict(
      config.as_dict(),
      model_name=FLAGS.model_name,
      iterations_per_loop=FLAGS.iterations_per_loop,
      model_dir=FLAGS.model_dir,
      num_shards=num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      strategy=FLAGS.strategy,
      backbone_ckpt=FLAGS.backbone_ckpt,
      ckpt=FLAGS.ckpt,
      val_json_file=FLAGS.val_json_file,
      testdev_dir=FLAGS.testdev_dir,
      mode=FLAGS.mode)
  config_proto = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  if FLAGS.strategy != 'tpu':
    if FLAGS.use_xla:
      config_proto.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_1)
    config_proto.gpu_options.allow_growth = True

  tpu_config = tf.estimator.tpu.TPUConfig(
      FLAGS.iterations_per_loop,
      num_cores_per_replica=num_cores_per_replica,
      input_partition_dims=input_partition_dims,
      per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
      .PER_HOST_V2)

  if FLAGS.strategy == 'horovod':
    model_dir = FLAGS.model_dir if hvd.rank() == 0 else None
  else:
    model_dir = FLAGS.model_dir

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=model_dir,
      log_step_count_steps=FLAGS.iterations_per_loop,
      session_config=config_proto,
      tpu_config=tpu_config,
      tf_random_seed=FLAGS.tf_random_seed,
  )

  model_fn_instance = det_model_fn.get_model_fn(FLAGS.model_name)
  max_instances_per_image = config.max_instances_per_image
  eval_steps = int(FLAGS.eval_samples // FLAGS.eval_batch_size)
  use_tpu = (FLAGS.strategy == 'tpu')
  logging.info(params)

  def _train(steps):
    """Build train estimator and run training if steps > 0."""
    train_estimator = tf.estimator.tpu.TPUEstimator(
        model_fn=model_fn_instance,
        use_tpu=use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        config=run_config,
        params=params)
    train_estimator.train(
        input_fn=dataloader.InputReader(
            FLAGS.training_file_pattern,
            is_training=True,
            use_fake_data=FLAGS.use_fake_data,
            max_instances_per_image=max_instances_per_image),
        max_steps=steps)

  def _eval(steps):
    """Build estimator and eval the latest checkpoint if steps > 0."""
    eval_params = dict(
        params,
        strategy=FLAGS.strategy,
        input_rand_hflip=False,
        is_training_bn=False,
    )
    eval_estimator = tf.estimator.tpu.TPUEstimator(
        model_fn=model_fn_instance,
        use_tpu=use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        config=run_config,
        params=eval_params)
    eval_results = eval_estimator.evaluate(
        input_fn=dataloader.InputReader(
            FLAGS.validation_file_pattern,
            is_training=False,
            max_instances_per_image=max_instances_per_image),
        steps=steps,
        name=FLAGS.eval_name)
    logging.info('Evaluation results: %s', eval_results)
    return eval_results

  # start train/eval flow.
  if FLAGS.mode == 'train':
    total_examples = int(config.num_epochs * FLAGS.num_examples_per_epoch)
    _train(total_examples // FLAGS.train_batch_size)
    if FLAGS.eval_after_training:
      _eval(eval_steps)

  elif FLAGS.mode == 'eval':
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout):

      logging.info('Starting to evaluate.')
      try:
        eval_results = _eval(eval_steps)
        # Terminate eval job when final checkpoint is reached.
        try:
          current_step = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          logging.info('%s has no global step info: stop!', ckpt)
          break

        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
        total_step = int((config.num_epochs * FLAGS.num_examples_per_epoch) /
                         FLAGS.train_batch_size)
        if current_step >= total_step:
          logging.info('Evaluation finished after training step %d',
                       current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        logging.info('Checkpoint %s no longer exists, skipping.', ckpt)

  elif FLAGS.mode == 'train_and_eval':
    for cycle in range(config.num_epochs):
      logging.info('Starting training cycle, epoch: %d.', cycle)
      _train(FLAGS.num_examples_per_epoch // FLAGS.train_batch_size)
      logging.info('Starting evaluation cycle, epoch: %d.', cycle)
      eval_results = _eval(eval_steps)
      ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
      utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)

  else:
    logging.info('Mode not found.')


if __name__ == '__main__':
  app.run(main)
