# Copyright 2021 Google Research. All Rights Reserved.
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
"""Training script."""
import copy
import os
import re
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import datasets
import effnetv2_configs
import effnetv2_model
import hparams
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'efficientnetv2-b0', 'model name.')
flags.DEFINE_string('dataset_cfg', 'Imagenet', 'dataset config name.')
flags.DEFINE_string('hparam_str', '', 'Comma separated k=v pairs of hparams.')
flags.DEFINE_string('sweeps', '', 'Comma separated k=v pairs for sweeping.')
flags.DEFINE_bool('use_tpu', True, 'If true, use TPU; otherwise use CPU/GPU.')
flags.DEFINE_string('tpu_job_name', None, 'tpu job name default to tpu_worker.')
# Cloud TPU Cluster Resolvers
flags.DEFINE_string('tpu', None, 'address e.g. grpc://ip.address.of.tpu:8470')
flags.DEFINE_string('gcp_project', None, 'Project name.')
flags.DEFINE_string('tpu_zone', None, 'GCE zone')
# Model specific flags
flags.DEFINE_string('data_dir', None, 'The directory for training images.')
flags.DEFINE_string('eval_name', None, 'Evaluation name.')
flags.DEFINE_bool('archive_ckpt', True, 'If true, archive the best ckpt.')
flags.DEFINE_string('model_dir', None, 'Dir for checkpoint and summaries.')
flags.DEFINE_string('mode', 'train', 'One of {"train", "eval"}.')
flags.DEFINE_bool('export_to_tpu', False, 'Export metagraph.')


def model_fn(features, labels, mode, params):
  """The model_fn to be used with TPUEstimator.

  Args:
    features: A dict of `Tensor` of batched images and other features.
    labels: a Tensor or a dict of Tensor representing the batched labels.
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
      `params['batch_size']` is always provided and should be used as the
      effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  logging.info('params=%s', params)
  images = features['image'] if isinstance(features, dict) else features
  labels = labels['label'] if isinstance(labels, dict) else labels
  config = params['config']
  image_size = params['image_size']
  utils.scalar('model/resolution', image_size)

  if config.model.data_format == 'channels_first':
    images = tf.transpose(images, [0, 3, 1, 2])

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  has_moving_average_decay = (config.train.ema_decay > 0)
  if FLAGS.use_tpu and not config.model.bn_type:
    config.model.bn_type = 'tpu_bn'
  # This is essential, if using a keras-derived model.
  tf.keras.backend.set_learning_phase(is_training)

  def build_model(in_images):
    """Build model using the model_name given through the command line."""
    config.model.num_classes = config.data.num_classes
    model = effnetv2_model.EffNetV2Model(config.model.model_name, config.model)
    logits = model(in_images, training=is_training)
    return logits

  pre_num_params, pre_num_flops = utils.num_params_flops(readable_format=True)

  if config.runtime.mixed_precision:
    precision = 'mixed_bfloat16' if FLAGS.use_tpu else 'mixed_float16'
    logits = utils.build_model_with_precision(precision, build_model, images,
                                              is_training)
    logits = tf.cast(logits, tf.float32)
  else:
    logits = build_model(images)

  num_params, num_flops = utils.num_params_flops(readable_format=True)
  num_params = num_params - pre_num_params
  num_flops = (num_flops - pre_num_flops) / params['batch_size']
  logging.info('backbone params/flops = %.4f M / %.4f B', num_params, num_flops)
  utils.scalar('model/params', num_params)
  utils.scalar('model/flops', num_flops)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  if config.train.loss_type == 'sigmoid':
    cross_entropy = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.cast(labels, dtype=logits.dtype),
        logits=logits,
        label_smoothing=config.train.label_smoothing)
  elif config.train.loss_type == 'custom':
    xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, dtype=logits.dtype), logits=logits)
    cross_entropy = tf.reduce_mean(tf.reduce_sum(xent, axis=-1))
  else:
    if config.data.multiclass:
      logging.info('use multi-class loss: %s', config.data.multiclass)
      labels /= tf.reshape(tf.reduce_sum(labels, axis=1), (-1, 1))
    cross_entropy = tf.losses.softmax_cross_entropy(
        onehot_labels=labels,
        logits=logits,
        label_smoothing=config.train.label_smoothing)

  train_steps = max(config.train.min_steps,
                    config.train.epochs * params['steps_per_epoch'])
  global_step = tf.train.get_global_step()
  weight_decay_inc = config.train.weight_decay_inc * (
      tf.cast(global_step, tf.float32) / tf.cast(train_steps, tf.float32))
  weight_decay = (1 + weight_decay_inc) * config.train.weight_decay
  utils.scalar('train/weight_decay', weight_decay)
  # Add weight decay to the loss for non-batch-normalization variables.
  matcher = re.compile(config.train.weight_decay_exclude)
  l2loss = weight_decay * tf.add_n([
      tf.nn.l2_loss(v)
      for v in tf.trainable_variables()
      if not matcher.match(v.name)
  ])
  loss = cross_entropy + l2loss
  utils.scalar('loss/l2reg', l2loss)
  utils.scalar('loss/xent', cross_entropy)

  if has_moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=config.train.ema_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  host_call = None
  restore_vars_dict = None
  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (
        tf.cast(global_step, tf.float32) / params['steps_per_epoch'])
    utils.scalar('train/epoch', current_epoch)

    scaled_lr = config.train.lr_base * (config.train.batch_size / 256.0)
    scaled_lr_min = config.train.lr_min * (config.train.batch_size / 256.0)
    learning_rate = utils.WarmupLearningRateSchedule(
        scaled_lr,
        steps_per_epoch=params['steps_per_epoch'],
        decay_epochs=config.train.lr_decay_epoch,
        warmup_epochs=config.train.lr_warmup_epoch,
        decay_factor=config.train.lr_decay_factor,
        lr_decay_type=config.train.lr_sched,
        total_steps=train_steps,
        minimal_lr=scaled_lr_min)(global_step)
    utils.scalar('train/lr', learning_rate)
    optimizer = utils.build_optimizer(
        learning_rate, optimizer_name=config.train.optimizer)

    if config.runtime.mixed_precision and precision=='mixed_float16':
        # Wrap optimizer with loss scale when precision is mixed_float16
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # filter trainable variables if needed.
    var_list = tf.trainable_variables()
    if config.train.varsexp:
      vars2 = [v for v in var_list if re.match(config.train.varsexp, v.name)]
      if len(vars2) == len(var_list):
        logging.warning('%s has no match.', config.train.freeze)
      logging.info('Filter variables: orig=%d, final=%d, delta=%d',
                   len(var_list), len(vars2),
                   len(var_list) - len(vars2))
      var_list = vars2

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if config.train.gclip and is_training:
      logging.info('clip gradients norm by %f', config.train.gclip)
      grads_and_vars = optimizer.compute_gradients(loss, var_list)
      with tf.name_scope('gclip'):
        grads = [gv[0] for gv in grads_and_vars]
        tvars = [gv[1] for gv in grads_and_vars]
        utils.scalar('train/gnorm', tf.linalg.global_norm(grads))
        utils.scalar('train/gnormmax',
                     tf.math.reduce_max([tf.norm(g) for g in grads]))
        # First clip each variable's norm, then clip global norm.
        clip_norm = abs(config.train.gclip)
        clipped_grads = [
            tf.clip_by_norm(g, clip_norm) if g is not None else None
            for g in grads
        ]
        clipped_grads, _ = tf.clip_by_global_norm(clipped_grads, clip_norm)
        grads_and_vars = list(zip(clipped_grads, tvars))

      with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    else:
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step, var_list=var_list)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    if not config.runtime.skip_host_call:
      host_call = utils.get_tpu_host_call(global_step, FLAGS.model_dir,
                                          config.runtime.iterations_per_loop)
  else:
    train_op = None
    if has_moving_average_decay:
      # Load moving average variables for eval.
      restore_vars_dict = ema.variables_to_restore(ema_vars)

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:

    def metric_fn(labels, logits):
      """Evaluation metric function.

      Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch, num_classes]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """
      metrics = {}
      if config.data.multiclass:
        metrics['eval/global_ap'] = tf.metrics.auc(
            labels,
            tf.nn.sigmoid(logits),
            curve='PR',
            num_thresholds=200,
            summation_method='careful_interpolation',
            name='global_ap')

        # Convert labels to set: be careful, tf.metrics.xx_at_k are horrible.
        labels = tf.cast(labels, dtype=tf.int64)
        label_to_repeat = tf.expand_dims(tf.argmax(labels, axis=-1), axis=-1)
        all_labels_set = tf.range(0, labels.shape[-1], dtype=tf.int64)
        all_labels_set = tf.expand_dims(all_labels_set, axis=0)
        labels_set = labels * all_labels_set + (1 - labels) * label_to_repeat

        metrics['eval/precision@1'] = tf.metrics.precision_at_k(
            labels_set, logits, k=1)
        metrics['eval/recall@1'] = tf.metrics.recall_at_k(
            labels_set, logits, k=1)
        metrics['eval/precision@5'] = tf.metrics.precision_at_k(
            labels_set, logits, k=5)
        metrics['eval/recall@5'] = tf.metrics.recall_at_k(
            labels_set, logits, k=5)

      # always add accuracy.
      labels = tf.argmax(labels, axis=1)
      predictions = tf.argmax(logits, axis=1)
      metrics['eval/acc_top1'] = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      metrics['eval/acc_top5'] = tf.metrics.mean(in_top_5)
      metrics['model/resolution'] = tf.metrics.mean(image_size)
      metrics['model/flops'] = tf.metrics.mean(num_flops)
      metrics['model/params'] = tf.metrics.mean(num_params)
      return metrics

    eval_metrics = (metric_fn, [labels, logits])

  if has_moving_average_decay and not is_training:

    def scaffold_fn():  # read ema for eval jobs.
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)
  elif config.train.ft_init_ckpt and is_training:

    def scaffold_fn():
      logging.info('restore variables from %s', config.train.ft_init_ckpt)
      var_map = utils.get_ckpt_var_map(
          ckpt_path=config.train.ft_init_ckpt,
          skip_mismatch=True,
          init_ema=config.train.ft_init_ema)
      tf.train.init_from_checkpoint(config.train.ft_init_ckpt, var_map)
      return tf.train.Scaffold()
  else:
    scaffold_fn = None

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


def main(unused_argv):
  config = copy.deepcopy(hparams.base_config)
  config.override(effnetv2_configs.get_model_config(FLAGS.model_name))
  config.override(datasets.get_dataset_config(FLAGS.dataset_cfg))
  config.override(FLAGS.hparam_str)
  config.override(FLAGS.sweeps)

  train_size = config.train.isize
  eval_size = config.eval.isize
  if train_size <= 16.:
    train_size = int(eval_size * train_size) // 16 * 16
  input_image_size = eval_size if FLAGS.mode == 'eval' else train_size

  if FLAGS.mode == 'train':
    if not tf.io.gfile.exists(FLAGS.model_dir):
      tf.io.gfile.makedirs(FLAGS.model_dir)
    config.save_to_yaml(os.path.join(FLAGS.model_dir, 'config.yaml'))

  train_split = config.train.split or 'train'
  eval_split = config.eval.split or 'eval'
  num_train_images = config.data.splits[train_split].num_images
  num_eval_images = config.data.splits[eval_split].num_images

  if FLAGS.tpu or FLAGS.use_tpu:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=max(100, config.runtime.iterations_per_loop),
      keep_checkpoint_max=config.runtime.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=(
          config.runtime.keep_checkpoint_every_n_hours),
      log_step_count_steps=config.runtime.log_step_count_steps,
      session_config=tf.ConfigProto(
          isolate_session_state=True, log_device_placement=False),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=config.runtime.iterations_per_loop,
          tpu_job_name=FLAGS.tpu_job_name,
          per_host_input_for_training=(
              tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2)))
  # Initializes model parameters.
  params = dict(
      steps_per_epoch=num_train_images / config.train.batch_size,
      image_size=input_image_size,
      config=config)

  est = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=config.train.batch_size,
      eval_batch_size=config.eval.batch_size,
      export_to_tpu=FLAGS.export_to_tpu,
      params=params)

  image_dtype = None
  if config.runtime.mixed_precision:
    image_dtype = 'bfloat16' if FLAGS.use_tpu else 'float16'

  train_steps = max(
      config.train.min_steps,
      config.train.epochs * num_train_images // config.train.batch_size)
  dataset_eval = datasets.build_dataset_input(False, input_image_size,
                                              image_dtype, FLAGS.data_dir,
                                              eval_split, config.data)

  if FLAGS.mode == 'eval':
    eval_steps = num_eval_images // config.eval.batch_size
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=60 * 60 * 24):
      logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = est.evaluate(
            input_fn=dataset_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt,
            name=FLAGS.eval_name)
        elapsed_time = int(time.time() - start_timestamp)
        logging.info('Eval results: %s. Elapsed seconds: %d', eval_results,
                     elapsed_time)
        if FLAGS.archive_ckpt:
          utils.archive_ckpt(eval_results, eval_results['eval/acc_top1'], ckpt)

        # Terminate eval job when final checkpoint is reached
        try:
          current_step = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          logging.info('%s has no global step info: stop!', ckpt)
          break

        logging.info('Finished step: %d, total %d', current_step, train_steps)
        if current_step >= train_steps:
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        logging.info('Checkpoint %s no longer exists, skip saving.', ckpt)
  else:  # FLAGS.mode == 'train'
    try:
      checkpoint_reader = tf.compat.v1.train.NewCheckpointReader(
          tf.train.latest_checkpoint(FLAGS.model_dir))
      current_step = checkpoint_reader.get_tensor(
          tf.compat.v1.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
      current_step = 0

    logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', train_steps, config.train.epochs, current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []  # add hooks if needed.
      if not config.train.stages:
        dataset_train = datasets.build_dataset_input(True, input_image_size,
                                                     image_dtype,
                                                     FLAGS.data_dir,
                                                     train_split, config.data)
        est.train(
            input_fn=dataset_train.input_fn, max_steps=train_steps, hooks=hooks)
      else:
        curr_step = 0
        try:
          ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
          curr_step = int(os.path.basename(ckpt).split('-')[1])
        except (IndexError, TypeError):
          logging.info('%s has no ckpt with valid step.', FLAGS.model_dir)

        total_stages = config.train.stages
        if config.train.sched:
          if config.model.dropout_rate:
            dp_list = np.linspace(0, config.model.dropout_rate, total_stages)
          else:
            dp_list = [None] * total_stages

          del dp_list
          ram_list = np.linspace(5, config.data.ram, total_stages)
          mixup_list = np.linspace(0, config.data.mixup_alpha, total_stages)
          cutmix_list = np.linspace(0, config.data.cutmix_alpha, total_stages)

        ibase = config.data.ibase or (input_image_size / 2)
        # isize_list = np.linspace(ibase, input_image_size, total_stages)
        for stage in range(curr_step // train_steps, total_stages):
          tf.compat.v1.reset_default_graph()
          ratio = float(stage + 1) / float(total_stages)
          max_steps = int(ratio * train_steps)
          image_size = int(ibase + (input_image_size - ibase) * ratio)
          params['image_size'] = image_size

          if config.train.sched:
            config.data.ram = ram_list[stage]
            config.data.mixup_alpha = mixup_list[stage]
            config.data.cutmix_alpha = cutmix_list[stage]
            # config.model.dropout_rate = dp_list[stage]

          ds_lab_cls = datasets.build_dataset_input(True, image_size,
                                                    image_dtype, FLAGS.data_dir,
                                                    train_split, config.data)

          est = tf.estimator.tpu.TPUEstimator(
              use_tpu=FLAGS.use_tpu,
              model_fn=model_fn,
              config=run_config,
              train_batch_size=config.train.batch_size,
              eval_batch_size=config.eval.batch_size,
              export_to_tpu=FLAGS.export_to_tpu,
              params=params)
          est.train(
              input_fn=ds_lab_cls.input_fn, max_steps=max_steps, hooks=hooks)
    else:
      raise ValueError('Unknown mode %s' % FLAGS.mode)


if __name__ == '__main__':
  tf.disable_eager_execution()
  app.run(main)
