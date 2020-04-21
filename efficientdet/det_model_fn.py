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
"""Model function definition, including both architecture and loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import anchors
import coco_metric
import efficientdet_arch
import hparams_config
import retinanet_arch
import utils

_DEFAULT_BATCH_SIZE = 64


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule."""
  # params['batch_size'] is per-shard within model_fn if use_tpu=true.
  batch_size = (params['batch_size'] * params['num_shards'] if params['use_tpu']
                else params['batch_size'])
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (params['learning_rate'] * batch_size /
                                      _DEFAULT_BATCH_SIZE)
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)


def stepwise_lr_schedule(adjusted_learning_rate, lr_warmup_init,
                         lr_warmup_step, first_lr_drop_step,
                         second_lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_step` before decaying.
  logging.info('LR schedule method: stepwise')
  linear_warmup = (lr_warmup_init +
                   (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step *
                    (adjusted_learning_rate - lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step,
                           linear_warmup, adjusted_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step],
                 [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate


def cosine_lr_schedule_tf2(adjusted_lr, lr_warmup_init, lr_warmup_step,
                           total_steps, step):
  """TF2 friendly cosine learning rate schedule."""
  logging.info('LR schedule method: cosine')
  def warmup_lr(step):
    return lr_warmup_init + (adjusted_lr - lr_warmup_init) * (
        tf.cast(step, tf.float32) / tf.cast(lr_warmup_step, tf.float32))
  def cosine_lr(step):
    decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)
    step = tf.cast(step - lr_warmup_step, tf.float32)
    cosine_decay = 0.5 * (1 + tf.cos(np.pi * step / decay_steps))
    alpha = 0.0
    decayed = (1 - alpha) * cosine_decay + alpha
    return adjusted_lr * tf.cast(decayed, tf.float32)
  return tf.cond(step <= lr_warmup_step,
                 lambda: warmup_lr(step),
                 lambda: cosine_lr(step))


def cosine_lr_schedule(adjusted_lr, lr_warmup_init, lr_warmup_step,
                       total_steps, step):
  logging.info('LR schedule method: cosine')
  linear_warmup = (
      lr_warmup_init +
      (tf.cast(step, dtype=tf.float32) / lr_warmup_step *
       (adjusted_lr - lr_warmup_init)))
  cosine_lr = 0.5 * adjusted_lr * (
      1 + tf.cos(np.pi * tf.cast(step, tf.float32) / total_steps))
  return tf.where(step < lr_warmup_step, linear_warmup, cosine_lr)


def polynomial_lr_schedule(adjusted_lr, lr_warmup_init, lr_warmup_step, power,
                           total_steps, step):
  logging.info('LR schedule method: polynomial')
  linear_warmup = (
      lr_warmup_init + (tf.cast(step, dtype=tf.float32) / lr_warmup_step *
                        (adjusted_lr - lr_warmup_init)))
  polynomial_lr = adjusted_lr * tf.pow(
      1 - (tf.cast(step, tf.float32) / total_steps), power)
  return tf.where(step < lr_warmup_step, linear_warmup, polynomial_lr)


def learning_rate_schedule(params, global_step):
  """Learning rate schedule based on global step."""
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    return stepwise_lr_schedule(params['adjusted_learning_rate'],
                                params['lr_warmup_init'],
                                params['lr_warmup_step'],
                                params['first_lr_drop_step'],
                                params['second_lr_drop_step'], global_step)

  if lr_decay_method == 'cosine':
    return cosine_lr_schedule(params['adjusted_learning_rate'],
                              params['lr_warmup_init'],
                              params['lr_warmup_step'],
                              params['total_steps'], global_step)

  if lr_decay_method == 'polynomial':
    return polynomial_lr_schedule(params['adjusted_learning_rate'],
                                  params['lr_warmup_init'],
                                  params['lr_warmup_step'],
                                  params['poly_lr_power'],
                                  params['total_steps'], global_step)

  raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))


def focal_loss(logits, targets, alpha, gamma, normalizer):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.

  Args:
    logits: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    targets: A float32 tensor of size
      [batch, height_in, width_in, num_predictions].
    alpha: A float32 scalar multiplying alpha to the loss from positive examples
      and (1-alpha) to the loss from negative examples.
    gamma: A float32 scalar modulating loss from hard and easy examples.
    normalizer: A float32 scalar normalizes the total loss from all examples.
  Returns:
    loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    positive_label_mask = tf.equal(targets, 1.0)
    cross_entropy = (
        tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = tf.exp(gamma * targets * neg_logits - gamma * tf.log1p(
        tf.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = tf.where(positive_label_mask, alpha * loss,
                             (1.0 - alpha) * loss)
    weighted_loss /= normalizer
  return weighted_loss


def _classification_loss(cls_outputs,
                         cls_targets,
                         num_positives,
                         alpha=0.25,
                         gamma=2.0):
  """Computes classification loss."""
  normalizer = num_positives
  classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
                                   normalizer)
  return classification_loss


def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
  """Computes box regression loss."""
  # delta is typically around the mean value of regression target.
  # for instances, the regression targets of 512x512 input with 6 anchors on
  # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
  normalizer = num_positives * 4.0
  mask = tf.not_equal(box_targets, 0.0)
  box_loss = tf.losses.huber_loss(
      box_targets,
      box_outputs,
      weights=mask,
      delta=delta,
      reduction=tf.losses.Reduction.SUM)
  box_loss /= normalizer
  return box_loss


def detection_loss(cls_outputs, box_outputs, labels, params):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundtruth targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
  Returns:
    total_loss: an integer tensor representing total loss reducing from
      class and box losses from all levels.
    cls_loss: an integer tensor representing total class loss.
    box_loss: an integer tensor representing total box regression loss.
  """
  # Sum all positives in a batch for normalization and avoid zero
  # num_positives_sum, which would lead to inf loss during training
  num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
  levels = cls_outputs.keys()

  cls_losses = []
  box_losses = []
  for level in levels:
    if params['data_format'] == 'channels_first':
      labels['cls_targets_%d' % level] = tf.transpose(
          labels['cls_targets_%d' % level], [0, 3, 1, 2])
      labels['box_targets_%d' % level] = tf.transpose(
          labels['box_targets_%d' % level], [0, 3, 1, 2])
    # Onehot encoding for classification labels.
    cls_targets_at_level = tf.one_hot(
        labels['cls_targets_%d' % level],
        params['num_classes'])
    if params['data_format'] == 'channels_first':
      bs, _, width, height, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, -1, width, height])
    else:
      bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
      cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                        [bs, width, height, -1])
    box_targets_at_level = labels['box_targets_%d' % level]
    cls_loss = _classification_loss(
        cls_outputs[level],
        cls_targets_at_level,
        num_positives_sum,
        alpha=params['alpha'],
        gamma=params['gamma'])
    if params['data_format'] == 'channels_first':
      cls_loss = tf.reshape(cls_loss,
                            [bs, -1, width, height, params['num_classes']])
    else:
      cls_loss = tf.reshape(cls_loss,
                            [bs, width, height, -1, params['num_classes']])
    cls_loss *= tf.cast(tf.expand_dims(
        tf.not_equal(labels['cls_targets_%d' % level], -2), -1), tf.float32)
    cls_losses.append(tf.reduce_sum(cls_loss))
    box_losses.append(
        _box_loss(
            box_outputs[level],
            box_targets_at_level,
            num_positives_sum,
            delta=params['delta']))

  # Sum per level losses to total loss.
  cls_loss = tf.add_n(cls_losses)
  box_loss = tf.add_n(box_losses)
  total_loss = cls_loss + params['box_loss_weight'] * box_loss
  return total_loss, cls_loss, box_loss


def add_metric_fn_inputs(params,
                         cls_outputs,
                         box_outputs,
                         metric_fn_inputs,
                         max_detection_points=anchors.MAX_DETECTION_POINTS):
  """Selects top-k predictions and adds the selected to metric_fn_inputs.

  Args:
    params: a parameter dictionary that includes `min_level`, `max_level`,
      `batch_size`, and `num_classes`.
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    metric_fn_inputs: a dictionary that will hold the top-k selections.
    max_detection_points: an integer specifing the maximum detection points to
      keep before NMS. Keep all anchors if max_detection_points <= 0.
  """
  num_classes = params['num_classes']
  cls_outputs_all = []
  box_outputs_all = []
  # Concatenates class and box of all levels into one tensor.
  for level in range(params['min_level'], params['max_level'] + 1):
    if params['data_format'] == 'channels_first':
      cls_outputs[level] = tf.transpose(cls_outputs[level], [0, 2, 3, 1])
      box_outputs[level] = tf.transpose(box_outputs[level], [0, 2, 3, 1])

    cls_outputs_all.append(tf.reshape(
        cls_outputs[level],
        [params['batch_size'], -1, num_classes]))
    box_outputs_all.append(tf.reshape(
        box_outputs[level], [params['batch_size'], -1, 4]))
  cls_outputs_all = tf.concat(cls_outputs_all, 1)
  box_outputs_all = tf.concat(box_outputs_all, 1)

  if max_detection_points > 0:
    # Prune anchors and detections to only keep max_detection_points.
    # Due to some issues, top_k is currently slow in graph model.
    cls_outputs_all_reshape = tf.reshape(cls_outputs_all,
                                         [params['batch_size'], -1])
    _, cls_topk_indices = tf.math.top_k(cls_outputs_all_reshape,
                                        k=anchors.MAX_DETECTION_POINTS,
                                        sorted=False)
    indices = cls_topk_indices // num_classes
    classes = cls_topk_indices % num_classes
    cls_indices = tf.stack([indices, classes], axis=2)
    cls_outputs_all_after_topk = tf.gather_nd(
        cls_outputs_all, cls_indices, batch_dims=1)
    box_outputs_all_after_topk = tf.gather_nd(
        box_outputs_all, tf.expand_dims(indices, 2), batch_dims=1)
  else:
    # Keep all anchors, but for each anchor, just keep the max probablity for
    # each class.
    cls_outputs_idx = tf.math.argmax(cls_outputs_all, axis=-1)
    num_anchors = cls_outputs_all.shape[1]

    classes = cls_outputs_idx
    indices = tf.reshape(
        tf.tile(tf.range(num_anchors), [params['batch_size']]),
        [-1, num_anchors])
    cls_outputs_all_after_topk = tf.reduce_max(cls_outputs_all, -1)
    box_outputs_all_after_topk = box_outputs_all

  metric_fn_inputs['cls_outputs_all'] = cls_outputs_all_after_topk
  metric_fn_inputs['box_outputs_all'] = box_outputs_all_after_topk
  metric_fn_inputs['indices_all'] = indices
  metric_fn_inputs['classes_all'] = classes


def coco_metric_fn(batch_size,
                   anchor_labeler,
                   filename=None,
                   testdev_dir=None,
                   **kwargs):
  """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
  # add metrics to output
  detections_bs = []
  for index in range(batch_size):
    cls_outputs_per_sample = kwargs['cls_outputs_all'][index]
    box_outputs_per_sample = kwargs['box_outputs_all'][index]
    indices_per_sample = kwargs['indices_all'][index]
    classes_per_sample = kwargs['classes_all'][index]
    detections = anchor_labeler.generate_detections(
        cls_outputs_per_sample, box_outputs_per_sample, indices_per_sample,
        classes_per_sample, tf.slice(kwargs['source_ids'], [index], [1]),
        tf.slice(kwargs['image_scales'], [index], [1]),
        disable_pyfun=kwargs.get('disable_pyfun', None),
    )
    detections_bs.append(detections)

  if testdev_dir:
    eval_metric = coco_metric.EvaluationMetric(testdev_dir=testdev_dir)
    coco_metrics = eval_metric.estimator_metric_fn(detections_bs, tf.zeros([1]))
  else:
    eval_metric = coco_metric.EvaluationMetric(filename=filename)
    coco_metrics = eval_metric.estimator_metric_fn(detections_bs,
                                                   kwargs['groundtruth_data'])
  return coco_metrics


def reg_l2_loss(weight_decay, regex=r'.*(kernel|weight):0$'):
  """Return regularization l2 loss loss."""
  var_match = re.compile(regex)
  return weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if var_match.match(v.name)])


def _model_fn(features, labels, mode, params, model, variable_filter_fn=None):
  """Model definition entry.

  Args:
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    model: the model outputs class logits and box regression outputs.
    variable_filter_fn: the filter function that takes trainable_variables and
      returns the variable list after applying the filter rule.

  Returns:
    tpu_spec: the TPUEstimatorSpec to run training, evaluation, or prediction.

  Raises:
    RuntimeError: if both ckpt and backbone_ckpt are set.
  """
  # Convert params (dict) to Config for easier access.
  if params['data_format'] == 'channels_first':
    features = tf.transpose(features, [0, 3, 1, 2])
  def _model_outputs():
    return model(features, config=hparams_config.Config(params))

  if params['use_bfloat16']:
    with tf.tpu.bfloat16_scope():
      cls_outputs, box_outputs = _model_outputs()
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = _model_outputs()
    levels = cls_outputs.keys()

  # First check if it is in PREDICT mode.
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'image': features,
    }
    for level in levels:
      predictions['cls_outputs_%d' % level] = cls_outputs[level]
      predictions['box_outputs_%d' % level] = box_outputs[level]
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)

  # cls_loss and box_loss are for logging. only total_loss is optimized.
  det_loss, cls_loss, box_loss = detection_loss(cls_outputs, box_outputs,
                                                labels, params)
  l2loss = reg_l2_loss(params['weight_decay'])
  total_loss = det_loss + l2loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    utils.scalar('lrn_rate', learning_rate)
    utils.scalar('trainloss/cls_loss', cls_loss)
    utils.scalar('trainloss/box_loss', box_loss)
    utils.scalar('trainloss/det_loss', det_loss)
    utils.scalar('trainloss/l2_loss', l2loss)
    utils.scalar('trainloss/loss', total_loss)

  moving_average_decay = params['moving_average_decay']
  if moving_average_decay:
    ema = tf.train.ExponentialMovingAverage(
        decay=moving_average_decay, num_updates=global_step)
    ema_vars = utils.get_ema_vars()

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    if params['use_tpu']:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    var_list = tf.trainable_variables()
    if variable_filter_fn:
      var_list = variable_filter_fn(var_list, params['resnet_depth'])

    if params.get('clip_gradients_norm', 0) > 0:
      logging.info('clip gradients norm by %f', params['clip_gradients_norm'])
      grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
      with tf.name_scope('clip'):
        grads = [gv[0] for gv in grads_and_vars]
        tvars = [gv[1] for gv in grads_and_vars]
        clipped_grads, gnorm = tf.clip_by_global_norm(
            grads, params['clip_gradients_norm'])
        utils.scalar('gnorm', gnorm)
        grads_and_vars = list(zip(clipped_grads, tvars))

      with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)
    else:
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            total_loss, global_step, var_list=var_list)

    if moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(**kwargs):
      """Returns a dictionary that has the evaluation metrics."""
      batch_size = params['batch_size']
      eval_anchors = anchors.Anchors(params['min_level'],
                                     params['max_level'],
                                     params['num_scales'],
                                     params['aspect_ratios'],
                                     params['anchor_scale'],
                                     params['image_size'])
      anchor_labeler = anchors.AnchorLabeler(eval_anchors,
                                             params['num_classes'])
      cls_loss = tf.metrics.mean(kwargs['cls_loss_repeat'])
      box_loss = tf.metrics.mean(kwargs['box_loss_repeat'])

      if params.get('testdev_dir', None):
        logging.info('Eval testdev_dir %s', params['testdev_dir'])
        coco_metrics = coco_metric_fn(
            batch_size,
            anchor_labeler,
            params['val_json_file'],
            testdev_dir=params['testdev_dir'],
            disable_pyfun=params.get('disable_pyfun', None),
            **kwargs)
      else:
        logging.info('Eval val with groudtruths %s.', params['val_json_file'])
        coco_metrics = coco_metric_fn(batch_size, anchor_labeler,
                                      params['val_json_file'], **kwargs)

      # Add metrics to output.
      output_metrics = {
          'cls_loss': cls_loss,
          'box_loss': box_loss,
      }
      output_metrics.update(coco_metrics)
      return output_metrics

    cls_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(cls_loss, 0), [params['batch_size'],]),
        [params['batch_size'], 1])
    box_loss_repeat = tf.reshape(
        tf.tile(tf.expand_dims(box_loss, 0), [params['batch_size'],]),
        [params['batch_size'], 1])
    metric_fn_inputs = {
        'cls_loss_repeat': cls_loss_repeat,
        'box_loss_repeat': box_loss_repeat,
        'source_ids': labels['source_ids'],
        'groundtruth_data': labels['groundtruth_data'],
        'image_scales': labels['image_scales'],
    }
    add_metric_fn_inputs(params, cls_outputs, box_outputs, metric_fn_inputs)
    eval_metrics = (metric_fn, metric_fn_inputs)

  checkpoint = params.get('ckpt') or params.get('backbone_ckpt')

  if checkpoint and mode == tf.estimator.ModeKeys.TRAIN:
    # Initialize the model from an EfficientDet or backbone checkpoint.
    if params.get('ckpt') and params.get('backbone_ckpt'):
      raise RuntimeError(
          '--backbone_ckpt and --checkpoint are mutually exclusive')

    if params.get('backbone_ckpt'):
      var_scope = params['backbone_name'] + '/'
      if params['ckpt_var_scope'] is None:
        # Use backbone name as default checkpoint scope.
        ckpt_scope = params['backbone_name'] + '/'
      else:
        ckpt_scope = params['ckpt_var_scope'] + '/'
    else:
      # Load every var in the given checkpoint
      var_scope = ckpt_scope = '/'

    def scaffold_fn():
      """Loads pretrained model through scaffold function."""
      logging.info('restore variables from %s', checkpoint)

      var_map = utils.get_ckpt_var_map(
          ckpt_path=checkpoint,
          ckpt_scope=ckpt_scope,
          var_scope=var_scope,
          var_exclude_expr=params.get('var_exclude_expr', None))

      tf.train.init_from_checkpoint(checkpoint, var_map)

      return tf.train.Scaffold()
  elif mode == tf.estimator.ModeKeys.EVAL and moving_average_decay:
    def scaffold_fn():
      """Load moving average variables for eval."""
      logging.info('Load EMA vars with ema_decay=%f', moving_average_decay)
      restore_vars_dict = ema.variables_to_restore(ema_vars)
      saver = tf.train.Saver(restore_vars_dict)
      return tf.train.Scaffold(saver=saver)
  else:
    scaffold_fn = None

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op,
      eval_metrics=eval_metrics,
      host_call=utils.get_tpu_host_call(global_step, params),
      scaffold_fn=scaffold_fn)


def retinanet_model_fn(features, labels, mode, params):
  """RetinaNet model."""
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=retinanet_arch.retinanet,
      variable_filter_fn=retinanet_arch.remove_variables)


def efficientdet_model_fn(features, labels, mode, params):
  """EfficientDet model."""
  return _model_fn(
      features,
      labels,
      mode,
      params,
      model=efficientdet_arch.efficientdet)


def get_model_arch(model_name='efficientdet-d0'):
  """Get model architecture for a given model name."""
  if 'retinanet' in model_name:
    return retinanet_arch.retinanet

  if 'efficientdet' in model_name:
    return efficientdet_arch.efficientdet

  raise ValueError('Invalide model name {}'.format(model_name))


def get_model_fn(model_name='efficientdet-d0'):
  """Get model fn for a given model name."""
  if 'retinanet' in model_name:
    return retinanet_model_fn

  if 'efficientdet' in model_name:
    return efficientdet_model_fn

  raise ValueError('Invalide model name {}'.format(model_name))
