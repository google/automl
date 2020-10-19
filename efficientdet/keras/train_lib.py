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
"""Training related libraries."""
from concurrent import futures
import math
import os
import re
from absl import logging
import numpy as np
import tensorflow as tf

import inference
import iou_utils
import utils
from keras import anchors
from keras import efficientdet_keras


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule."""
  batch_size = params['batch_size'] * params['num_shards']
  # Learning rate is proportional to the batch size
  params['adjusted_learning_rate'] = (params['learning_rate'] * batch_size / 64)
  steps_per_epoch = params['steps_per_epoch']
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)
  params['total_steps'] = int(params['num_epochs'] * steps_per_epoch)


class StepwiseLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Stepwise learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, first_lr_drop_step: int,
               second_lr_drop_step: int):
    """Build a StepwiseLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      first_lr_drop_step: `int`, First lr decay step.
      second_lr_drop_step: `int`, Second lr decay step.
    """
    super().__init__()
    logging.info('LR schedule method: stepwise')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.first_lr_drop_step = first_lr_drop_step
    self.second_lr_drop_step = second_lr_drop_step

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    learning_rate = tf.where(step < self.lr_warmup_step, linear_warmup,
                             self.adjusted_lr)
    lr_schedule = [[1.0, self.lr_warmup_step], [0.1, self.first_lr_drop_step],
                   [0.01, self.second_lr_drop_step]]
    for mult, start_global_step in lr_schedule:
      learning_rate = tf.where(step < start_global_step, learning_rate,
                               self.adjusted_lr * mult)
    return learning_rate


class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Cosine learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, total_steps: int):
    """Build a CosineLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      total_steps: `int`, Total train steps.
    """
    super().__init__()
    logging.info('LR schedule method: cosine')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

  def __call__(self, step):

    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    cosine_lr = 0.5 * self.adjusted_lr * (
        1 + tf.cos(math.pi * tf.cast(step, tf.float32) / self.decay_steps))
    return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)


class PolynomialLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
  """Polynomial learning rate schedule."""

  def __init__(self, adjusted_lr: float, lr_warmup_init: float,
               lr_warmup_step: int, power: float, total_steps: int):
    """Build a PolynomialLrSchedule.

    Args:
      adjusted_lr: `float`, The initial learning rate.
      lr_warmup_init: `float`, The warm up learning rate.
      lr_warmup_step: `int`, The warm up step.
      power: `float`, power.
      total_steps: `int`, Total train steps.
    """
    super().__init__()
    logging.info('LR schedule method: polynomial')
    self.adjusted_lr = adjusted_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.power = power
    self.total_steps = total_steps

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.adjusted_lr - self.lr_warmup_init)))
    polynomial_lr = self.adjusted_lr * tf.pow(
        1 - (tf.cast(step, dtype=tf.float32) / self.total_steps), self.power)
    return tf.where(step < self.lr_warmup_step, linear_warmup, polynomial_lr)


def learning_rate_schedule(params):
  """Learning rate schedule based on global step."""
  update_learning_rate_schedule_parameters(params)
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'stepwise':
    return StepwiseLrSchedule(params['adjusted_learning_rate'],
                              params['lr_warmup_init'],
                              params['lr_warmup_step'],
                              params['first_lr_drop_step'],
                              params['second_lr_drop_step'])

  if lr_decay_method == 'cosine':
    return CosineLrSchedule(params['adjusted_learning_rate'],
                            params['lr_warmup_init'], params['lr_warmup_step'],
                            params['total_steps'])

  if lr_decay_method == 'polynomial':
    return PolynomialLrSchedule(params['adjusted_learning_rate'],
                                params['lr_warmup_init'],
                                params['lr_warmup_step'],
                                params['poly_lr_power'], params['total_steps'])

  raise ValueError('unknown lr_decay_method: {}'.format(lr_decay_method))


def get_optimizer(params):
  """Get optimizer."""
  learning_rate = learning_rate_schedule(params)
  if params['optimizer'].lower() == 'sgd':
    logging.info('Use SGD optimizer')
    optimizer = tf.keras.optimizers.SGD(
        learning_rate, momentum=params['momentum'])
  elif params['optimizer'].lower() == 'adam':
    logging.info('Use Adam optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate)
  else:
    raise ValueError('optimizers should be adam or sgd')

  moving_average_decay = params['moving_average_decay']
  if moving_average_decay:
    # TODO(tanmingxing): potentially add dynamic_decay for new tfa release.
    from tensorflow_addons import optimizers as tfa_optimizers  # pylint: disable=g-import-not-at-top
    optimizer = tfa_optimizers.MovingAverage(
        optimizer, average_decay=moving_average_decay, dynamic_decay=True)
  if params['mixed_precision']:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
        optimizer, loss_scale='dynamic')
  return optimizer


class DisplayCallback(tf.keras.callbacks.Callback):
  """Display inference result callback."""

  def __init__(self, sample_image, output_dir, update_freq=1):
    super().__init__()
    image_file = tf.io.read_file(sample_image)
    self.sample_image = tf.expand_dims(
        tf.image.decode_jpeg(image_file, channels=3), axis=0)
    self.executor = futures.ThreadPoolExecutor(max_workers=1)
    self.update_freq = update_freq
    self.output_dir = output_dir

  def set_model(self, model: tf.keras.Model):
    self.train_model = model
    with tf.device('/cpu:0'):
      self.model = efficientdet_keras.EfficientDetModel(config=model.config)
    height, width = utils.parse_image_size(model.config.image_size)
    self.model.build((1, height, width, 3))
    self.file_writer = tf.summary.create_file_writer(self.output_dir)
    self.min_score_thresh = self.model.config.nms_configs['score_thresh'] or 0.4
    self.max_boxes_to_draw = (
        self.model.config.nms_configs['max_output_size'] or 100)

  def on_epoch_end(self, epoch, logs=None):
    if epoch % self.update_freq == 0:
      self.executor.submit(self.draw_inference, epoch)

  @tf.function
  def inference(self):
    return self.model(self.sample_image, training=False)

  def draw_inference(self, epoch):
    self.model.set_weights(self.train_model.get_weights())
    boxes, scores, classes, valid_len = self.inference()
    length = valid_len[0]
    image = inference.visualize_image(
        self.sample_image[0],
        boxes[0].numpy()[:length],
        classes[0].numpy().astype(np.int)[:length],
        scores[0].numpy()[:length],
        label_map=self.model.config.label_map,
        min_score_thresh=self.min_score_thresh,
        max_boxes_to_draw=self.max_boxes_to_draw)

    with self.file_writer.as_default():
      tf.summary.image('Test image', tf.expand_dims(image, axis=0), step=epoch)


def get_callbacks(params, profile=False):
  """Get callbacks for given params."""
  tb_callback = tf.keras.callbacks.TensorBoard(
      log_dir=params['model_dir'], profile_batch=2 if profile else 0)
  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
      os.path.join(params['model_dir'], 'ckpt'),
      verbose=1,
      save_weights_only=True)
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', min_delta=0, patience=10, verbose=1)
  callbacks = [tb_callback, ckpt_callback, early_stopping]
  if params.get('sample_image', None):
    display_callback = DisplayCallback(
        params.get('sample_image', None),
        os.path.join(params['model_dir'], 'train'))
    callbacks.append(display_callback)
  return callbacks


class FocalLoss(tf.keras.losses.Loss):
  """Compute the focal loss between `logits` and the golden `target` values.

  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  """

  def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
    """Initialize focal loss.

    Args:
      alpha: A float32 scalar multiplying alpha to the loss from positive
        examples and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.alpha = alpha
    self.gamma = gamma
    self.label_smoothing = label_smoothing

  @tf.autograph.experimental.do_not_convert
  def call(self, y, y_pred):
    """Compute focal loss for y and y_pred.

    Args:
      y: A tuple of (normalizer, y_true), where y_true is the target class.
      y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].

    Returns:
      the focal loss.
    """
    normalizer, y_true = y
    alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
    gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)

    # compute focal loss multipliers before label smoothing, such that it will
    # not blow up the loss.
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma

    # apply label smoothing for cross_entropy for each entry.
    y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute the final loss and return
    return alpha_factor * modulating_factor * ce / normalizer


class BoxLoss(tf.keras.losses.Loss):
  """L2 box regression loss."""

  def __init__(self, delta=0.1, **kwargs):
    """Initialize box loss.

    Args:
      delta: `float`, the point where the huber loss function changes from a
        quadratic to linear. It is typically around the mean value of regression
        target. For instances, the regression targets of 512x512 input with 6
        anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.huber = tf.keras.losses.Huber(
        delta, reduction=tf.keras.losses.Reduction.NONE)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    num_positives, box_targets = y_true
    normalizer = num_positives * 4.0
    mask = tf.cast(box_targets != 0.0, tf.float32)
    box_targets = tf.expand_dims(box_targets, axis=-1)
    box_outputs = tf.expand_dims(box_outputs, axis=-1)
    box_loss = self.huber(box_targets, box_outputs) * mask
    box_loss = tf.reduce_sum(box_loss)
    box_loss /= normalizer
    return box_loss


class BoxIouLoss(tf.keras.losses.Loss):
  """Box iou loss."""

  def __init__(self, iou_loss_type, min_level, max_level, num_scales,
               aspect_ratios, anchor_scale, image_size, **kwargs):
    super().__init__(**kwargs)
    self.iou_loss_type = iou_loss_type
    self.input_anchors = anchors.Anchors(min_level, max_level, num_scales,
                                         aspect_ratios, anchor_scale,
                                         image_size)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    anchor_boxes = tf.tile(
        self.input_anchors.boxes,
        [box_outputs.shape[0] // self.input_anchors.boxes.shape[0], 1])
    num_positives, box_targets = y_true
    box_outputs = anchors.decode_box_outputs(box_outputs, anchor_boxes)
    box_targets = anchors.decode_box_outputs(box_targets, anchor_boxes)
    normalizer = num_positives * 4.0
    box_iou_loss = iou_utils.iou_loss(box_outputs, box_targets,
                                      self.iou_loss_type)
    box_iou_loss = tf.reduce_sum(box_iou_loss) / normalizer
    return box_iou_loss


class EfficientDetNetTrain(efficientdet_keras.EfficientDetNet):
  """A customized trainer for EfficientDet.

  see https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
  """

  def _freeze_vars(self):
    if self.config.var_freeze_expr:
      return [
          v for v in self.trainable_variables
          if not re.match(self.config.var_freeze_expr, v.name)
      ]
    return self.trainable_variables

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self.trainable_variables
        if var_match.match(v.name)
    ])

  def _detection_loss(self, cls_outputs, box_outputs, labels, loss_vals):
    """Computes total detection loss.

    Computes total detection loss including box and class loss from all levels.
    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width, num_anchors].
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in [batch_size, height, width,
        num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundtruth targets.
      loss_vals: A dict of loss values.

    Returns:
      total_loss: an integer tensor representing total loss reducing from
        class and box losses from all levels.
      cls_loss: an integer tensor representing total class loss.
      box_loss: an integer tensor representing total box regression loss.
      box_iou_loss: an integer tensor representing total box iou loss.
    """
    # convert to float32 for loss computing.
    cls_outputs = [tf.cast(i, tf.float32) for i in cls_outputs]
    box_outputs = [tf.cast(i, tf.float32) for i in box_outputs]

    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = tf.reduce_sum(labels['mean_num_positives']) + 1.0
    levels = range(len(cls_outputs))
    cls_losses = []
    box_losses = []
    for level in levels:
      # Onehot encoding for classification labels.
      cls_targets_at_level = tf.one_hot(labels['cls_targets_%d' % (level + 3)],
                                        self.config.num_classes)

      if self.config.data_format == 'channels_first':
        targets_shape = tf.shape(cls_targets_at_level)
        bs = targets_shape[0]
        width = targets_shape[2]
        height = targets_shape[3]
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, -1, width, height])
      else:
        targets_shape = tf.shape(cls_targets_at_level)
        bs = targets_shape[0]
        width = targets_shape[1]
        height = targets_shape[2]
        cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                          [bs, width, height, -1])
      box_targets_at_level = labels['box_targets_%d' % (level + 3)]

      class_loss_layer = self.loss.get('class_loss', None)
      if class_loss_layer:
        cls_loss = class_loss_layer([num_positives_sum, cls_targets_at_level],
                                    cls_outputs[level])

        if self.config.data_format == 'channels_first':
          cls_loss = tf.reshape(
              cls_loss, [bs, -1, width, height, self.config.num_classes])
        else:
          cls_loss = tf.reshape(
              cls_loss, [bs, width, height, -1, self.config.num_classes])
        cls_loss *= tf.cast(
            tf.expand_dims(
                tf.not_equal(labels['cls_targets_%d' % (level + 3)], -2), -1),
            tf.float32)
        cls_losses.append(tf.reduce_sum(cls_loss))

      if self.config.box_loss_weight and self.loss.get('box_loss', None):
        box_loss_layer = self.loss['box_loss']
        box_losses.append(
            box_loss_layer([num_positives_sum, box_targets_at_level],
                           box_outputs[level]))

    if self.config.iou_loss_type:
      box_outputs = tf.concat([tf.reshape(v, [-1, 4]) for v in box_outputs],
                              axis=0)
      box_targets = tf.concat([
          tf.reshape(labels['box_targets_%d' % (level + 3)], [-1, 4])
          for level in levels
      ],
                              axis=0)
      box_iou_loss_layer = self.loss['box_iou_loss']
      box_iou_loss = box_iou_loss_layer([num_positives_sum, box_targets],
                                        box_outputs)
      loss_vals['box_iou_loss'] = box_iou_loss
    else:
      box_iou_loss = 0

    cls_loss = tf.add_n(cls_losses) if cls_losses else 0
    box_loss = tf.add_n(box_losses) if box_losses else 0
    total_loss = (
        cls_loss + self.config.box_loss_weight * box_loss +
        self.config.iou_loss_weight * box_iou_loss)
    loss_vals['det_loss'] = total_loss
    loss_vals['cls_loss'] = cls_loss
    loss_vals['box_loss'] = box_loss
    return total_loss

  def train_step(self, data):
    """Train step.

    Args:
      data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
      A dict record loss info.
    """
    images, labels = data
    with tf.GradientTape() as tape:
      if len(self.config.heads) == 2:
        cls_outputs, box_outputs, seg_outputs = self(images, training=True)
      elif 'object_detection' in self.config.heads:
        cls_outputs, box_outputs = self(images, training=True)
      elif 'segmentation' in self.config.heads:
        seg_outputs, = self(images, training=True)
      reg_l2loss = self._reg_l2_loss(self.config.weight_decay)
      total_loss = reg_l2loss
      loss_vals = {}
      if 'object_detection' in self.config.heads:
        det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                        loss_vals)
        total_loss += det_loss
      if 'segmentation' in self.config.heads:
        seg_loss_layer = self.loss['seg_loss']
        seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
        total_loss += seg_loss
        loss_vals['seg_loss'] = seg_loss
      if isinstance(self.optimizer,
                    tf.keras.mixed_precision.experimental.LossScaleOptimizer):
        scaled_loss = self.optimizer.get_scaled_loss(total_loss)
      else:
        scaled_loss = total_loss
    loss_vals['loss'] = total_loss
    trainable_vars = self._freeze_vars()
    scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
    if isinstance(self.optimizer,
                  tf.keras.mixed_precision.experimental.LossScaleOptimizer):
      gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
    else:
      gradients = scaled_gradients
    if self.config.clip_gradients_norm > 0:
      gradients, gnorm = tf.clip_by_global_norm(gradients,
                                                self.config.clip_gradients_norm)
      loss_vals['gnorm'] = gnorm
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss_vals

  def test_step(self, data):
    """Test step.

    Args:
      data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
      A dict record loss info.
    """
    images, labels = data
    if len(self.config.heads) == 2:
      cls_outputs, box_outputs, seg_outputs = self(images, training=False)
    elif 'object_detection' in self.config.heads:
      cls_outputs, box_outputs = self(images, training=False)
    elif 'segmentation' in self.config.heads:
      seg_outputs, = self(images, training=False)
    reg_l2loss = self._reg_l2_loss(self.config.weight_decay)
    total_loss = reg_l2loss
    loss_vals = {}
    if 'object_detection' in self.config.heads:
      det_loss = self._detection_loss(cls_outputs, box_outputs, labels,
                                      loss_vals)
      total_loss += det_loss
    if 'segmentation' in self.config.heads:
      seg_loss_layer = self.loss['seg_loss']
      seg_loss = seg_loss_layer(labels['image_masks'], seg_outputs)
      total_loss += seg_loss
      loss_vals['seg_loss'] = seg_loss
    loss_vals['loss'] = total_loss
    return loss_vals
