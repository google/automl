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
"""Keras implementation of efficientdet."""
import functools
from absl import logging
import numpy as np
import tensorflow as tf

import efficientdet_arch as legacy_arch
import hparams_config
import utils
from keras import utils_keras


class BiFPNLayer(tf.keras.layers.Layer):
  """A Keras Layer implementing Bidirectional Feature Pyramids."""

  def __init__(self, min_level: int, max_level: int, image_size: int,
               fpn_weight_method: str, apply_bn_for_resampling: bool,
               is_training_bn: bool, conv_after_downsample: bool,
               use_native_resize_op: bool, data_format: str, pooling_type: str,
               fpn_num_filters: int, conv_bn_act_pattern: bool, act_type: str,
               separable_conv: bool, strategy: bool, fpn_name: str, **kwargs):
    self.min_level = min_level
    self.max_level = max_level
    self.image_size = image_size
    self.feat_sizes = utils.get_feat_sizes(image_size, max_level)

    self.fpn_weight_method = fpn_weight_method
    self.apply_bn_for_resampling = apply_bn_for_resampling
    self.is_training_bn = is_training_bn
    self.conv_after_downsample = conv_after_downsample
    self.use_native_resize_op = use_native_resize_op
    self.data_format = data_format
    self.fpn_num_filters = fpn_num_filters
    self.pooling_type = pooling_type
    self.conv_bn_act_pattern = conv_bn_act_pattern
    self.act_type = act_type
    self.strategy = strategy
    self.separable_conv = separable_conv

    self.fpn_config = None
    self.fpn_name = fpn_name

    super(BiFPNLayer, self).__init__(**kwargs)

  def call(self, feats):
    # @TODO: Implement this with keras logic
    return legacy_arch.build_bifpn_layer(feats, self.feat_sizes, self)

  def get_config(self):
    base_config = super(BiFPNLayer, self).get_config()

    return {
        **base_config,
        'min_level': self.min_level,
        'max_level': self.max_level,
        'image_size': self.image_size,
        'fpn_name': self.fpn_name,
        'fpn_weight_method': self.fpn_weight_method,
        'apply_bn_for_resampling': self.apply_bn_for_resampling,
        'is_training_bn': self.is_training_bn,
        'conv_after_downsample': self.conv_after_downsample,
        'use_native_resize_op': self.use_native_resize_op,
        'data_format': self.data_format,
        'pooling_type': self.pooling_type,
        'fpn_num_filters': self.fpn_num_filters,
        'conv_bn_act_pattern': self.conv_bn_act_pattern,
        'act_type': self.act_type,
        'separable_conv': self.separable_conv,
        'strategy': self.strategy,
    }


class ResampleFeatureMap(tf.keras.layers.Layer):
  """Resample feature map for downsampling or upsampling."""

  def __init__(self,
               target_height,
               target_width,
               target_num_channels,
               apply_bn=False,
               is_training=None,
               conv_after_downsample=False,
               use_native_resize_op=False,
               pooling_type=None,
               strategy=None,
               data_format=None,
               name='resample_feature_map'):
    super(ResampleFeatureMap, self).__init__(name='resample_{}'.format(name))
    self.apply_bn = apply_bn
    self.is_training = is_training
    self.data_format = data_format
    self.target_num_channels = target_num_channels
    self.target_height = target_height
    self.target_width = target_width
    self.strategy = strategy
    self.conv_after_downsample = conv_after_downsample
    self.use_native_resize_op = use_native_resize_op
    self.pooling_type = pooling_type
    self.conv2d = tf.keras.layers.Conv2D(self.target_num_channels, (1, 1),
                                         padding='same',
                                         data_format=self.data_format)
    self.bn = utils_keras.build_batch_norm(is_training_bn=self.is_training,
                                           data_format=self.data_format,
                                           strategy=self.strategy,
                                           name='bn')

  def build(self, input_shape):
    """Resample input feature map to have target number of channels and size."""
    if self.data_format == 'channels_first':
      _, num_channels, height, width = input_shape.as_list()
    else:
      _, height, width, num_channels = input_shape.as_list()

    if height is None or width is None or num_channels is None:
      raise ValueError(
          'shape[1] or shape[2] or shape[3] of feat is None (shape:{}).'.format(
              input_shape.as_list()))
    if self.apply_bn and self.is_training is None:
      raise ValueError('If BN is applied, need to provide is_training')
    self.num_channels = num_channels
    self.height = height
    self.width = width
    height_stride_size = int((self.height - 1) // self.target_height + 1)
    width_stride_size = int((self.width - 1) // self.target_width + 1)

    if self.pooling_type == 'max' or self.pooling_type is None:
      # Use max pooling in default.
      self.pool2d = tf.keras.layers.MaxPooling2D(
          pool_size=[height_stride_size + 1, width_stride_size + 1],
          strides=[height_stride_size, width_stride_size],
          padding='SAME',
          data_format=self.data_format)
    elif self.pooling_type == 'avg':
      self.pool2d = tf.keras.layers.AveragePooling2D(
          pool_size=[height_stride_size + 1, width_stride_size + 1],
          strides=[height_stride_size, width_stride_size],
          padding='SAME',
          data_format=self.data_format)
    else:
      raise ValueError('Unknown pooling type: {}'.format(self.pooling_type))

    height_scale = self.target_height // self.height
    width_scale = self.target_width // self.width
    if (self.use_native_resize_op or self.target_height % self.height != 0 or
        self.target_width % self.width != 0):
      self.upsample2d = tf.keras.layers.UpSampling2D(
          (height_scale, width_scale), data_format=self.data_format)
    else:
      self.upsample2d = functools.partial(legacy_arch.nearest_upsampling,
                                          height_scale=height_scale,
                                          width_scale=width_scale,
                                          data_format=self.data_format)
    super(ResampleFeatureMap, self).build(input_shape)

  def _maybe_apply_1x1(self, feat):
    """Apply 1x1 conv to change layer width if necessary."""
    if self.num_channels != self.target_num_channels:
      feat = self.conv2d(feat)
      if self.apply_bn:
        feat = self.bn(feat, training=self.is_training)
    return feat

  def call(self, feat):
    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if self.height > self.target_height and self.width > self.target_width:
      if not self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat)
      feat = self.pool2d(feat)
      if self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat)
    elif self.height <= self.target_height and self.width <= self.target_width:
      feat = self._maybe_apply_1x1(feat)
      if self.height < self.target_height or self.width < self.target_width:
        feat = self.upsample2d(feat)
    else:
      raise ValueError(
          'Incompatible target feature map size: target_height: {},'
          'target_width: {}'.format(self.target_height, self.target_width))

    return feat

  def get_config(self):
    config = {
        'apply_bn': self.apply_bn,
        'is_training': self.is_training,
        'data_format': self.data_format,
        'target_num_channels': self.target_num_channels,
        'target_height': self.target_height,
        'target_width': self.target_width,
        'strategy': self.strategy,
        'conv_after_downsample': self.conv_after_downsample,
        'use_native_resize_op': self.use_native_resize_op,
        'pooling_type': self.pooling_type,
    }
    base_config = super(ResampleFeatureMap, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ClassNet(tf.keras.layers.Layer):
  """Object class prediction network."""

  def __init__(self,
               num_classes=90,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training=False,
               act_type='swish',
               repeats=4,
               separable_conv=True,
               survival_prob=None,
               strategy=None,
               data_format='channels_last',
               name='class_net',
               **kwargs):
    """Initialize the ClassNet.

    Args:
      num_classes: number of classes.
      num_anchors: number of anchors.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      is_training: True if we train the BatchNorm.
      act_type: String of the activation used.
      repeats: number of intermediate layers.
      separable_conv: True to use separable_conv instead of conv2D.
      survival_prob: if a value is set then drop connect will be used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      data_format: string of 'channel_first' or 'channels_last'.
      name: the name of this layerl.
      **kwargs: other parameters.
    """

    super(ClassNet, self).__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.separable_conv = separable_conv
    self.is_training = is_training
    self.survival_prob = survival_prob
    self.act_type = act_type
    self.strategy = strategy
    self.data_format = data_format
    self.use_dc = survival_prob and is_training

    self.conv_ops = []
    self.bns = []

    for i in range(self.repeats):
      # If using SeparableConv2D
      if self.separable_conv:
        self.conv_ops.append(
            tf.keras.layers.SeparableConv2D(
                filters=self.num_filters,
                depth_multiplier=1,
                pointwise_initializer=tf.initializers.VarianceScaling(),
                depthwise_initializer=tf.initializers.VarianceScaling(),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='class-%d' % i))
      # If using Conv2d
      else:
        self.conv_ops.append(
            tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='class-%d' % i))

      bn_per_level = {}
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level[level] = utils_keras.build_batch_norm(
            is_training_bn=self.is_training,
            init_zero=False,
            strategy=self.strategy,
            data_format=self.data_format,
            name='class-%d-bn-%d' % (i, level),
        )
      self.bns.append(bn_per_level)

    if self.separable_conv:
      self.classes = tf.keras.layers.SeparableConv2D(
          filters=self.num_classes * self.num_anchors,
          depth_multiplier=1,
          pointwise_initializer=tf.initializers.VarianceScaling(),
          depthwise_initializer=tf.initializers.VarianceScaling(),
          data_format=self.data_format,
          kernel_size=3,
          activation=None,
          bias_initializer=tf.constant_initializer(-np.math.log((1 - 0.01) /
                                                                0.01)),
          padding='same',
          name='class-predict')

    else:
      self.classes = tf.keras.layers.Conv2D(
          filters=self.num_classes * self.num_anchors,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          data_format=self.data_format,
          kernel_size=3,
          activation=None,
          bias_initializer=tf.constant_initializer(-np.math.log((1 - 0.01) /
                                                                0.01)),
          padding='same',
          name='class-predict')

  def call(self, inputs, **kwargs):
    """Call ClassNet."""

    class_outputs = {}
    for level in range(self.min_level, self.max_level + 1):
      image = inputs[level]
      for i in range(self.repeats):
        original_image = image
        image = self.conv_ops[i](image)
        image = self.bns[i][level](image, training=self.is_training)
        if self.act_type:
          image = utils.activation_fn(image, self.act_type)
        if i > 0 and self.use_dc:
          image = utils.drop_connect(image, self.is_training,
                                     self.survival_prob)
          image = image + original_image

      class_outputs[level] = self.classes(image)

    return class_outputs

  def get_config(self):
    base_config = super(ClassNet, self).get_config()

    return {
        **base_config,
        'num_classes': self.num_classes,
        'num_anchors': self.num_anchors,
        'num_filters': self.num_filters,
        'min_level': self.min_level,
        'max_level': self.max_level,
        'is_training': self.is_training,
        'act_type': self.act_type,
        'repeats': self.repeats,
        'separable_conv': self.separable_conv,
        'survival_prob': self.survival_prob,
        'strategy': self.strategy,
        'data_format': self.data_format,
    }


class BoxNet(tf.keras.layers.Layer):
  """Box regression network."""

  def __init__(self,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training=False,
               act_type='swish',
               repeats=4,
               separable_conv=True,
               survival_prob=None,
               strategy=None,
               data_format='channels_last',
               name='box_net',
               **kwargs):
    """Initialize BoxNet.

    Args:
      num_anchors: number of  anchors used.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      is_training: True if we train the BatchNorm.
      act_type: String of the activation used.
      repeats: number of "intermediate" layers.
      separable_conv: True to use separable_conv instead of conv2D.
      survival_prob: if a value is set then drop connect will be used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      data_format: string of 'channel_first' or 'channels_last'.
      name: Name of the layer.
      **kwargs: other parameters.
    """

    super(BoxNet, self).__init__(name=name, **kwargs)

    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.separable_conv = separable_conv
    self.is_training = is_training
    self.survival_prob = survival_prob
    self.act_type = act_type
    self.strategy = strategy
    self.data_format = data_format
    self.use_dc = survival_prob and is_training

    self.conv_ops = []
    self.bns = []

    for i in range(self.repeats):
      # If using SeparableConv2D
      if self.separable_conv:
        self.conv_ops.append(
            tf.keras.layers.SeparableConv2D(
                filters=self.num_filters,
                depth_multiplier=1,
                pointwise_initializer=tf.initializers.VarianceScaling(),
                depthwise_initializer=tf.initializers.VarianceScaling(),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-%d' % i))
      # If using Conv2d
      else:
        self.conv_ops.append(
            tf.keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-%d' % i))

      bn_per_level = {}
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level[level] = utils_keras.build_batch_norm(
            is_training_bn=self.is_training,
            init_zero=False,
            strategy=self.strategy,
            data_format=self.data_format,
            name='box-%d-bn-%d' % (i, level))
      self.bns.append(bn_per_level)

    if self.separable_conv:
      self.boxes = tf.keras.layers.SeparableConv2D(
          filters=4 * self.num_anchors,
          depth_multiplier=1,
          pointwise_initializer=tf.initializers.VarianceScaling(),
          depthwise_initializer=tf.initializers.VarianceScaling(),
          data_format=self.data_format,
          kernel_size=3,
          activation=None,
          bias_initializer=tf.zeros_initializer(),
          padding='same',
          name='box-predict')

    else:
      self.boxes = tf.keras.layers.Conv2D(
          filters=4 * self.num_anchors,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          data_format=self.data_format,
          kernel_size=3,
          activation=None,
          bias_initializer=tf.zeros_initializer(),
          padding='same',
          name='box-predict')

  def call(self, inputs, **kwargs):
    """Call boxnet."""
    box_outputs = {}
    for level in range(self.min_level, self.max_level + 1):
      image = inputs[level]
      for i in range(self.repeats):
        original_image = image
        image = self.conv_ops[i](image)
        image = self.bns[i][level](image, training=self.is_training)
        if self.act_type:
          image = utils.activation_fn(image, self.act_type)
        if i > 0 and self.use_dc:
          image = utils.drop_connect(image, self.is_training,
                                     self.survival_prob)
          image = image + original_image

      box_outputs[level] = self.boxes(image)

    return box_outputs

  def get_config(self):
    base_config = super(BoxNet, self).get_config()

    return {
        **base_config,
        'num_anchors': self.num_anchors,
        'num_filters': self.num_filters,
        'min_level': self.min_level,
        'max_level': self.max_level,
        'is_training': self.is_training,
        'act_type': self.act_type,
        'repeats': self.repeats,
        'separable_conv': self.separable_conv,
        'survival_prob': self.survival_prob,
        'strategy': self.strategy,
        'data_format': self.data_format,
    }


def build_class_and_box_outputs(feats, config):
  """Builds box net and class net.

  Args:
   feats: input tensor.
   config: a dict-like config, including all parameters.

  Returns:
   A tuple (class_outputs, box_outputs) for class/box predictions.
  """
  num_anchors = len(config.aspect_ratios) * config.num_scales
  num_filters = config.fpn_num_filters
  class_outputs = ClassNet(num_classes=config.num_classes,
                           num_anchors=num_anchors,
                           num_filters=num_filters,
                           min_level=config.min_level,
                           max_level=config.max_level,
                           is_training=config.is_training_bn,
                           act_type=config.act_type,
                           repeats=config.box_class_repeats,
                           separable_conv=config.separable_conv,
                           survival_prob=config.survival_prob,
                           strategy=config.strategy,
                           data_format=config.data_format)(feats)

  box_outputs = BoxNet(num_anchors=num_anchors,
                       num_filters=num_filters,
                       min_level=config.min_level,
                       max_level=config.max_level,
                       is_training=config.is_training_bn,
                       act_type=config.act_type,
                       repeats=config.box_class_repeats,
                       separable_conv=config.separable_conv,
                       survival_prob=config.survival_prob,
                       strategy=config.strategy,
                       data_format=config.data_format)(feats)

  return class_outputs, box_outputs


def efficientdet(features, model_name=None, config=None, **kwargs):
  """Build EfficientDet model.

  Args:
    features: input tensor.
    model_name: String of the model (eg. efficientdet-d0)
    config: Dict of parameters for the network
    **kwargs: other parameters.

  Returns:
    A tuple (class_outputs, box_outputs) for predictions.
  """
  if not config and not model_name:
    raise ValueError('please specify either model name or config')

  if not config:
    config = hparams_config.get_efficientdet_config(model_name)
  elif isinstance(config, dict):
    config = hparams_config.Config(config)  # wrap dict in Config object

  if kwargs:
    config.override(kwargs)

  logging.info(config)

  # build backbone features.
  features = legacy_arch.build_backbone(features, config)
  logging.info('backbone params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  # build feature network.
  fpn_feats = legacy_arch.build_feature_network(features, config)
  logging.info('backbone+fpn params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  # build class and box predictions.
  class_outputs, box_outputs = build_class_and_box_outputs(fpn_feats, config)
  logging.info('backbone+fpn+box params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  return class_outputs, box_outputs
