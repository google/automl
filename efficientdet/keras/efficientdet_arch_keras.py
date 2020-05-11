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
import tensorflow.compat.v1 as tf

import efficientdet_arch as legacy_arch
import utils


class BiFPNLayer(tf.keras.layers.Layer):
  """A Keras Layer implementing Bidirectional Feature Pyramids."""

  def __init__(self, min_level: int, max_level: int, image_size: int,
               fpn_weight_method: str, apply_bn_for_resampling: bool,
               is_training_bn: bool, conv_after_downsample: bool,
               use_native_resize_op: bool, data_format: str, pooling_type: str,
               fpn_num_filters: int, conv_bn_act_pattern: bool, act_type: str,
               separable_conv: bool, use_tpu: bool, fpn_name: str, **kwargs):
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
    self.use_tpu = use_tpu
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
        'use_tpu': self.use_tpu,
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
               use_tpu=False,
               data_format=None,
               name='resample_feature_map'):
    super(ResampleFeatureMap, self).__init__(name='resample_{}'.format(name))
    self.apply_bn = apply_bn
    self.is_training = is_training
    self.data_format = data_format
    self.target_num_channels = target_num_channels
    self.target_height = target_height
    self.target_width = target_width
    self.use_tpu = use_tpu
    self.conv_after_downsample = conv_after_downsample
    self.use_native_resize_op = use_native_resize_op
    self.pooling_type = pooling_type
    self.conv2d = tf.keras.layers.Conv2D(
        self.target_num_channels, (1, 1),
        padding='same',
        data_format=self.data_format)

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
      self.upsample2d = functools.partial(
          legacy_arch.nearest_upsampling,
          height_scale=height_scale,
          width_scale=width_scale,
          data_format=self.data_format)
    super(ResampleFeatureMap, self).build(input_shape)

  def _maybe_apply_1x1(self, feat):
    """Apply 1x1 conv to change layer width if necessary."""
    if self.num_channels != self.target_num_channels:
      feat = self.conv2d(feat)
      if self.apply_bn:
        feat = utils.batch_norm_act(
            feat,
            is_training_bn=self.is_training,
            act_type=None,
            data_format=self.data_format,
            use_tpu=self.use_tpu,
            name='bn')
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
        'use_tpu': self.use_tpu,
        'conv_after_downsample': self.conv_after_downsample,
        'use_native_resize_op': self.use_native_resize_op,
        'pooling_type': self.pooling_type,
    }
    base_config = super(ResampleFeatureMap, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
