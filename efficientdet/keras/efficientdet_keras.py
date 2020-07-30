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

import dataloader
import hparams_config
import utils
from backbone import backbone_factory
from backbone import efficientnet_builder
from keras import fpn_configs
from keras import postprocess
from keras import utils_keras
# pylint: disable=arguments-differ  # fo keras layers.


class FNode(tf.keras.layers.Layer):
  """A Keras Layer implementing BiFPN Node."""

  def __init__(self,
               new_node_height,
               new_node_width,
               inputs_offsets,
               fpn_num_filters,
               apply_bn_for_resampling,
               is_training_bn,
               conv_after_downsample,
               conv_bn_act_pattern,
               separable_conv,
               act_type,
               strategy,
               weight_method,
               data_format,
               name='fnode'):
    super().__init__(name=name)
    self.new_node_height = new_node_height
    self.new_node_width = new_node_width
    self.inputs_offsets = inputs_offsets
    self.fpn_num_filters = fpn_num_filters
    self.apply_bn_for_resampling = apply_bn_for_resampling
    self.separable_conv = separable_conv
    self.act_type = act_type
    self.is_training_bn = is_training_bn
    self.conv_after_downsample = conv_after_downsample
    self.strategy = strategy
    self.data_format = data_format
    self.weight_method = weight_method
    self.conv_bn_act_pattern = conv_bn_act_pattern
    self.resample_feature_maps = []
    self.vars = []

  def fuse_features(self, nodes):
    """Fuse features from different resolutions and return a weighted sum.

    Args:
      nodes: a list of tensorflow features at different levels

    Returns:
      A tensor denoting the fused feature.
    """
    dtype = nodes[0].dtype

    if self.weight_method == 'attn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'fastattn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)
      weights_sum = tf.add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = tf.add_n(nodes)
    elif self.weight_method == 'channel_attn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)
      normalized_weights = tf.nn.softmax(tf.stack(edge_weights, -1), axis=-1)
      nodes = tf.stack(nodes, axis=-1)
      new_node = tf.reduce_sum(nodes * normalized_weights, -1)
    elif self.weight_method == 'channel_fastattn':
      edge_weights = []
      for var in self.vars:
        var = tf.cast(var, dtype=dtype)
        edge_weights.append(var)

      weights_sum = tf.add_n(edge_weights)
      nodes = [
          nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
          for i in range(len(nodes))
      ]
      new_node = tf.add_n(nodes)
    elif self.weight_method == 'sum':
      new_node = tf.add_n(nodes)
    else:
      raise ValueError('unknown weight_method %s' % self.weight_method)

    return new_node

  def _add_wsm(self, initializer):
    for i, _ in enumerate(self.inputs_offsets):
      name = 'WSM' + ('' if i == 0 else '_' + str(i))
      self.vars.append(
          self.add_weight(
              initializer=initializer, name=name,
              trainable=self.is_training_bn))

  def build(self, feats_shape):
    for i, input_offset in enumerate(self.inputs_offsets):
      name = 'resample_{}_{}_{}'.format(i, input_offset, len(feats_shape))
      resample_feature_map = ResampleFeatureMap(
          self.new_node_height,
          self.new_node_width,
          self.fpn_num_filters,
          self.apply_bn_for_resampling,
          self.is_training_bn,
          self.conv_after_downsample,
          strategy=self.strategy,
          data_format=self.data_format,
          name=name)
      self.resample_feature_maps.append(resample_feature_map)
    if self.weight_method == 'attn':
      self._add_wsm('ones')
    elif self.weight_method == 'fastattn':
      self._add_wsm('ones')
    elif self.weight_method == 'channel_attn':
      num_filters = int(self.fpn_num_filters)
      self._add_wsm(lambda: tf.ones([num_filters]))
    elif self.weight_method == 'channel_fastattn':
      num_filters = int(self.fpn_num_filters)
      self._add_wsm(lambda: tf.ones([num_filters]))
    self.op_after_combine = OpAfterCombine(
        self.is_training_bn,
        self.conv_bn_act_pattern,
        self.separable_conv,
        self.fpn_num_filters,
        self.act_type,
        self.data_format,
        self.strategy,
        name='op_after_combine{}'.format(len(feats_shape)))
    self.built = True

  def call(self, feats, training):
    nodes = []
    for i, input_offset in enumerate(self.inputs_offsets):
      input_node = feats[input_offset]
      input_node = self.resample_feature_maps[i](input_node, training)
      nodes.append(input_node)
    new_node = self.fuse_features(nodes)
    new_node = self.op_after_combine(new_node)
    feats.append(new_node)
    return feats


class OpAfterCombine(tf.keras.layers.Layer):
  """Operation after combining input features during feature fusiong."""

  def __init__(self,
               is_training_bn,
               conv_bn_act_pattern,
               separable_conv,
               fpn_num_filters,
               act_type,
               data_format,
               strategy,
               name='op_after_combine'):
    super().__init__(name=name)
    self.conv_bn_act_pattern = conv_bn_act_pattern
    self.separable_conv = separable_conv
    self.fpn_num_filters = fpn_num_filters
    self.act_type = act_type
    self.data_format = data_format
    self.strategy = strategy
    self.is_training_bn = is_training_bn
    if self.separable_conv:
      conv2d_layer = functools.partial(
          tf.keras.layers.SeparableConv2D, depth_multiplier=1)
    else:
      conv2d_layer = tf.keras.layers.Conv2D

    self.conv_op = conv2d_layer(
        filters=fpn_num_filters,
        kernel_size=(3, 3),
        padding='same',
        use_bias=not self.conv_bn_act_pattern,
        data_format=self.data_format,
        name='conv')
    self.bn = utils_keras.build_batch_norm(
        is_training_bn=self.is_training_bn,
        data_format=self.data_format,
        strategy=self.strategy,
        name='bn')

  def call(self, new_node, training):
    if not self.conv_bn_act_pattern:
      new_node = utils.activation_fn(new_node, self.act_type)
    new_node = self.conv_op(new_node)
    new_node = self.bn(new_node, training=training)
    if self.conv_bn_act_pattern:
      new_node = utils.activation_fn(new_node, self.act_type)
    return new_node


class ResampleFeatureMap(tf.keras.layers.Layer):
  """Resample feature map for downsampling or upsampling."""

  def __init__(self,
               target_height,
               target_width,
               target_num_channels,
               apply_bn=False,
               is_training_bn=None,
               conv_after_downsample=False,
               strategy=None,
               data_format=None,
               name='resample_p0'):
    super().__init__(name=name)
    self.apply_bn = apply_bn
    self.is_training_bn = is_training_bn
    self.data_format = data_format
    self.target_num_channels = target_num_channels
    self.target_height = target_height
    self.target_width = target_width
    self.strategy = strategy
    self.conv_after_downsample = conv_after_downsample
    self.conv2d = tf.keras.layers.Conv2D(
        self.target_num_channels, (1, 1),
        padding='same',
        data_format=self.data_format,
        name='conv2d')
    self.bn = utils_keras.build_batch_norm(
        is_training_bn=self.is_training_bn,
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
    if self.apply_bn and self.is_training_bn is None:
      raise ValueError('If BN is applied, need to provide is_training_bn')
    self.num_channels = num_channels
    self.height = height
    self.width = width
    height_stride_size = int((self.height - 1) // self.target_height + 1)
    width_stride_size = int((self.width - 1) // self.target_width + 1)

    # Use max pooling in default.
    self.pool2d = tf.keras.layers.MaxPooling2D(
        pool_size=[height_stride_size + 1, width_stride_size + 1],
        strides=[height_stride_size, width_stride_size],
        padding='SAME',
        data_format=self.data_format)

    height_scale = self.target_height // self.height
    width_scale = self.target_width // self.width
    self.upsample2d = tf.keras.layers.UpSampling2D((height_scale, width_scale),
                                                   data_format=self.data_format)
    super().build(input_shape)

  def _maybe_apply_1x1(self, feat, training):
    """Apply 1x1 conv to change layer width if necessary."""
    if self.num_channels != self.target_num_channels:
      feat = self.conv2d(feat)
      if self.apply_bn:
        feat = self.bn(feat, training=training)
    return feat

  def call(self, feat, training):
    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if self.height > self.target_height and self.width > self.target_width:
      if not self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat, training)
      feat = self.pool2d(feat)
      if self.conv_after_downsample:
        feat = self._maybe_apply_1x1(feat, training)
    elif self.height <= self.target_height and self.width <= self.target_width:
      feat = self._maybe_apply_1x1(feat, training)
      if self.height < self.target_height or self.width < self.target_width:
        feat = self.upsample2d(feat)
    else:
      raise ValueError(
          'Incompatible target feature map size: target_height: {},'
          'target_width: {}'.format(self.target_height, self.target_width))

    return feat


class ClassNet(tf.keras.layers.Layer):
  """Object class prediction network."""

  def __init__(self,
               num_classes=90,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training_bn=False,
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
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      repeats: number of intermediate layers.
      separable_conv: True to use separable_conv instead of conv2D.
      survival_prob: if a value is set then drop connect will be used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      data_format: string of 'channel_first' or 'channels_last'.
      name: the name of this layerl.
      **kwargs: other parameters.
    """

    super().__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.separable_conv = separable_conv
    self.is_training_bn = is_training_bn
    self.survival_prob = survival_prob
    self.act_type = act_type
    self.strategy = strategy
    self.data_format = data_format
    self.conv_ops = []
    self.bns = []
    if separable_conv:
      conv2d_layer = functools.partial(
          tf.keras.layers.SeparableConv2D,
          depth_multiplier=1,
          data_format=data_format,
          pointwise_initializer=tf.initializers.VarianceScaling(),
          depthwise_initializer=tf.initializers.VarianceScaling())
    else:
      conv2d_layer = functools.partial(
          tf.keras.layers.Conv2D,
          data_format=data_format,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    for i in range(self.repeats):
      # If using SeparableConv2D
      self.conv_ops.append(
          conv2d_layer(
              self.num_filters,
              kernel_size=3,
              bias_initializer=tf.zeros_initializer(),
              activation=None,
              padding='same',
              name='class-%d' % i))

      bn_per_level = []
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level.append(
            utils_keras.build_batch_norm(
                is_training_bn=self.is_training_bn,
                strategy=self.strategy,
                data_format=self.data_format,
                name='class-%d-bn-%d' % (i, level),
            ))
      self.bns.append(bn_per_level)

    self.classes = conv2d_layer(
        num_classes * num_anchors,
        kernel_size=3,
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        padding='same',
        name='class-predict')

  def call(self, inputs, training, **kwargs):
    """Call ClassNet."""

    class_outputs = []
    for level_id in range(0, self.max_level - self.min_level + 1):
      image = inputs[level_id]
      for i in range(self.repeats):
        original_image = image
        image = self.conv_ops[i](image)
        image = self.bns[i][level_id](image, training=training)
        if self.act_type:
          image = utils.activation_fn(image, self.act_type)
        if i > 0 and self.survival_prob:
          image = utils.drop_connect(image, training, self.survival_prob)
          image = image + original_image

      class_outputs.append(self.classes(image))

    return class_outputs


class BoxNet(tf.keras.layers.Layer):
  """Box regression network."""

  def __init__(self,
               num_anchors=9,
               num_filters=32,
               min_level=3,
               max_level=7,
               is_training_bn=False,
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
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      repeats: number of "intermediate" layers.
      separable_conv: True to use separable_conv instead of conv2D.
      survival_prob: if a value is set then drop connect will be used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      data_format: string of 'channel_first' or 'channels_last'.
      name: Name of the layer.
      **kwargs: other parameters.
    """

    super().__init__(name=name, **kwargs)

    self.num_anchors = num_anchors
    self.num_filters = num_filters
    self.min_level = min_level
    self.max_level = max_level
    self.repeats = repeats
    self.separable_conv = separable_conv
    self.is_training_bn = is_training_bn
    self.survival_prob = survival_prob
    self.act_type = act_type
    self.strategy = strategy
    self.data_format = data_format

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

      bn_per_level = []
      for level in range(self.min_level, self.max_level + 1):
        bn_per_level.append(
            utils_keras.build_batch_norm(
                is_training_bn=self.is_training_bn,
                strategy=self.strategy,
                data_format=self.data_format,
                name='box-%d-bn-%d' % (i, level)))
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

  def call(self, inputs, training):
    """Call boxnet."""
    box_outputs = []
    for level_id in range(0, self.max_level - self.min_level + 1):
      image = inputs[level_id]
      for i in range(self.repeats):
        original_image = image
        image = self.conv_ops[i](image)
        image = self.bns[i][level_id](image, training=training)
        if self.act_type:
          image = utils.activation_fn(image, self.act_type)
        if i > 0 and self.survival_prob:
          image = utils.drop_connect(image, training, self.survival_prob)
          image = image + original_image

      box_outputs.append(self.boxes(image))

    return box_outputs


class SegmentationHead(tf.keras.layers.Layer):
  """Keras layer for semantic segmentation head."""

  def __init__(self,
               num_classes,
               num_filters,
               min_level,
               max_level,
               data_format,
               is_training_bn,
               act_type,
               strategy,
               **kwargs):
    """Initialize SegmentationHead.

    Args:
      num_classes: number of classes.
      num_filters: number of filters for "intermediate" layers.
      min_level: minimum level for features.
      max_level: maximum level for features.
      data_format: string of 'channel_first' or 'channels_last'.
      is_training_bn: True if we train the BatchNorm.
      act_type: String of the activation used.
      strategy: string to specify training strategy for TPU/GPU/CPU.
      **kwargs: other parameters.
    """
    super().__init__(**kwargs)
    self.act_type = act_type
    self.con2d_ts = []
    self.con2d_t_bns = []
    for _ in range(max_level - min_level):
      self.con2d_ts.append(
          tf.keras.layers.Conv2DTranspose(
              num_filters,
              3,
              strides=2,
              padding='same',
              data_format=data_format,
              use_bias=False))
      self.con2d_t_bns.append(
          utils_keras.build_batch_norm(
              is_training_bn=is_training_bn,
              data_format=data_format,
              strategy=strategy,
              name='bn'))
    self.head_transpose = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2, padding='same')

  def call(self, feats, training):
    x = feats[-1]
    skips = list(reversed(feats[:-1]))

    for con2d_t, con2d_t_bn, skip in zip(self.con2d_ts, self.con2d_t_bns,
                                         skips):
      x = con2d_t(x)
      x = con2d_t_bn(x, training)
      x = utils.activation_fn(x, self.act_type)
      x = tf.concat([x, skip], axis=-1)

    # This is the last layer of the model
    return self.head_transpose(x)  # 64x64 -> 128x128


class FPNCells(tf.keras.layers.Layer):
  """FPN cells."""

  def __init__(self, config, name='fpn_cells'):
    super().__init__(name=name)
    self.config = config
    self.feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)

    if config.fpn_config:
      self.fpn_config = config.fpn_config
    else:
      self.fpn_config = fpn_configs.get_fpn_config(config.fpn_name,
                                                   config.min_level,
                                                   config.max_level,
                                                   config.fpn_weight_method)

    self.cells = [
        FPNCell(self.feat_sizes, self.config, name='cell_%d' % rep)
        for rep in range(self.config.fpn_cell_repeats)
    ]

  def call(self, feats, training):
    for cell in self.cells:
      cell_feats = cell(feats, training)
      min_level = self.config.min_level
      max_level = self.config.max_level

      feats = []
      for level in range(min_level, max_level + 1):
        for i, fnode in enumerate(reversed(self.fpn_config.nodes)):
          if fnode['feat_level'] == level:
            feats.append(cell_feats[-1 - i])
            break

    return feats


class FPNCell(tf.keras.layers.Layer):
  """A single FPN cell."""

  def __init__(self, feat_sizes, config, name='fpn_cell'):
    super().__init__(name=name)
    self.feat_sizes = feat_sizes
    self.config = config
    if config.fpn_config:
      self.fpn_config = config.fpn_config
    else:
      self.fpn_config = fpn_configs.get_fpn_config(config.fpn_name,
                                                   config.min_level,
                                                   config.max_level,
                                                   config.fpn_weight_method)
    self.fnodes = []
    for i, fnode_cfg in enumerate(self.fpn_config.nodes):
      logging.info('fnode %d : %s', i, fnode_cfg)
      fnode = FNode(
          feat_sizes[fnode_cfg['feat_level']]['height'],
          feat_sizes[fnode_cfg['feat_level']]['width'],
          fnode_cfg['inputs_offsets'],
          config.fpn_num_filters,
          config.apply_bn_for_resampling,
          config.is_training_bn,
          config.conv_after_downsample,
          config.conv_bn_act_pattern,
          config.separable_conv,
          config.act_type,
          strategy=config.strategy,
          weight_method=self.fpn_config.weight_method,
          data_format=config.data_format,
          name='fnode%d' % i)
      self.fnodes.append(fnode)

  def call(self, feats, training):
    for fnode in self.fnodes:
      feats = fnode(feats, training)
    return feats


class EfficientDetNet(tf.keras.Model):
  """EfficientDet keras network without pre/post-processing."""

  def __init__(self, model_name=None, config=None, name=''):
    """Initialize model."""
    super().__init__(name=name)

    config = config or hparams_config.get_efficientdet_config(model_name)
    self.config = config

    # Backbone.
    backbone_name = config.backbone_name
    is_training_bn = config.is_training_bn
    if 'efficientnet' in backbone_name:
      override_params = {
          'batch_norm':
              utils.batch_norm_class(is_training_bn, config.strategy),
          'relu_fn':
              functools.partial(utils.activation_fn, act_type=config.act_type),
      }
      if 'b0' in backbone_name:
        override_params['survival_prob'] = 0.0
      if config.backbone_config is not None:
        override_params['blocks_args'] = (
            efficientnet_builder.BlockDecoder().encode(
                config.backbone_config.blocks))
      override_params['data_format'] = config.data_format
      self.backbone = backbone_factory.get_model(
          backbone_name, override_params=override_params)

    # Feature network.
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    self.resample_layers = []  # additional resampling layers.
    for level in range(6, config.max_level + 1):
      # Adds a coarser level by downsampling the last feature map.
      self.resample_layers.append(
          ResampleFeatureMap(
              target_height=feat_sizes[level]['height'],
              target_width=feat_sizes[level]['width'],
              target_num_channels=config.fpn_num_filters,
              apply_bn=config.apply_bn_for_resampling,
              is_training_bn=config.is_training_bn,
              conv_after_downsample=config.conv_after_downsample,
              strategy=config.strategy,
              data_format=config.data_format,
              name='resample_p%d' % level,
          ))
    self.fpn_cells = FPNCells(config)

    # class/box output prediction network.
    num_anchors = len(config.aspect_ratios) * config.num_scales
    num_filters = config.fpn_num_filters
    for head in config.heads:
      if head == 'object_detection':
        self.class_net = ClassNet(
            num_classes=config.num_classes,
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            is_training_bn=config.is_training_bn,
            act_type=config.act_type,
            repeats=config.box_class_repeats,
            separable_conv=config.separable_conv,
            survival_prob=config.survival_prob,
            strategy=config.strategy,
            data_format=config.data_format)

        self.box_net = BoxNet(
            num_anchors=num_anchors,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            is_training_bn=config.is_training_bn,
            act_type=config.act_type,
            repeats=config.box_class_repeats,
            separable_conv=config.separable_conv,
            survival_prob=config.survival_prob,
            strategy=config.strategy,
            data_format=config.data_format)

      if head == 'segmentation':
        self.seg_head = SegmentationHead(
            num_classes=config.seg_num_classes,
            num_filters=num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            is_training_bn=config.is_training_bn,
            act_type=config.act_type,
            strategy=config.strategy,
            data_format=config.data_format)

  def _init_set_name(self, name, zero_based=True):
    """A hack to allow empty model name for legacy checkpoint compitability."""
    if name == '':  # pylint: disable=g-explicit-bool-comparison
      self._name = name
    else:
      self._name = super().__init__(name, zero_based)

  def call(self, inputs, training):
    config = self.config
    # call backbone network.
    self.backbone(inputs, training=training, features_only=True)
    all_feats = [
        inputs,
        self.backbone.endpoints['reduction_1'],
        self.backbone.endpoints['reduction_2'],
        self.backbone.endpoints['reduction_3'],
        self.backbone.endpoints['reduction_4'],
        self.backbone.endpoints['reduction_5'],
    ]
    feats = all_feats[config.min_level:config.max_level + 1]

    # Build additional input features that are not from backbone.
    for resample_layer in self.resample_layers:
      feats.append(resample_layer(feats[-1], training))

    # call feature network.
    feats = self.fpn_cells(feats, training)

    # call class/box/seg output network.
    outputs = []
    if 'object_detection' in config.heads:
      class_outputs = self.class_net(feats, training)
      box_outputs = self.box_net(feats, training)
      outputs.extend([class_outputs, box_outputs])
    if 'segmentation' in config.heads:
      seg_outputs = self.seg_head(feats, training)
      outputs.append(seg_outputs)
    return tuple(outputs)


class EfficientDetModel(EfficientDetNet):
  """EfficientDet full keras model with pre and post processing."""

  def _preprocessing(self, raw_images, image_size, mode=None):
    """Preprocess images before feeding to the network."""
    if not mode:
      return raw_images, None

    image_size = utils.parse_image_size(image_size)
    if mode != 'infer':
      # We only support inference for now.
      raise ValueError('preprocessing must be infer or empty')

    scales, images = [], []
    if isinstance(raw_images, tf.Tensor):
      batch_size = raw_images.shape[0]
    else:
      batch_size = len(raw_images)
    for i in range(batch_size):
      input_processor = dataloader.DetectionInputProcessor(
          raw_images[i], image_size)
      input_processor.normalize_image()
      input_processor.set_scale_factors_to_output_size()
      images.append(input_processor.resize_and_crop_image())
      scales.append(input_processor.image_scale_to_original)
    return tf.stack(images), tf.stack(scales)

  def _postprocess(self, cls_outputs, box_outputs, scales, mode='global'):
    if not mode:
      return cls_outputs, box_outputs

    # TODO(tanmingxing): remove this cast once FP16 works postprocessing.
    cls_outputs = [tf.cast(i, tf.float32) for i in cls_outputs]
    box_outputs = [tf.cast(i, tf.float32) for i in box_outputs]

    if mode == 'global':
      return postprocess.postprocess_global(self.config.as_dict(), cls_outputs,
                                            box_outputs, scales)
    if mode == 'per_class':
      return postprocess.postprocess_per_class(self.config.as_dict(),
                                               cls_outputs, box_outputs, scales)
    raise ValueError('Unsupported postprocess mode {}'.format(mode))

  def call(self, inputs, training, pre_mode='infer', post_mode='global'):
    """Call this model.

    Args:
      inputs: a tensor with common shape [batch, height, width, channels].
      training: If true, it is training mode. Otherwise, eval mode.
      pre_mode: preprocessing mode, must be {None, 'infer'}.
      post_mode: postprrocessing mode, must be {None, 'global', 'per_class'}.

    Returns:
      the output tensor list.
    """
    config = self.config

    # preprocess.
    inputs, scales = self._preprocessing(inputs, config.image_size, pre_mode)
    # network.
    outputs = super().call(inputs, training)

    if 'object_detection' in config.heads and post_mode:
      # postprocess for detection
      det_outputs = self._postprocess(outputs[0], outputs[1], scales, post_mode)
      outputs = det_outputs + outputs[2:]

    return outputs
