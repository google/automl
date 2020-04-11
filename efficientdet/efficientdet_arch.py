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
"""EfficientDet model definition.

[1] Mingxing Tan, Ruoming Pang, Quoc Le.
    EfficientDet: Scalable and Efficient Object Detection.
    CVPR 2020, https://arxiv.org/abs/1911.09070
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import hparams_config
import utils
from backbone import backbone_factory
from backbone import efficientnet_builder


################################################################################
def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation."""
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])


def resize_bilinear(images, size, output_type):
  """Returns resized images as output_type."""
  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, output_type)


def remove_variables(variables, resnet_depth=50):
  """Removes low-level variables from the input.

  Removing low-level parameters (e.g., initial convolution layer) from training
  usually leads to higher training speed and slightly better testing accuracy.
  The intuition is that the low-level architecture (e.g., ResNet-50) is able to
  capture low-level features such as edges; therefore, it does not need to be
  fine-tuned for the detection task.

  Args:
    variables: all the variables in training
    resnet_depth: the depth of ResNet model

  Returns:
    var_list: a list containing variables for training

  """
  var_list = [v for v in variables
              if v.name.find('resnet%s/conv2d/' % resnet_depth) == -1]
  return var_list


def resample_feature_map(feat, name, target_width, target_num_channels,
                         apply_bn=False, is_training=None,
                         conv_after_downsample=False,
                         use_native_resize_op=False, pooling_type=None,
                         use_tpu=False):
  """Resample input feature map to have target number of channels and width."""

  _, width, _, num_channels = feat.get_shape().as_list()
  if width is None or num_channels is None:
    raise ValueError('shape[1] or shape[3] of feat is None (shape:{}).'.format(
        feat.shape))
  if apply_bn and is_training is None:
    raise ValueError('If BN is applied, need to provide is_training')

  def _maybe_apply_1x1(feat):
    """Apply 1x1 conv to change layer width if necessary."""
    if num_channels != target_num_channels:
      feat = tf.layers.conv2d(
          feat, filters=target_num_channels, kernel_size=(1, 1), padding='same')
      if apply_bn:
        feat = utils.batch_norm_relu(
            feat,
            is_training_bn=is_training,
            relu=False,
            data_format='channels_last',
            use_tpu=use_tpu,
            name='bn')
    return feat

  with tf.variable_scope('resample_{}'.format(name)):
    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if width > target_width:
      if not conv_after_downsample:
        feat = _maybe_apply_1x1(feat)
      stride_size = int((width - 1) // target_width + 1)
      if pooling_type == 'max' or pooling_type is None:
        # Use max pooling in default.
        feat = tf.layers.max_pooling2d(
            inputs=feat,
            pool_size=stride_size + 1,
            strides=[stride_size, stride_size],
            padding='SAME',
            data_format='channels_last')
      elif pooling_type == 'avg':
        feat = tf.layers.average_pooling2d(
            inputs=feat,
            pool_size=stride_size + 1,
            strides=[stride_size, stride_size],
            padding='SAME',
            data_format='channels_last')
      else:
        raise ValueError('Unknown pooling type: {}'.format(pooling_type))
      if conv_after_downsample:
        feat = _maybe_apply_1x1(feat)
    else:
      feat = _maybe_apply_1x1(feat)
      if width < target_width:
        scale = target_width // width
        if use_native_resize_op or target_width % width != 0:
          feat = tf.cast(
              tf.image.resize_nearest_neighbor(
                  tf.cast(feat, tf.float32), [target_width, target_width]),
              dtype=feat.dtype)
        else:
          feat = nearest_upsampling(feat, scale=scale)

  return feat


def _verify_feats_size(feats, input_size, feat_sizes, min_level, max_level):
  """Verify the feature map sizes."""
  expected_output_width = feat_sizes[min_level:max_level + 1]
  for cnt, width in enumerate(expected_output_width):
    if feats[cnt].shape[1] != width:
      raise ValueError('feats[{}] has shape {} but its width should be {}.'
                       '(input_size: {}, min_level: {}, max_level: {}.)'.format(
                           cnt, feats[cnt].shape, width, input_size, min_level,
                           max_level))


###############################################################################
def class_net(images,
              level,
              num_classes,
              num_anchors,
              num_filters,
              is_training,
              separable_conv=True,
              repeats=4,
              survival_prob=None,
              use_tpu=False):
  """Class prediction network."""
  if separable_conv:
    conv_op = functools.partial(
        tf.layers.separable_conv2d, depth_multiplier=1,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling())
  else:
    conv_op = functools.partial(
        tf.layers.conv2d,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

  for i in range(repeats):
    orig_images = images
    images = conv_op(
        images,
        num_filters,
        kernel_size=3,
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        padding='same',
        name='class-%d' % i)
    images = utils.batch_norm_relu(
        images,
        is_training,
        relu=True,
        init_zero=False,
        use_tpu=use_tpu,
        name='class-%d-bn-%d' % (i, level))

    if i > 0 and survival_prob:
      images = utils.drop_connect(images, is_training, survival_prob)
      images = images + orig_images

  classes = conv_op(
      images,
      num_classes * num_anchors,
      kernel_size=3,
      bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
      padding='same',
      name='class-predict')
  return classes


def box_net(images, level, num_anchors, num_filters, is_training,
            repeats=4, separable_conv=True, survival_prob=None, use_tpu=False):
  """Box regression network."""
  if separable_conv:
    conv_op = functools.partial(
        tf.layers.separable_conv2d, depth_multiplier=1,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling())
  else:
    conv_op = functools.partial(
        tf.layers.conv2d,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

  for i in range(repeats):
    orig_images = images
    images = conv_op(
        images,
        num_filters,
        kernel_size=3,
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        padding='same',
        name='box-%d' % i)
    images = utils.batch_norm_relu(
        images,
        is_training,
        relu=True,
        init_zero=False,
        use_tpu=use_tpu,
        name='box-%d-bn-%d' % (i, level))

    if i > 0 and survival_prob:
      images = utils.drop_connect(images, is_training, survival_prob)
      images = images + orig_images

  boxes = conv_op(
      images,
      4 * num_anchors,
      kernel_size=3,
      bias_initializer=tf.zeros_initializer(),
      padding='same',
      name='box-predict')

  return boxes


def build_class_and_box_outputs(feats, config):
  """Builds box net and class net.

  Args:
   feats: input tensor.
   config: a dict-like config, including all parameters.

  Returns:
   A tuple (class_outputs, box_outputs) for class/box predictions.
  """

  class_outputs = {}
  box_outputs = {}
  num_anchors = len(config.aspect_ratios) * config.num_scales
  cls_fsize = config.fpn_num_filters
  with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
    for level in range(config.min_level,
                       config.max_level + 1):
      class_outputs[level] = class_net(
          images=feats[level],
          level=level,
          num_classes=config.num_classes,
          num_anchors=num_anchors,
          num_filters=cls_fsize,
          is_training=config.is_training_bn,
          repeats=config.box_class_repeats,
          separable_conv=config.separable_conv,
          survival_prob=config.survival_prob,
          use_tpu=config.use_tpu
          )

  box_fsize = config.fpn_num_filters
  with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
    for level in range(config.min_level,
                       config.max_level + 1):
      box_outputs[level] = box_net(
          images=feats[level],
          level=level,
          num_anchors=num_anchors,
          num_filters=box_fsize,
          is_training=config.is_training_bn,
          repeats=config.box_class_repeats,
          separable_conv=config.separable_conv,
          survival_prob=config.survival_prob,
          use_tpu=config.use_tpu)

  return class_outputs, box_outputs


def build_backbone(features, config):
  """Builds backbone model.

  Args:
   features: input tensor.
   config: config for backbone, such as is_training_bn and backbone name.

  Returns:
    A dict from levels to the feature maps from the output of the backbone model
    with strides of 8, 16 and 32.

  Raises:
    ValueError: if backbone_name is not supported.
  """
  backbone_name = config.backbone_name
  is_training_bn = config.is_training_bn
  if 'efficientnet' in backbone_name:
    override_params = {
        'relu_fn': utils.backbone_relu_fn,
        'batch_norm': utils.batch_norm_class(is_training_bn, config.use_tpu),
    }
    if 'b0' in backbone_name:
      override_params['survival_prob'] = 0.0
    if config.backbone_config is not None:
      override_params['blocks_args'] = (
          efficientnet_builder.BlockDecoder().encode(
              config.backbone_config.blocks))
    model_builder = backbone_factory.get_model_builder(backbone_name)
    _, endpoints = model_builder.build_model_base(
        features,
        backbone_name,
        training=is_training_bn,
        override_params=override_params)
    u2 = endpoints['reduction_2']
    u3 = endpoints['reduction_3']
    u4 = endpoints['reduction_4']
    u5 = endpoints['reduction_5']
  else:
    raise ValueError(
        'backbone model {} is not supported.'.format(backbone_name))
  return {2: u2, 3: u3, 4: u4, 5: u5}


def build_feature_network(features, config):
  """Build FPN input features.

  Args:
   features: input tensor.
   config: a dict-like config, including all parameters.

  Returns:
    A dict from levels to the feature maps processed after feature network.
  """
  feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
  feats = []
  if config.min_level not in features.keys():
    raise ValueError('features.keys ({}) should include min_level ({})'.format(
        features.keys(), config.min_level))

  # Build additional input features that are not from backbone.
  for level in range(config.min_level, config.max_level + 1):
    if level in features.keys():
      feats.append(features[level])
    else:
      # Adds a coarser level by downsampling the last feature map.
      feats.append(
          resample_feature_map(
              feats[-1],
              name='p%d' % level,
              target_width=(feats[-1].shape[1] - 1) // 2 + 1,
              target_num_channels=config.fpn_num_filters,
              apply_bn=config.apply_bn_for_resampling,
              is_training=config.is_training_bn,
              conv_after_downsample=config.conv_after_downsample,
              use_native_resize_op=config.use_native_resize_op,
              pooling_type=config.pooling_type,
              use_tpu=config.use_tpu
          ))

  _verify_feats_size(
      feats,
      input_size=config.image_size,
      feat_sizes=feat_sizes,
      min_level=config.min_level,
      max_level=config.max_level)

  with tf.variable_scope('fpn_cells'):
    for rep in range(config.fpn_cell_repeats):
      with tf.variable_scope('cell_{}'.format(rep)):
        logging.info('building cell %d', rep)
        new_feats = build_bifpn_layer(
            feats=feats,
            feat_sizes=feat_sizes,
            fpn_name=config.fpn_name,
            fpn_config=config.fpn_config,
            fpn_num_filters=config.fpn_num_filters,
            min_level=config.min_level,
            max_level=config.max_level,
            separable_conv=config.separable_conv,
            is_training=config.is_training_bn,
            apply_bn_for_resampling=config.apply_bn_for_resampling,
            conv_after_downsample=config.conv_after_downsample,
            use_native_resize_op=config.use_native_resize_op,
            conv_bn_relu_pattern=config.conv_bn_relu_pattern,
            pooling_type=config.pooling_type,
            use_tpu=config.use_tpu)

        feats = [
            new_feats[level]
            for level in range(
                config.min_level, config.max_level + 1)
        ]

        _verify_feats_size(
            feats,
            input_size=config.image_size,
            feat_sizes=feat_sizes,
            min_level=config.min_level,
            max_level=config.max_level)

  return new_feats


def bifpn_sum_config():
  """BiFPN config with sum."""
  p = hparams_config.Config()
  p.nodes = [
      {'width_index': 6, 'inputs_offsets': [3, 4]},
      {'width_index': 5, 'inputs_offsets': [2, 5]},
      {'width_index': 4, 'inputs_offsets': [1, 6]},
      {'width_index': 3, 'inputs_offsets': [0, 7]},
      {'width_index': 4, 'inputs_offsets': [1, 7, 8]},
      {'width_index': 5, 'inputs_offsets': [2, 6, 9]},
      {'width_index': 6, 'inputs_offsets': [3, 5, 10]},
      {'width_index': 7, 'inputs_offsets': [4, 11]},
  ]
  p.weight_method = 'sum'
  return p


def bifpn_fa_config():
  """BiFPN config with fast weighted sum."""
  p = bifpn_sum_config()
  p.weight_method = 'fastattn'
  return p


def get_fpn_config(fpn_name):
  if not fpn_name:
    fpn_name = 'bifpn_fa'
  name_to_config = {
      'bifpn_sum': bifpn_sum_config(),
      'bifpn_fa': bifpn_fa_config(),
  }
  return name_to_config[fpn_name]


def build_bifpn_layer(
    feats, feat_sizes, fpn_name, fpn_config, is_training,
    fpn_num_filters, min_level, max_level, separable_conv,
    apply_bn_for_resampling, conv_after_downsample,
    use_native_resize_op, conv_bn_relu_pattern, pooling_type, use_tpu=False):
  """Builds a feature pyramid given previous feature pyramid and config."""
  config = fpn_config or get_fpn_config(fpn_name)

  num_output_connections = [0 for _ in feats]
  for i, fnode in enumerate(config.nodes):
    with tf.variable_scope('fnode{}'.format(i)):
      logging.info('fnode %d : %s', i, fnode)
      new_node_width = feat_sizes[fnode['width_index']]
      nodes = []
      for idx, input_offset in enumerate(fnode['inputs_offsets']):
        input_node = feats[input_offset]
        num_output_connections[input_offset] += 1
        input_node = resample_feature_map(
            input_node, '{}_{}_{}'.format(idx, input_offset, len(feats)),
            new_node_width, fpn_num_filters,
            apply_bn_for_resampling, is_training,
            conv_after_downsample,
            use_native_resize_op,
            pooling_type)
        nodes.append(input_node)

      # Combine all nodes.
      dtype = nodes[0].dtype
      if config.weight_method == 'attn':
        edge_weights = [tf.cast(tf.Variable(1.0, name='WSM'), dtype=dtype)
                        for _ in range(len(fnode['inputs_offsets']))]
        normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
        nodes = tf.stack(nodes, axis=-1)
        new_node = tf.reduce_sum(tf.multiply(nodes, normalized_weights), -1)
      elif config.weight_method == 'fastattn':
        edge_weights = [
            tf.nn.relu(tf.cast(tf.Variable(1.0, name='WSM'), dtype=dtype))
            for _ in range(len(fnode['inputs_offsets']))
        ]
        weights_sum = tf.add_n(edge_weights)
        nodes = [nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
                 for i in range(len(nodes))]
        new_node = tf.add_n(nodes)
      elif config.weight_method == 'sum':
        new_node = tf.add_n(nodes)
      else:
        raise ValueError(
            'unknown weight_method {}'.format(config.weight_method))

      with tf.variable_scope('op_after_combine{}'.format(len(feats))):
        if not conv_bn_relu_pattern:
          new_node = utils.relu_fn(new_node)

        if separable_conv:
          conv_op = functools.partial(
              tf.layers.separable_conv2d, depth_multiplier=1)
        else:
          conv_op = tf.layers.conv2d

        new_node = conv_op(
            new_node,
            filters=fpn_num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True if not conv_bn_relu_pattern else False,
            name='conv')

        new_node = utils.batch_norm_relu(
            new_node,
            is_training_bn=is_training,
            relu=False if not conv_bn_relu_pattern else True,
            data_format='channels_last',
            use_tpu=use_tpu,
            name='bn')

      feats.append(new_node)
      num_output_connections.append(0)

  output_feats = {}
  for l in range(min_level, max_level + 1):
    for i, fnode in enumerate(reversed(config.nodes)):
      if fnode['width_index'] == l:
        output_feats[l] = feats[-1 - i]
        break
  return output_feats


def efficientdet(features, model_name=None, config=None, **kwargs):
  """Build EfficientDet model."""
  if not config and not model_name:
    raise ValueError('please specify either model name or config')

  if not config:
    config = hparams_config.get_efficientdet_config(model_name)

  if kwargs:
    config.override(kwargs)

  logging.info(config)

  # build backbone features.
  features = build_backbone(features, config)
  logging.info('backbone params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  # build feature network.
  fpn_feats = build_feature_network(features, config)
  logging.info('backbone+fpn params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  # build class and box predictions.
  class_outputs, box_outputs = build_class_and_box_outputs(fpn_feats, config)
  logging.info('backbone+fpn+box params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  return class_outputs, box_outputs
