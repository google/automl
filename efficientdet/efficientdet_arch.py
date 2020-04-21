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
import itertools
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import hparams_config
import utils
from backbone import backbone_factory
from backbone import efficientnet_builder


################################################################################
def nearest_upsampling(data, height_scale, width_scale, data_format):
  """Nearest neighbor upsampling implementation."""
  with tf.name_scope('nearest_upsampling'):
    # Use reshape to quickly upsample the input. The nearest pixel is selected
    # implicitly via broadcasting.
    if data_format == 'channels_first':
      # Possibly faster for certain GPUs only.
      bs, c, h, w = data.get_shape().as_list()
      bs = -1 if bs is None else bs
      data = tf.reshape(data, [bs, c, h, 1, w, 1]) * tf.ones(
          [1, 1, 1, height_scale, 1, width_scale], dtype=data.dtype)
      return tf.reshape(data, [bs, c, h * height_scale, w * width_scale])

    # Normal format for CPU/TPU/GPU.
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, height_scale, 1, width_scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * height_scale, w * width_scale, c])


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


def resample_feature_map(feat,
                         name,
                         target_height,
                         target_width,
                         target_num_channels,
                         apply_bn=False,
                         is_training=None,
                         conv_after_downsample=False,
                         use_native_resize_op=False,
                         pooling_type=None,
                         use_tpu=False,
                         data_format='channels_last'):
  """Resample input feature map to have target number of channels and size."""
  if data_format == 'channels_first':
    _, num_channels, height, width = feat.get_shape().as_list()
  else:
    _, height, width, num_channels = feat.get_shape().as_list()

  if height is None or width is None or num_channels is None:
    raise ValueError(
        'shape[1] or shape[2] or shape[3] of feat is None (shape:{}).'.format(
            feat.shape))
  if apply_bn and is_training is None:
    raise ValueError('If BN is applied, need to provide is_training')

  def _maybe_apply_1x1(feat):
    """Apply 1x1 conv to change layer width if necessary."""
    if num_channels != target_num_channels:
      feat = tf.layers.conv2d(
          feat,
          filters=target_num_channels,
          kernel_size=(1, 1),
          padding='same',
          data_format=data_format)
      if apply_bn:
        feat = utils.batch_norm_act(
            feat,
            is_training_bn=is_training,
            act_type=None,
            data_format=data_format,
            use_tpu=use_tpu,
            name='bn')
    return feat

  with tf.variable_scope('resample_{}'.format(name)):
    # If conv_after_downsample is True, when downsampling, apply 1x1 after
    # downsampling for efficiency.
    if height > target_height and width > target_width:
      if not conv_after_downsample:
        feat = _maybe_apply_1x1(feat)
      height_stride_size = int((height - 1) // target_height + 1)
      width_stride_size = int((width - 1) // target_width + 1)
      if pooling_type == 'max' or pooling_type is None:
        # Use max pooling in default.
        feat = tf.layers.max_pooling2d(
            inputs=feat,
            pool_size=[height_stride_size + 1, width_stride_size + 1],
            strides=[height_stride_size, width_stride_size],
            padding='SAME',
            data_format=data_format)
      elif pooling_type == 'avg':
        feat = tf.layers.average_pooling2d(
            inputs=feat,
            pool_size=[height_stride_size + 1, width_stride_size + 1],
            strides=[height_stride_size, width_stride_size],
            padding='SAME',
            data_format=data_format)
      else:
        raise ValueError('Unknown pooling type: {}'.format(pooling_type))
      if conv_after_downsample:
        feat = _maybe_apply_1x1(feat)
    elif height <= target_height and width <= target_width:
      feat = _maybe_apply_1x1(feat)
      if height < target_height or width < target_width:
        height_scale = target_height // height
        width_scale = target_width // width
        if (use_native_resize_op or target_height % height != 0 or
            target_width % width != 0):
          if data_format == 'channels_first':
            feat = tf.transpose(feat, [0, 2, 3, 1])
          feat = tf.image.resize_nearest_neighbor(feat,
                                                  [target_height, target_width])
          if data_format == 'channels_first':
            feat = tf.transpose(feat, [0, 3, 1, 2])
        else:
          feat = nearest_upsampling(
              feat,
              height_scale=height_scale,
              width_scale=width_scale,
              data_format=data_format)
    else:
      raise ValueError(
          'Incompatible target feature map size: target_height: {},'
          'target_width: {}'.format(target_height, target_width))

  return feat


def _verify_feats_size(feats,
                       feat_sizes,
                       min_level,
                       max_level,
                       data_format='channels_last'):
  """Verify the feature map sizes."""
  expected_output_size = feat_sizes[min_level:max_level + 1]
  for cnt, size in enumerate(expected_output_size):
    h_id, w_id = (2, 3) if data_format == 'channels_first' else (1, 2)
    if feats[cnt].shape[h_id] != size['height']:
      raise ValueError(
          'feats[{}] has shape {} but its height should be {}.'
          '(input_height: {}, min_level: {}, max_level: {}.)'.format(
              cnt, feats[cnt].shape, size['height'], feat_sizes[0]['height'],
              min_level, max_level))
    if feats[cnt].shape[w_id] != size['width']:
      raise ValueError(
          'feats[{}] has shape {} but its width should be {}.'
          '(input_width: {}, min_level: {}, max_level: {}.)'.format(
              cnt, feats[cnt].shape, size['width'], feat_sizes[0]['width'],
              min_level, max_level))


###############################################################################
def class_net(images,
              level,
              num_classes,
              num_anchors,
              num_filters,
              is_training,
              act_type,
              separable_conv=True,
              repeats=4,
              survival_prob=None,
              use_tpu=False,
              data_format='channels_last'):
  """Class prediction network."""
  if separable_conv:
    conv_op = functools.partial(
        tf.layers.separable_conv2d, depth_multiplier=1,
        data_format=data_format,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling())
  else:
    conv_op = functools.partial(
        tf.layers.conv2d,
        data_format=data_format,
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
    images = utils.batch_norm_act(
        images,
        is_training,
        act_type=act_type,
        init_zero=False,
        use_tpu=use_tpu,
        data_format=data_format,
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


def box_net(images,
            level,
            num_anchors,
            num_filters,
            is_training,
            act_type,
            repeats=4,
            separable_conv=True,
            survival_prob=None,
            use_tpu=False,
            data_format='channels_last'):
  """Box regression network."""
  if separable_conv:
    conv_op = functools.partial(
        tf.layers.separable_conv2d, depth_multiplier=1,
        data_format=data_format,
        pointwise_initializer=tf.initializers.variance_scaling(),
        depthwise_initializer=tf.initializers.variance_scaling())
  else:
    conv_op = functools.partial(
        tf.layers.conv2d,
        data_format=data_format,
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
    images = utils.batch_norm_act(
        images,
        is_training,
        act_type=act_type,
        init_zero=False,
        use_tpu=use_tpu,
        data_format=data_format,
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
          act_type=config.act_type,
          repeats=config.box_class_repeats,
          separable_conv=config.separable_conv,
          survival_prob=config.survival_prob,
          use_tpu=config.use_tpu,
          data_format=config.data_format
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
          act_type=config.act_type,
          repeats=config.box_class_repeats,
          separable_conv=config.separable_conv,
          survival_prob=config.survival_prob,
          use_tpu=config.use_tpu,
          data_format=config.data_format)

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
        'batch_norm': utils.batch_norm_class(is_training_bn, config.use_tpu),
    }
    if 'b0' in backbone_name:
      override_params['survival_prob'] = 0.0
    if config.backbone_config is not None:
      override_params['blocks_args'] = (
          efficientnet_builder.BlockDecoder().encode(
              config.backbone_config.blocks))
    override_params['data_format'] = config.data_format
    model_builder = backbone_factory.get_model_builder(backbone_name)
    _, endpoints = model_builder.build_model_base(
        features,
        backbone_name,
        training=is_training_bn,
        override_params=override_params)
    u1 = endpoints['reduction_1']
    u2 = endpoints['reduction_2']
    u3 = endpoints['reduction_3']
    u4 = endpoints['reduction_4']
    u5 = endpoints['reduction_5']
  else:
    raise ValueError(
        'backbone model {} is not supported.'.format(backbone_name))
  return {0: features, 1: u1, 2: u2, 3: u3, 4: u4, 5: u5}


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
      h_id, w_id = (2, 3) if config.data_format == 'channels_first' else (1, 2)
      # Adds a coarser level by downsampling the last feature map.
      feats.append(
          resample_feature_map(
              feats[-1],
              name='p%d' % level,
              target_height=(feats[-1].shape[h_id] - 1) // 2 + 1,
              target_width=(feats[-1].shape[w_id] - 1) // 2 + 1,
              target_num_channels=config.fpn_num_filters,
              apply_bn=config.apply_bn_for_resampling,
              is_training=config.is_training_bn,
              conv_after_downsample=config.conv_after_downsample,
              use_native_resize_op=config.use_native_resize_op,
              pooling_type=config.pooling_type,
              use_tpu=config.use_tpu,
              data_format=config.data_format
          ))

  _verify_feats_size(
      feats,
      feat_sizes=feat_sizes,
      min_level=config.min_level,
      max_level=config.max_level,
      data_format=config.data_format)

  with tf.variable_scope('fpn_cells'):
    for rep in range(config.fpn_cell_repeats):
      with tf.variable_scope('cell_{}'.format(rep)):
        logging.info('building cell %d', rep)
        new_feats = build_bifpn_layer(feats, feat_sizes, config)

        feats = [
            new_feats[level]
            for level in range(
                config.min_level, config.max_level + 1)
        ]

        _verify_feats_size(
            feats,
            feat_sizes=feat_sizes,
            min_level=config.min_level,
            max_level=config.max_level,
            data_format=config.data_format)

  return new_feats


def bifpn_sum_config():
  """BiFPN config with sum."""
  p = hparams_config.Config()
  p.nodes = [
      {'feat_level': 6, 'inputs_offsets': [3, 4]},
      {'feat_level': 5, 'inputs_offsets': [2, 5]},
      {'feat_level': 4, 'inputs_offsets': [1, 6]},
      {'feat_level': 3, 'inputs_offsets': [0, 7]},
      {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},
      {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},
      {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},
      {'feat_level': 7, 'inputs_offsets': [4, 11]},
  ]
  p.weight_method = 'sum'
  return p


def bifpn_fa_config():
  """BiFPN config with fast weighted sum."""
  p = bifpn_sum_config()
  p.weight_method = 'fastattn'
  return p


def bifpn_dynamic_config(min_level, max_level, weight_method):
  """A dynamic bifpn config that can adapt to different min/max levels."""
  p = hparams_config.Config()
  p.weight_method = weight_method or 'fastattn'

  # Node id starts from the input features and monotonically increase whenever
  # a new node is added. Here is an example for level P3 - P7:
  #     P7 (4)              P7" (12)
  #     P6 (3)    P6' (5)   P6" (11)
  #     P5 (2)    P5' (6)   P5" (10)
  #     P4 (1)    P4' (7)   P4" (9)
  #     P3 (0)              P3" (8)
  # So output would be like:
  # [
  #   {'feat_level': 6, 'inputs_offsets': [3, 4]},  # for P6'
  #   {'feat_level': 5, 'inputs_offsets': [2, 5]},  # for P5'
  #   {'feat_level': 4, 'inputs_offsets': [1, 6]},  # for P4'
  #   {'feat_level': 3, 'inputs_offsets': [0, 7]},  # for P3"
  #   {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},  # for P4"
  #   {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},  # for P5"
  #   {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},  # for P6"
  #   {'feat_level': 7, 'inputs_offsets': [4, 11]},  # for P7"
  # ]
  num_levels = max_level - min_level + 1
  node_ids = {min_level + i: [i] for i in range(num_levels)}

  level_last_id = lambda level: node_ids[level][-1]
  level_all_ids = lambda level: node_ids[level]
  id_cnt = itertools.count(num_levels)

  p.nodes = []
  for i in range(max_level - 1, min_level - 1, -1):
    # top-down path.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': [level_last_id(i), level_last_id(i + 1)]
    })
    node_ids[i].append(next(id_cnt))

  for i in range(min_level + 1, max_level + 1):
    # bottom-up path.
    p.nodes.append({
        'feat_level': i,
        'inputs_offsets': level_all_ids(i) + [level_last_id(i - 1)]
    })
    node_ids[i].append(next(id_cnt))

  return p


def get_fpn_config(fpn_name, min_level, max_level, weight_method):
  """Get fpn related configuration."""
  if not fpn_name:
    fpn_name = 'bifpn_fa'
  name_to_config = {
      'bifpn_sum': bifpn_sum_config(),
      'bifpn_fa': bifpn_fa_config(),
      'bifpn_dyn': bifpn_dynamic_config(min_level, max_level, weight_method)
  }
  return name_to_config[fpn_name]


def build_bifpn_layer(feats, feat_sizes, config):
  """Builds a feature pyramid given previous feature pyramid and config."""
  p = config  # use p to denote the network config.
  if p.fpn_config:
    fpn_config = p.fpn_config
  else:
    fpn_config = get_fpn_config(p.fpn_name, p.min_level, p.max_level,
                                p.fpn_weight_method)

  num_output_connections = [0 for _ in feats]
  for i, fnode in enumerate(fpn_config.nodes):
    with tf.variable_scope('fnode{}'.format(i)):
      logging.info('fnode %d : %s', i, fnode)
      new_node_height = feat_sizes[fnode['feat_level']]['height']
      new_node_width = feat_sizes[fnode['feat_level']]['width']
      nodes = []
      for idx, input_offset in enumerate(fnode['inputs_offsets']):
        input_node = feats[input_offset]
        num_output_connections[input_offset] += 1
        input_node = resample_feature_map(
            input_node, '{}_{}_{}'.format(idx, input_offset, len(feats)),
            new_node_height, new_node_width, p.fpn_num_filters,
            p.apply_bn_for_resampling, p.is_training_bn,
            p.conv_after_downsample,
            p.use_native_resize_op,
            p.pooling_type,
            data_format=config.data_format)
        nodes.append(input_node)

      # Combine all nodes.
      dtype = nodes[0].dtype
      if fpn_config.weight_method == 'attn':
        edge_weights = [tf.cast(tf.Variable(1.0, name='WSM'), dtype=dtype)
                        for _ in range(len(fnode['inputs_offsets']))]
        normalized_weights = tf.nn.softmax(tf.stack(edge_weights))
        nodes = tf.stack(nodes, axis=-1)
        new_node = tf.reduce_sum(tf.multiply(nodes, normalized_weights), -1)
      elif fpn_config.weight_method == 'fastattn':
        edge_weights = [
            tf.nn.relu(tf.cast(tf.Variable(1.0, name='WSM'), dtype=dtype))
            for _ in range(len(fnode['inputs_offsets']))
        ]
        weights_sum = tf.add_n(edge_weights)
        nodes = [nodes[i] * edge_weights[i] / (weights_sum + 0.0001)
                 for i in range(len(nodes))]
        new_node = tf.add_n(nodes)
      elif fpn_config.weight_method == 'sum':
        new_node = tf.add_n(nodes)
      else:
        raise ValueError(
            'unknown weight_method {}'.format(fpn_config.weight_method))

      with tf.variable_scope('op_after_combine{}'.format(len(feats))):
        if not p.conv_bn_act_pattern:
          new_node = utils.activation_fn(new_node, p.act_type)

        if p.separable_conv:
          conv_op = functools.partial(
              tf.layers.separable_conv2d, depth_multiplier=1)
        else:
          conv_op = tf.layers.conv2d

        new_node = conv_op(
            new_node,
            filters=p.fpn_num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True if not p.conv_bn_act_pattern else False,
            data_format=config.data_format,
            name='conv')

        new_node = utils.batch_norm_act(
            new_node,
            is_training_bn=p.is_training_bn,
            act_type=None if not p.conv_bn_act_pattern else p.act_type,
            data_format=config.data_format,
            use_tpu=p.use_tpu,
            name='bn')

      feats.append(new_node)
      num_output_connections.append(0)

  output_feats = {}
  for l in range(p.min_level, p.max_level + 1):
    for i, fnode in enumerate(reversed(fpn_config.nodes)):
      if fnode['feat_level'] == l:
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
