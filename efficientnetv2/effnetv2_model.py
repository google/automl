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
"""EfficientNet V1 and V2 model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946

[2] Mingxing Tan, Quoc V. Le
  EfficientNetV2: Smaller Models and Faster Training.
  https://arxiv.org/abs/2104.00298
"""
import copy
import itertools
import math

from absl import logging
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

import effnetv2_configs
import hparams
import utils


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, mconfig, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = mconfig.width_coefficient
  divisor = mconfig.depth_divisor
  min_depth = mconfig.min_depth
  if skip or not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  return int(new_filters)


def round_repeats(repeats, multiplier, skip=False):
  """Round number of filters based on depth multiplier."""
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


class SE(tf.keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(self, mconfig, se_filters, output_filters, name=None):
    super().__init__(name=name)

    self._local_pooling = mconfig.local_pooling
    self._data_format = mconfig.data_format
    self._act = utils.get_act_fn(mconfig.act_fn)

    # Squeeze and Excitation layer.
    self._se_reduce = tf.keras.layers.Conv2D(
        se_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d')
    self._se_expand = tf.keras.layers.Conv2D(
        output_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        name='conv2d_1')

  def call(self, inputs):
    h_axis, w_axis = [2, 3] if self._data_format == 'channels_first' else [1, 2]
    if self._local_pooling:
      se_tensor = tf.nn.avg_pool(
          inputs,
          ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
          strides=[1, 1, 1, 1],
          padding='VALID')
    else:
      se_tensor = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se_tensor = self._se_expand(self._act(self._se_reduce(se_tensor)))
    logging.info('Built SE %s : %s', self.name, se_tensor.shape)
    return tf.sigmoid(se_tensor) * inputs


class MBConvBlock(tf.keras.layers.Layer):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, mconfig, name=None):
    """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      mconfig: GlobalParams, a set of global parameters.
      name: layer name.
    """
    super().__init__(name=name)

    self._block_args = copy.deepcopy(block_args)
    self._mconfig = copy.deepcopy(mconfig)
    self._local_pooling = mconfig.local_pooling
    self._data_format = mconfig.data_format
    self._channel_axis = 1 if self._data_format == 'channels_first' else -1

    self._act = utils.get_act_fn(mconfig.act_fn)
    self._has_se = (
        self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)

    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()

  @property
  def block_args(self):
    return self._block_args

  def _build(self):
    """Builds block according to the arguments."""
    # pylint: disable=g-long-lambda
    bid = itertools.count(0)
    get_norm_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    # pylint: enable=g-long-lambda

    mconfig = self._mconfig
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    # Expansion phase. Called if not using fused convolutions and expansion
    # phase is necessary.
    if self._block_args.expand_ratio != 1:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters=filters,
          kernel_size=1,
          strides=1,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          name=get_conv_name())
      self._norm0 = utils.normalization(
          mconfig.bn_type,
          axis=self._channel_axis,
          momentum=mconfig.bn_momentum,
          epsilon=mconfig.bn_epsilon,
          groups=mconfig.gn_groups,
          name=get_norm_name())

    # Depth-wise convolution phase. Called if not using fused convolutions.
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        name='depthwise_conv2d')

    self._norm1 = utils.normalization(
        mconfig.bn_type,
        axis=self._channel_axis,
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups,
        name=get_norm_name())

    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      self._se = SE(self._mconfig, num_reduced_filters, filters, name='se')
    else:
      self._se = None

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        name=get_conv_name())
    self._norm2 = utils.normalization(
        mconfig.bn_type,
        axis=self._channel_axis,
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups,
        name=get_norm_name())

  def residual(self, inputs, x, training, survival_prob):
    if (self._block_args.strides == 1 and
        self._block_args.input_filters == self._block_args.output_filters):
      # Apply only if skip connection presents.
      if survival_prob:
        x = utils.drop_connect(x, training, survival_prob)
      x = tf.add(x, inputs)

    return x

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    logging.info('Block %s input shape: %s (%s)', self.name, inputs.shape,
                 inputs.dtype)
    x = inputs
    if self._block_args.expand_ratio != 1:
      x = self._act(self._norm0(self._expand_conv(x), training=training))
      logging.info('Expand shape: %s', x.shape)

    x = self._act(self._norm1(self._depthwise_conv(x), training=training))
    logging.info('DWConv shape: %s', x.shape)

    if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
      x = tf.keras.layers.Dropout(self._mconfig.conv_dropout)(
          x, training=training)

    if self._se:
      x = self._se(x)

    self.endpoints = {'expansion_output': x}

    x = self._norm2(self._project_conv(x), training=training)
    x = self.residual(inputs, x, training, survival_prob)

    logging.info('Project shape: %s', x.shape)
    return x


class FusedMBConvBlock(MBConvBlock):
  """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""

  def _build(self):
    """Builds block according to the arguments."""
    # pylint: disable=g-long-lambda
    bid = itertools.count(0)
    get_norm_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))
    # pylint: enable=g-long-lambda

    mconfig = self._mconfig
    block_args = self._block_args
    filters = block_args.input_filters * block_args.expand_ratio
    kernel_size = block_args.kernel_size
    if block_args.expand_ratio != 1:
      # Expansion phase:
      self._expand_conv = tf.keras.layers.Conv2D(
          filters,
          kernel_size=kernel_size,
          strides=block_args.strides,
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False,
          name=get_conv_name())
      self._norm0 = utils.normalization(
          mconfig.bn_type,
          axis=self._channel_axis,
          momentum=mconfig.bn_momentum,
          epsilon=mconfig.bn_epsilon,
          groups=mconfig.gn_groups,
          name=get_norm_name())

    if self._has_se:
      num_reduced_filters = max(
          1, int(block_args.input_filters * block_args.se_ratio))
      self._se = SE(mconfig, num_reduced_filters, filters, name='se')
    else:
      self._se = None
    # Output phase:
    filters = block_args.output_filters
    self._project_conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=1 if block_args.expand_ratio != 1 else kernel_size,
        strides=1 if block_args.expand_ratio != 1 else block_args.strides,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name=get_conv_name())
    self._norm1 = utils.normalization(
        mconfig.bn_type,
        axis=self._channel_axis,
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups,
        name=get_norm_name())

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    logging.info('Block %s  input shape: %s', self.name, inputs.shape)
    x = inputs
    if self._block_args.expand_ratio != 1:
      x = self._act(self._norm0(self._expand_conv(x), training=training))
    logging.info('Expand shape: %s', x.shape)

    self.endpoints = {'expansion_output': x}

    if self._mconfig.conv_dropout and self._block_args.expand_ratio > 1:
      x = tf.keras.layers.Dropout(self._mconfig.conv_dropout)(x, training)

    if self._se:
      x = self._se(x)

    x = self._norm1(self._project_conv(x), training=training)
    if self._block_args.expand_ratio == 1:
      x = self._act(x)  # add act if no expansion.

    x = self.residual(inputs, x, training, survival_prob)
    logging.info('Project shape: %s', x.shape)
    return x


class Stem(tf.keras.layers.Layer):
  """Stem layer at the begining of the network."""

  def __init__(self, mconfig, stem_filters, name=None):
    super().__init__(name=name)
    self._conv_stem = tf.keras.layers.Conv2D(
        filters=round_filters(stem_filters, mconfig),
        kernel_size=3,
        strides=2,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=mconfig.data_format,
        use_bias=False,
        name='conv2d')
    self._norm = utils.normalization(
        mconfig.bn_type,
        axis=(1 if mconfig.data_format == 'channels_first' else -1),
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups)
    self._act = utils.get_act_fn(mconfig.act_fn)

  def call(self, inputs, training):
    return self._act(self._norm(self._conv_stem(inputs), training=training))


class Head(tf.keras.layers.Layer):
  """Head layer for network outputs."""

  def __init__(self, mconfig, name=None):
    super().__init__(name=name)

    self.endpoints = {}
    self._mconfig = mconfig

    self._conv_head = tf.keras.layers.Conv2D(
        filters=round_filters(mconfig.feature_size or 1280, mconfig),
        kernel_size=1,
        strides=1,
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=mconfig.data_format,
        use_bias=False,
        name='conv2d')
    self._norm = utils.normalization(
        mconfig.bn_type,
        axis=(1 if mconfig.data_format == 'channels_first' else -1),
        momentum=mconfig.bn_momentum,
        epsilon=mconfig.bn_epsilon,
        groups=mconfig.gn_groups)
    self._act = utils.get_act_fn(mconfig.act_fn)

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=mconfig.data_format)
    if mconfig.num_classes:
      self._fc = tf.keras.layers.Dense(
          mconfig.num_classes,
          kernel_initializer=dense_kernel_initializer,
          bias_initializer=tf.constant_initializer(mconfig.headbias or 0))
    else:
      self._fc = None

    if mconfig.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(mconfig.dropout_rate)
    else:
      self._dropout = None

    self.h_axis, self.w_axis = ([2, 3] if mconfig.data_format
                                == 'channels_first' else [1, 2])

  def call(self, inputs, training):
    """Call the layer."""
    outputs = self._act(self._norm(self._conv_head(inputs), training=training))
    self.endpoints['head_1x1'] = outputs

    if self._mconfig.local_pooling:
      shape = outputs.get_shape().as_list()
      kernel_size = [1, shape[self.h_axis], shape[self.w_axis], 1]
      outputs = tf.nn.avg_pool(
          outputs, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
      self.endpoints['pooled_features'] = outputs
      if self._dropout:
        outputs = self._dropout(outputs, training=training)
      self.endpoints['global_pool'] = outputs
      if self._fc:
        outputs = tf.squeeze(outputs, [self.h_axis, self.w_axis])
        outputs = self._fc(outputs)
      self.endpoints['head'] = outputs
    else:
      outputs = self._avg_pooling(outputs)
      self.endpoints['pooled_features'] = outputs
      if self._dropout:
        outputs = self._dropout(outputs, training=training)
      self.endpoints['global_pool'] = outputs
      if self._fc:
        outputs = self._fc(outputs)
      self.endpoints['head'] = outputs
    return outputs


class EffNetV2Model(tf.keras.Model):
  """A class implements tf.keras.Model.

    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self,
               model_name='efficientnetv2-s',
               model_config=None,
               name=None):
    """Initializes an `Model` instance.

    Args:
      model_name: A string of model name.
      model_config: A dict of model configureations or a string of hparams.
      name: A string of layer name.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super().__init__(name=name or model_name)
    cfg = copy.deepcopy(hparams.base_config)
    if model_name:
      cfg.override(effnetv2_configs.get_model_config(model_name))
    cfg.model.override(model_config)
    self.cfg = cfg
    self._mconfig = cfg.model
    self.endpoints = None
    self._build()

  def _build(self):
    """Builds a model."""
    self._blocks = []

    # Stem part.
    self._stem = Stem(self._mconfig, self._mconfig.blocks_args[0].input_filters)

    # Builds blocks.
    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    for block_args in self._mconfig.blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      input_filters = round_filters(block_args.input_filters, self._mconfig)
      output_filters = round_filters(block_args.output_filters, self._mconfig)
      repeats = round_repeats(block_args.num_repeat,
                              self._mconfig.depth_coefficient)
      block_args.update(
          dict(
              input_filters=input_filters,
              output_filters=output_filters,
              num_repeat=repeats))

      # The first block needs to take care of stride and filter size increase.
      conv_block = {0: MBConvBlock, 1: FusedMBConvBlock}[block_args.conv_type]
      self._blocks.append(
          conv_block(block_args, self._mconfig, name=block_name()))
      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args.input_filters = block_args.output_filters
        block_args.strides = 1
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(
            conv_block(block_args, self._mconfig, name=block_name()))

    # Head part.
    self._head = Head(self._mconfig)

  def summary(self, input_shape=(224, 224, 3), **kargs):
    x = tf.keras.Input(shape=input_shape)
    model = tf.keras.Model(inputs=[x], outputs=self.call(x, training=True))
    return model.summary()

  def call(self, inputs, training, features_only=None):
    """Implementation of call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.

    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    reduction_idx = 0

    # Calls Stem layers
    outputs = self._stem(inputs, training)
    logging.info('Built stem: %s (%s)', outputs.shape, outputs.dtype)
    self.endpoints['stem'] = outputs

    # Calls blocks.
    for idx, block in enumerate(self._blocks):
      is_reduction = False  # reduction flag for blocks after the stem layer
      if ((idx == len(self._blocks) - 1) or
          self._blocks[idx + 1].block_args.strides > 1):
        is_reduction = True
        reduction_idx += 1

      survival_prob = self._mconfig.survival_prob
      if survival_prob:
        drop_rate = 1.0 - survival_prob
        survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
        logging.info('block_%s survival_prob: %s', idx, survival_prob)
      outputs = block(outputs, training=training, survival_prob=survival_prob)
      self.endpoints['block_%s' % idx] = outputs
      if is_reduction:
        self.endpoints['reduction_%s' % reduction_idx] = outputs
      if block.endpoints:
        for k, v in six.iteritems(block.endpoints):
          self.endpoints['block_%s/%s' % (idx, k)] = v
          if is_reduction:
            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['features'] = outputs

    if not features_only:
      # Calls final layers and returns logits.
      outputs = self._head(outputs, training)
      self.endpoints.update(self._head.endpoints)

    return [outputs] + list(
        filter(lambda endpoint: endpoint is not None, [
            self.endpoints.get('reduction_1'),
            self.endpoints.get('reduction_2'),
            self.endpoints.get('reduction_3'),
            self.endpoints.get('reduction_4'),
            self.endpoints.get('reduction_5'),
        ]))
