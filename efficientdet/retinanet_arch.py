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
"""RetinaNet (via ResNet) model definition.

Defines the RetinaNet model architecture:
T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import hparams_config
import utils


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4
_RESNET_MAX_LEVEL = 5


def batch_norm_relu(inputs,
                    is_training_bn,
                    relu=True,
                    init_zero=False,
                    data_format='channels_last',
                    name=None,
                    use_swish=False):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training_bn: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    name: the name of the batch normalization layer
    use_swish:  Whether to use swish or relu as the activation function.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training_bn,
      fused=True,
      gamma_initializer=gamma_initializer,
      name=name)

  if relu:
    inputs = tf.nn.swish(inputs) if use_swish else tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last'):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def residual_block(inputs,
                   filters,
                   is_training_bn,
                   strides,
                   use_projection=False,
                   data_format='channels_last',
                   use_swish=False):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
   use_swish:  Whether to use swish or relu as the activation function.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut in first layer to match filters and strides
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, relu=False, data_format=data_format,
        use_swish=use_swish)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format,
                           use_swish=use_swish)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format,
      use_swish=use_swish)

  if use_swish:
    return tf.nn.swish(inputs + shortcut)
  else:
    return tf.nn.relu(inputs + shortcut)


def bottleneck_block(inputs,
                     filters,
                     is_training_bn,
                     strides,
                     use_projection=False,
                     data_format='channels_last',
                     use_swish=False):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training_bn: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
   use_swish:  Whether to use swish or relu as the activation function.

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs
  if use_projection:
    # Projection shortcut only in first block within a group. Bottleneck blocks
    # end with 4 times the number of filters.
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    shortcut = batch_norm_relu(
        shortcut, is_training_bn, relu=False, data_format=data_format,
        use_swish=use_swish)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format,
                           use_swish=use_swish)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format,
                           use_swish=use_swish)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training_bn,
      relu=False,
      init_zero=True,
      data_format=data_format,
      use_swish=use_swish)

  if use_swish:
    return tf.nn.swish(inputs + shortcut)
  else:
    return tf.nn.relu(inputs + shortcut)


def block_group(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training_bn,
                name,
                data_format='channels_last',
                use_swish=False):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training_bn: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
   use_swish:  Whether to use swish or relu as the activation function.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      is_training_bn,
      strides,
      use_projection=True,
      data_format=data_format,
      use_swish=use_swish)

  for _ in range(1, blocks):
    inputs = block_fn(
        inputs, filters, is_training_bn, 1, data_format=data_format,
        use_swish=use_swish)

  return tf.identity(inputs, name)


def resnet_v1_generator(block_fn, layers, data_format='channels_last',
                        use_swish=False):
  """Generator of ResNet v1 model with classification layers removed.

    Our actual ResNet network.  We return the output of c2, c3,c4,c5
    N.B. batch norm is always run with trained parameters, as we use very small
    batches when training the object layers.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
   use_swish:  Whether to use swish or relu as the activation function.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """
  def model(inputs, is_training_bn=False):
    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=64,
        kernel_size=7,
        strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = batch_norm_relu(inputs, is_training_bn, data_format=data_format,
                             use_swish=use_swish)

    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=3,
        strides=2,
        padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    c2 = block_group(
        inputs=inputs,
        filters=64,
        blocks=layers[0],
        strides=1,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group1',
        data_format=data_format,
        use_swish=use_swish)
    c3 = block_group(
        inputs=c2,
        filters=128,
        blocks=layers[1],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group2',
        data_format=data_format,
        use_swish=use_swish)
    c4 = block_group(
        inputs=c3,
        filters=256,
        blocks=layers[2],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group3',
        data_format=data_format,
        use_swish=use_swish)
    c5 = block_group(
        inputs=c4,
        filters=512,
        blocks=layers[3],
        strides=2,
        block_fn=block_fn,
        is_training_bn=is_training_bn,
        name='block_group4',
        data_format=data_format,
        use_swish=use_swish)
    return c2, c3, c4, c5

  return model


def resnet_v1(resnet_depth, data_format='channels_last', use_swish=False):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      10: {'block': residual_block, 'layers': [1, 1, 1, 1]},
      18: {'block': residual_block, 'layers': [2, 2, 2, 2]},
      26: {'block': bottleneck_block, 'layers': [2, 2, 2, 2]},
      34: {'block': residual_block, 'layers': [3, 4, 6, 3]},
      50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
      101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
      152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
      200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
  }

  if resnet_depth not in model_params:
    raise ValueError('Not a valid resnet_depth:', resnet_depth)

  params = model_params[resnet_depth]
  return resnet_v1_generator(
      params['block'], params['layers'], data_format, use_swish)


def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A float32 tensor of size [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  """
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])


# TODO(b/111271774): Removes this wrapper once b/111271774 is resolved.
def resize_bilinear(images, size, output_type):
  """Returns resized images as output_type.

  Args:
    images: A tensor of size [batch, height_in, width_in, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
      for the images.
    output_type: The destination type.
  Returns:
    A tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_type.
  """
  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, output_type)


## RetinaNet specific layers
def class_net(images, level, num_classes, num_anchors=6, is_training_bn=False):
  """Class prediction network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        activation=None,
        padding='same',
        name='class-%d' % i)
    # The convolution layers in the class net are shared among all levels, but
    # each level has its batch normlization to capture the statistical
    # difference among different levels.
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='class-%d-bn-%d' % (i, level))

  classes = tf.layers.conv2d(
      images,
      num_classes * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='class-predict')

  return classes


def box_net(images, level, num_anchors=6, is_training_bn=False):
  """Box regression network for RetinaNet."""
  for i in range(4):
    images = tf.layers.conv2d(
        images,
        256,
        kernel_size=(3, 3),
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        padding='same',
        name='box-%d' % i)
    # The convolution layers in the box net are shared among all levels, but
    # each level has its batch normlization to capture the statistical
    # difference among different levels.
    images = batch_norm_relu(images, is_training_bn, relu=True, init_zero=False,
                             name='box-%d-bn-%d' % (i, level))

  boxes = tf.layers.conv2d(
      images,
      4 * num_anchors,
      kernel_size=(3, 3),
      bias_initializer=tf.zeros_initializer(),
      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
      padding='same',
      name='box-predict')

  return boxes


def resnet_fpn(features,
               min_level=3,
               max_level=7,
               resnet_depth=50,
               is_training_bn=False,
               use_nearest_upsampling=True):
  """ResNet feature pyramid networks."""
  # upward layers
  with tf.variable_scope('resnet%s' % resnet_depth):
    resnet_fn = resnet_v1(resnet_depth)
    u2, u3, u4, u5 = resnet_fn(features, is_training_bn)

  feats_bottom_up = {
      2: u2,
      3: u3,
      4: u4,
      5: u5,
  }

  with tf.variable_scope('resnet_fpn'):
    # lateral connections
    feats_lateral = {}
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats_lateral[level] = tf.layers.conv2d(
          feats_bottom_up[level],
          filters=256,
          kernel_size=(1, 1),
          padding='same',
          name='l%d' % level)

    # add top-down path
    feats = {_RESNET_MAX_LEVEL: feats_lateral[_RESNET_MAX_LEVEL]}
    for level in range(_RESNET_MAX_LEVEL - 1, min_level - 1, -1):
      if use_nearest_upsampling:
        feats[level] = nearest_upsampling(feats[level + 1],
                                          2) + feats_lateral[level]
      else:
        feats[level] = resize_bilinear(
            feats[level + 1], tf.shape(feats_lateral[level])[1:3],
            feats[level + 1].dtype) + feats_lateral[level]

    # add post-hoc 3x3 convolution kernel
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats[level] = tf.layers.conv2d(
          feats[level],
          filters=256,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same',
          name='post_hoc_d%d' % level)

    # coarser FPN levels introduced for RetinaNet
    for level in range(_RESNET_MAX_LEVEL + 1, max_level + 1):
      feats_in = feats[level - 1]
      if level > _RESNET_MAX_LEVEL + 1:
        feats_in = tf.nn.relu(feats_in)
      feats[level] = tf.layers.conv2d(
          feats_in,
          filters=256,
          strides=(2, 2),
          kernel_size=(3, 3),
          padding='same',
          name='p%d' % level)
    # add batchnorm
    for level in range(min_level, max_level + 1):
      feats[level] = tf.layers.batch_normalization(
          inputs=feats[level],
          momentum=_BATCH_NORM_DECAY,
          epsilon=_BATCH_NORM_EPSILON,
          center=True,
          scale=True,
          training=is_training_bn,
          fused=True,
          name='p%d-bn' % level)

  return feats


def retinanet(features, model_name='retinanet-50', config=None, **kwargs):
  """RetinaNet classification and regression model."""
  if not config:
    config = hparams_config.get_retinanet_config(model_name)
  config.override(kwargs)

  min_level = config.get('min_level', 3)
  max_level = config.get('max_level', 7)
  num_classes = config.get('num_classes', 90)
  resnet_depth = config.get('resnet_depth', 50)
  use_nearest_upsampling = config.get('resnet_depth', True)
  is_training_bn = config.get('is_training_bn', False)
  num_anchors = len(config.aspect_ratios) * config.num_scales

  # create feature pyramid networks
  feats = resnet_fpn(features, min_level, max_level, resnet_depth,
                     is_training_bn, use_nearest_upsampling)
  logging.info('backbone+fpn params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))
  # add class net and box net in RetinaNet. The class net and the box net are
  # shared among all the levels.
  with tf.variable_scope('retinanet'):
    class_outputs = {}
    box_outputs = {}
    with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        class_outputs[level] = class_net(feats[level], level, num_classes,
                                         num_anchors, is_training_bn)
    with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
      for level in range(min_level, max_level + 1):
        box_outputs[level] = box_net(feats[level], level,
                                     num_anchors, is_training_bn)
  logging.info('backbone+fpn params/flops = {:.6f}M, {:.9f}B'.format(
      *utils.num_params_flops()))

  return class_outputs, box_outputs


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
