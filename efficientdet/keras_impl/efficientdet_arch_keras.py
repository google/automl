import functools
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.utils import conv_utils
from efficientdet_arch import nearest_upsampling
import utils


class ResampleFeatureMap(tf.keras.layers.Layer):
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
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.target_num_channels = target_num_channels
    self.target_height = target_height
    self.target_width = target_width
    self.use_tpu = use_tpu
    self.conv_after_downsample = conv_after_downsample
    self.use_native_resize_op = use_native_resize_op
    self.pooling_type = pooling_type
    self.conv2d = tf.keras.layers.Conv2D(
        self.target_num_channels,
        (1, 1),
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
          (height_scale, width_scale),
          data_format=self.data_format)
    else:
      self.upsample2d = functools.partial(nearest_upsampling,
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
