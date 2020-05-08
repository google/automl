"""A keras implementation of the efficientdet architecture."""

import tensorflow.compat.v1 as tf

import efficientdet_arch
import utils


class BiFPNLayer(tf.keras.layers.Layer):
    """A Keras Layer implementing Bidirectional Feature Pyramids"""

    def __init__(self,
                 min_level: int,
                 max_level: int,
                 image_size: int,
                 fpn_weight_method: str,
                 apply_bn_for_resampling: bool,
                 is_training_bn: bool,
                 conv_after_downsample: bool,
                 use_native_resize_op: bool,
                 data_format: str,
                 pooling_type: str,
                 fpn_num_filters: int,
                 conv_bn_act_pattern: bool,
                 act_type: str,
                 separable_conv: bool,
                 use_tpu: bool,
                 fpn_name: str,
                 **kwargs):

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
        return efficientdet_arch.build_bifpn_layer(feats, self.feat_sizes, self)

    def get_config(self):
        base_config = super(BiFPNLayer, self).get_config()

        return {
            **base_config,
            "min_level": self.min_level,
            "max_level": self.max_level,
            "image_size": self.image_size,
            "fpn_name": self.fpn_name,
            "fpn_weight_method": self.fpn_weight_method,
            "apply_bn_for_resampling": self.apply_bn_for_resampling,
            "is_training_bn": self.is_training_bn,
            "conv_after_downsample": self.conv_after_downsample,
            "use_native_resize_op": self.use_native_resize_op,
            "data_format": self.data_format,
            "pooling_type": self.pooling_type,
            "fpn_num_filters": self.fpn_num_filters,
            "conv_bn_act_pattern": self.conv_bn_act_pattern,
            "act_type": self.act_type,
            "separable_conv": self.separable_conv,
            "use_tpu": self.use_tpu,
        }
