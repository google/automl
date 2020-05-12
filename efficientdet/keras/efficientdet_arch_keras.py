from __future__ import absolute_import, division, print_function

import tensorflow.compat.v1 as tf
from absl import logging

import efficientdet_arch
import hparams_config
import keras.utils_keras
import utils


class ClassNet(tf.keras.layers.Layer):
    def __init__(self,
                 num_classes=80,
                 num_anchors=9,
                 num_filters=32,
                 min_level=3,
                 max_level=7,
                 is_training=False,
                 act_type='swish',
                 repeats=4,
                 separable_conv=True,
                 survival_prob=None,
                 use_tpu=False,
                 data_format='channels_last',
                 name='class_net', **kwargs):

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
        self.use_tpu = use_tpu
        self.data_format = data_format
        self.use_dc = survival_prob and is_training

        self.conv_ops = []
        self.bn_act_ops = []

        for i in range(self.repeats):
            # If using SeparableConv2D
            if self.separable_conv:
                self.conv_ops.append(tf.keras.layers.SeparableConv2D(
                    filters=self.num_filters,
                    depth_multiplier=1,
                    pointwise_initializer=tf.initializers.variance_scaling(),
                    depthwise_initializer=tf.initializers.variance_scaling(),
                    data_format=self.data_format,
                    kernel_size=3,
                    activation=None,
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    name=f'{self.name}/class-%d' % i))
            # If using Conv2d
            else:
                self.conv_ops.append(tf.keras.layers.Conv2D(
                    filters=self.num_filters,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    data_format=self.data_format,
                    kernel_size=3,
                    activation=None,
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    name=f'{self.name}/class-%d' % i))

            # Level only apply here so it's maybe better inside (no need to use tf.AUTO_REUSE anymore)
            bn_act_ops_per_level = {}
            for level in range(self.min_level, self.max_level + 1):
                bn_act_ops_per_level[level] = keras.utils_keras.BatchNormAct(self.is_training,
                                                                             act_type=self.act_type,
                                                                             init_zero=False,
                                                                             use_tpu=self.use_tpu,
                                                                             data_format=self.data_format,
                                                                             name='class-%d-bn-%d' % (i, level),
                                                                             parent_name=self.name)
            self.bn_act_ops.append(bn_act_ops_per_level)

        if self.use_dc:
            self.dc = keras.utils_keras.DropConnect(self.survival_prob)

        if self.separable_conv:
            self.classes = tf.keras.layers.SeparableConv2D(
                filters=self.num_classes * self.num_anchors,
                depth_multiplier=1,
                pointwise_initializer=tf.initializers.variance_scaling(),
                depthwise_initializer=tf.initializers.variance_scaling(),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=efficientdet_arch.FIANL_CONV_INITIALIZER,
                padding='same',
                name=f'{self.name}/class-predict')

        else:
            self.classes = tf.keras.layers.Conv2D(
                filters=self.num_classes * self.num_anchors,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=efficientdet_arch.FIANL_CONV_INITIALIZER,
                padding='same',
                name=f'{self.name}/class-predict')

    def call(self, feats, level=None, **kwargs):
        image = feats
        for i in range(self.repeats):
            original_image = image
            image = self.conv_ops[i](image)
            image = self.bn_act_ops[i][level].call(image)
            if i > 0 and self.use_dc:
                image = self.dc.call(image)
                image = image + original_image

        return self.classes(image)

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
            'use_tpu': self.use_tpu,
            'data_format': self.data_format,
        }


class BoxNet(tf.keras.layers.Layer):
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
                 use_tpu=False,
                 data_format='channels_last',
                 name='box_net', **kwargs):

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
        self.use_tpu = use_tpu
        self.data_format = data_format
        self.use_dc = survival_prob and is_training

        self.conv_ops = []
        self.bn_act_ops = []

        for i in range(self.repeats):
            # If using SeparableConv2D
            if self.separable_conv:
                self.conv_ops.append(tf.keras.layers.SeparableConv2D(
                    filters=self.num_filters,
                    depth_multiplier=1,
                    pointwise_initializer=tf.initializers.variance_scaling(),
                    depthwise_initializer=tf.initializers.variance_scaling(),
                    data_format=self.data_format,
                    kernel_size=3,
                    activation=None,
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    name=f'{self.name}/box-%d' % i))
            # If using Conv2d
            else:
                self.conv_ops.append(tf.keras.layers.Conv2D(
                    filters=self.num_filters,
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    data_format=self.data_format,
                    kernel_size=3,
                    activation=None,
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    name=f'{self.name}/box-%d' % i))

            # Level only apply here so it's maybe better inside (no need to use tf.AUTO_REUSE anymore)
            bn_act_ops_per_level = {}
            for level in range(self.min_level, self.max_level + 1):
                bn_act_ops_per_level[level] = keras.utils_keras.BatchNormAct(self.is_training,
                                                                             act_type=self.act_type,
                                                                             init_zero=False,
                                                                             use_tpu=self.use_tpu,
                                                                             data_format=self.data_format,
                                                                             name='box-%d-bn-%d' % (i, level),
                                                                             parent_name=self.name)
            self.bn_act_ops.append(bn_act_ops_per_level)

        if self.use_dc:
            self.dc = keras.utils_keras.DropConnect(self.survival_prob)

        if self.separable_conv:
            self.boxes = tf.keras.layers.SeparableConv2D(
                filters=4 * self.num_anchors,
                depth_multiplier=1,
                pointwise_initializer=tf.initializers.variance_scaling(),
                depthwise_initializer=tf.initializers.variance_scaling(),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name=f'{self.name}/box-predict')

        else:
            self.boxes = tf.keras.layers.Conv2D(
                filters=4 * self.num_anchors,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                data_format=self.data_format,
                kernel_size=3,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name=f'{self.name}/box-predict')

    def call(self, feats, level=None, **kwargs):
        image = feats
        for i in range(self.repeats):
            original_image = image
            image = self.conv_ops[i](image)
            image = self.bn_act_ops[i][level].call(image)
            if i > 0 and self.use_dc:
                image = self.dc.call(image)
                image = image + original_image

        return self.boxes(image)

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
            'use_tpu': self.use_tpu,
            'data_format': self.data_format,
        }


class BuildClassAndBoxOutputs(tf.keras.layers.Layer):
    """Builds box net and class net.

    Args:
    feats: input tensor.
    config: a dict-like config, including all parameters.

    Returns:
    A tuple (class_outputs, box_outputs) for class/box predictions.
    """

    def __init__(self, aspect_ratios, num_scales, num_classes, fpn_num_filters, min_level, max_level, is_training_bn,
                 act_type,
                 box_class_repeats, separable_conv, survival_prob, use_tpu, data_format, **kwargs):

        self.aspect_ratios = aspect_ratios
        self.num_scales = num_scales
        self.num_classes = num_classes
        self.fpn_num_filters = fpn_num_filters
        self.min_level = min_level
        self.max_level = max_level
        self.is_training_bn = is_training_bn
        self.act_type = act_type
        self.box_class_repeats = box_class_repeats
        self.separable_conv = separable_conv
        self.survival_prob = survival_prob
        self.use_tpu = use_tpu
        self.data_format = data_format

        options = {
            'num_anchors': len(aspect_ratios) * num_scales,
            'num_filters': fpn_num_filters,
            'min_level': min_level,
            'max_level': max_level,
            'is_training': is_training_bn,
            'act_type': act_type,
            'repeats': box_class_repeats,
            'separable_conv': separable_conv,
            'survival_prob': survival_prob,
            'use_tpu': use_tpu,
            'data_format': data_format
        }

        super(BuildClassAndBoxOutputs, self).__init__()

        self.box_net = BoxNet(**options)

        options['num_classes'] = num_classes

        self.class_net = ClassNet(**options)

    def call(self, feats):

        class_outputs = {}
        box_outputs = {}

        for level in range(self.min_level,
                           self.max_level + 1):
            class_outputs[level] = self.class_net.call(feats[level], level=level)

        for level in range(self.min_level,
                           self.max_level + 1):
            box_outputs[level] = self.box_net.call(feats[level], level=level)

        return class_outputs, box_outputs

    def get_config(self):
        base_config = super(BuildClassAndBoxOutputs, self).get_config()

        return {
            **base_config,
            'aspect_ratios': self.aspect_ratios,
            'num_scales': self.num_scales,
            'num_classes': self.num_classes,
            'fpn_num_filters': self.fpn_num_filters,
            'min_level': self.min_level,
            'max_level': self.max_level,
            'is_training_bn': self.is_training_bn,
            'act_type': self.act_type,
            'box_class_repeats': self.box_class_repeats,
            'separable_conv': self.separable_conv,
            'survival_prob': self.survival_prob,
            'use_tpu': self.use_tpu,
            'data_format': self.data_format
        }


def efficientdet(features, model_name=None, config=None, **kwargs):
    """Build EfficientDet model."""
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
    features = efficientdet_arch.build_backbone(features, config)
    logging.info('backbone params/flops = {:.6f}M, {:.9f}B'.format(
        *utils.num_params_flops()))

    # build feature network.
    fpn_feats = efficientdet_arch.build_feature_network(features, config)
    logging.info('backbone+fpn params/flops = {:.6f}M, {:.9f}B'.format(
        *utils.num_params_flops()))

    # build class and box predictions.
    class_box = BuildClassAndBoxOutputs(**config)
    class_outputs, box_outputs = class_box.call(fpn_feats)
    logging.info('backbone+fpn+box params/flops = {:.6f}M, {:.9f}B'.format(
        *utils.num_params_flops()))

    return class_outputs, box_outputs
