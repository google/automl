from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import re

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import hparams_config
import keras.utils
from backbone import backbone_factory
from backbone import efficientnet_builder




class ClassNet(tf.keras.layers.Layer):
    def __init__(self,
                 config,
                 name='class_net', **kwargs):

        super(ClassNet, self).__init__(name=name, **kwargs)

        self.config = config

        num_classes = config.num_classes
        num_anchors = len(config.aspect_ratios) * config.num_scales

        num_filters = config.fpn_num_filters
        is_training = config.is_training_bn
        act_type = config.act_type
        repeats = config.box_class_repeats
        separable_conv = config.separable_conv
        survival_prob = config.survival_prob
        use_tpu = config.use_tpu
        data_format = config.data_format

        self.repeats = repeats
        self.use_dc = survival_prob and is_training

        self.conv_ops = []
        self.bn_act_ops = []

        for i in range(repeats):
            # If using SeparableConv2D
            if separable_conv:
                self.conv_ops.append(tf.keras.layers.SeparableConv2D(
                                        filters=num_filters,
                                        depth_multiplier=1,
                                        pointwise_initializer=tf.initializers.variance_scaling(),
                                        depthwise_initializer=tf.initializers.variance_scaling(),
                                        data_format=data_format,
                                        kernel_size=3,
                                        activation=None,
                                        bias_initializer=tf.zeros_initializer(),
                                        padding='same',
                                        name='class-%d' % i))
            # If using Conv2d
            else:
                self.conv_ops.append(tf.keras.layers.Conv2D(
                                        filters=num_filters,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        data_format=data_format,
                                        kernel_size=3,
                                        activation=None,
                                        bias_initializer=tf.zeros_initializer(),
                                        padding='same',
                                        name='class-%d' % i))

            # Level only apply here so it's maybe better inside (no need to use tf.AUTO_REUSE anymore)
            bn_act_ops_per_level = {}
            for level in range(config.min_level, config.max_level + 1):
                bn_act_ops_per_level[level] = keras.utils.Batch_norm_act(is_training,
                                                                   act_type=act_type,
                                                                   init_zero=False,
                                                                   use_tpu=use_tpu,
                                                                   data_format=data_format,
                                                                   name='class-%d-bn-%d' % (i, level))
            self.bn_act_ops.append(bn_act_ops_per_level)

        if self.use_dc:
            self.dc = keras.utils.Drop_connect(survival_prob)

        if separable_conv:
            self.classes = tf.keras.layers.SeparableConv2D(
                                            filters=num_classes * num_anchors,
                                            depth_multiplier=1,
                                            pointwise_initializer=tf.initializers.variance_scaling(),
                                            depthwise_initializer=tf.initializers.variance_scaling(),
                                            data_format=data_format,
                                            kernel_size=3,
                                            activation=None,
                                            bias_initializer=tf.zeros_initializer(),
                                            padding='same',
                                            name='class-predict')

        else:
            self.classes = tf.keras.layers.Conv2D(
                                            filters=num_classes * num_anchors,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            data_format=data_format,
                                            kernel_size=3,
                                            activation=None,
                                            bias_initializer=tf.zeros_initializer(),
                                            padding='same',
                                            name='class-predict')

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
        return self.config


class BoxNet(tf.keras.layers.Layer):
    def __init__(self,
                 config,
                 name='box_net', **kwargs):

        super(BoxNet, self).__init__(name=name, **kwargs)

        self.config=config

        num_anchors = len(config.aspect_ratios) * config.num_scales

        num_filters = config.fpn_num_filters
        is_training = config.is_training_bn
        act_type = config.act_type
        repeats = config.box_class_repeats
        separable_conv = config.separable_conv
        survival_prob = config.survival_prob
        use_tpu = config.use_tpu
        data_format = config.data_format

        self.repeats = repeats
        self.use_dc = survival_prob and is_training

        self.conv_ops = []
        self.bn_act_ops = []

        for i in range(repeats):
            # If using SeparableConv2D
            if separable_conv:
                self.conv_ops.append(tf.keras.layers.SeparableConv2D(
                                        filters=num_filters,
                                        depth_multiplier=1,
                                        pointwise_initializer=tf.initializers.variance_scaling(),
                                        depthwise_initializer=tf.initializers.variance_scaling(),
                                        data_format=data_format,
                                        kernel_size=3,
                                        activation=None,
                                        bias_initializer=tf.zeros_initializer(),
                                        padding='same',
                                        name='box-%d' % i))
            # If using Conv2d
            else:
                self.conv_ops.append(tf.keras.layers.Conv2D(
                                        filters=num_filters,
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        data_format=data_format,
                                        kernel_size=3,
                                        activation=None,
                                        bias_initializer=tf.zeros_initializer(),
                                        padding='same',
                                        name='box-%d' % i))

            # Level only apply here so it's maybe better inside (no need to use tf.AUTO_REUSE anymore)
            bn_act_ops_per_level = {}
            for level in range(config.min_level, config.max_level + 1):
                bn_act_ops_per_level[level] = keras.utils.Batch_norm_act(is_training,
                                                                   act_type=act_type,
                                                                   init_zero=False,
                                                                   use_tpu=use_tpu,
                                                                   data_format=data_format,
                                                                   name='box-%d-bn-%d' % (i, level))
            self.bn_act_ops.append(bn_act_ops_per_level)

        if self.use_dc:
            self.dc = keras.utils.Drop_connect(survival_prob)

        if separable_conv:
            self.boxes = tf.keras.layers.SeparableConv2D(
                                            filters=4 * num_anchors,
                                            depth_multiplier=1,
                                            pointwise_initializer=tf.initializers.variance_scaling(),
                                            depthwise_initializer=tf.initializers.variance_scaling(),
                                            data_format=data_format,
                                            kernel_size=3,
                                            activation=None,
                                            bias_initializer=tf.zeros_initializer(),
                                            padding='same',
                                            name='box-predict')

        else:
            self.boxes = tf.keras.layers.Conv2D(
                                            filters=4 * num_anchors,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            data_format=data_format,
                                            kernel_size=3,
                                            activation=None,
                                            bias_initializer=tf.zeros_initializer(),
                                            padding='same',
                                            name='box-predict')

    def call(self, feats, level=None, **kwargs):
        image = feats
        for i in range(self.repeats):
            original_image = image
            image = self.conv_ops[i](image)
            image = self.bn_act_ops[i][level].call(image)
            if i>0 and self.use_dc:
                image = self.dc.call(image)
                image = image + original_image


        return self.boxes(image)

    def get_config(self):
        return self.config


class BuildClassAndBoxOutputs(tf.keras.layers.Layer):
    """Builds box net and class net.

    Args:
    feats: input tensor.
    config: a dict-like config, including all parameters.

    Returns:
    A tuple (class_outputs, box_outputs) for class/box predictions.
    """

    def __init__(self, config):

        self.class_net = ClassNet(config)
        self.box_net = BoxNet(config)
        self.config = config

    def call(self, feats):

        class_outputs = {}
        box_outputs = {}

        for level in range(self.config.min_level,
                           self.config.max_level + 1):
            class_outputs[level] = self.class_net.call(feats[level], level=level)

        for level in range(self.config.min_level,
                           self.config.max_level + 1):
            box_outputs[level] = self.box_net.call(feats[level], level=level)

        return class_outputs, box_outputs

    def get_config(self):
        return self.config