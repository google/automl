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
"""Tests for efficientdet_arch_keras."""
from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow as tf2

import efficientdet_arch as legacy_arch
import hparams_config
from keras import efficientdet_arch_keras
import functools
import utils

SEED = 111111


class KerasTest(tf.test.TestCase):
  def test_resample_feature_adder_compile(self):
    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    tf2.random.set_seed(SEED)
    inputs = [
          tf2.keras.Input(shape=[512, 512, 3]),
          tf2.keras.Input(shape=[256, 256, 16]),
          tf2.keras.Input(shape=[128, 128, 24]),
          tf2.keras.Input(shape=[64, 64, 40]),
          tf2.keras.Input(shape=[32, 32, 112]),
          tf2.keras.Input(shape=[16, 16, 320])
    ]
    outputs = efficientdet_arch_keras.ResampleFeatureAdder(config)(inputs)
    model = tf2.keras.Model(inputs=inputs, outputs=outputs)

    examples = [[
	  tf2.ones([1, 512, 512, 3]),
	  tf2.ones([1, 256, 256, 16]),
	  tf2.ones([1, 128, 128, 24]),
	  tf2.ones([1, 64, 64, 40]),
	  tf2.ones([1, 32, 32, 112]),
	  tf2.ones([1, 16, 16, 320])
    ]]

    preds = model(examples)

    try:
      utils.verify_feats_size(preds,
                              feat_sizes=feat_sizes,
                              min_level=config.min_level,
                              max_level=config.max_level,
                              data_format=config.data_format)
    except ValueError as err:
      self.assertFalse(True, msg=repr(err))
    self.assertEqual(len(preds), 5, "P3-P7")

  def test_fuse_features(self):
    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    for weight_method in ['attn', 'fastattn', 'channel_attn', 'channel_fastattn', 'sum']:
      try:
        fuse_feature = efficientdet_arch_keras.FuseFeatures([0, 1, 2], weight_method, config.fpn_num_filters)
      except ValueError as err:
        self.assertFalse(True, msg=repr(err))

  def test_fnode_compile(self):
    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    fpn_config = legacy_arch.get_fpn_config(config.fpn_name, config.min_level,
                                            config.max_level,
                                            config.fpn_weight_method)
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    i = 0
    fnode_cfg = fpn_config.nodes[i]

    examples = [[
          tf2.ones([1, 512, 512, 3]),
          tf2.ones([1, 256, 256, 16]),
          tf2.ones([1, 128, 128, 24]),
          tf2.ones([1, 64, 64, 40]),
          tf2.ones([1, 32, 32, 112]),
          tf2.ones([1, 16, 16, 320])
      ]]
    inputs = [
          tf2.keras.Input(shape=[512, 512, 3]),
          tf2.keras.Input(shape=[256, 256, 16]),
          tf2.keras.Input(shape=[128, 128, 24]),
          tf2.keras.Input(shape=[64, 64, 40]),
          tf2.keras.Input(shape=[32, 32, 112]),
          tf2.keras.Input(shape=[16, 16, 320])
    ]

    x = efficientdet_arch_keras.ResampleFeatureAdder(config)(inputs)
    outputs = efficientdet_arch_keras.FNode(feat_sizes[fnode_cfg['feat_level']]['height'],
                  feat_sizes[fnode_cfg['feat_level']]['width'],
                  fnode_cfg['inputs_offsets'],
                  config.fpn_num_filters,
                  config.apply_bn_for_resampling,
                  config.is_training_bn,
                  config.conv_after_downsample,
                  config.use_native_resize_op,
                  config.pooling_type,
                  config.conv_bn_act_pattern,
                  config.separable_conv,
                  config.act_type,
                  strategy=config.strategy,
                  weight_method=fpn_config.weight_method,
                  data_format=config.data_format,
                  name='fnode{}'.format(i))(x)
    model = tf2.keras.Model(inputs=inputs, outputs=outputs)
    preds = model(examples)

    self.assertEqual(len(preds), 6, msg="Expected that FNode will add one more node (P6') to initial 5 (P3 - P7)")
    self.assertEqual(feat_sizes[fnode_cfg['feat_level']]['height'], preds[5].shape[1])
    self.assertEqual(feat_sizes[fnode_cfg['feat_level']]['width'], preds[5].shape[2])

  def test_many_fnodes_compile(self):
    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    fpn_config = legacy_arch.get_fpn_config(config.fpn_name, config.min_level,
                                            config.max_level,
                                            config.fpn_weight_method)
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    fnode_cfg_0 = fpn_config.nodes[0]
    fnode_cfg_1 = fpn_config.nodes[1]

    examples = [[
          tf2.ones([1, 512, 512, 3]),
          tf2.ones([1, 256, 256, 16]),
          tf2.ones([1, 128, 128, 24]),
          tf2.ones([1, 64, 64, 40]),
          tf2.ones([1, 32, 32, 112]),
          tf2.ones([1, 16, 16, 320])
      ]]
    inputs = [
          tf2.keras.Input(shape=[512, 512, 3]),
          tf2.keras.Input(shape=[256, 256, 16]),
          tf2.keras.Input(shape=[128, 128, 24]),
          tf2.keras.Input(shape=[64, 64, 40]),
          tf2.keras.Input(shape=[32, 32, 112]),
          tf2.keras.Input(shape=[16, 16, 320])
    ]

    x = efficientdet_arch_keras.ResampleFeatureAdder(config)(inputs)
    x = efficientdet_arch_keras.FNode(feat_sizes[fnode_cfg_0['feat_level']]['height'],
                  feat_sizes[fnode_cfg_0['feat_level']]['width'],
                  fnode_cfg_0['inputs_offsets'],
                  config.fpn_num_filters,
                  config.apply_bn_for_resampling,
                  config.is_training_bn,
                  config.conv_after_downsample,
                  config.use_native_resize_op,
                  config.pooling_type,
                  config.conv_bn_act_pattern,
                  config.separable_conv,
                  config.act_type,
                  strategy=config.strategy,
                  weight_method=fpn_config.weight_method,
                  data_format=config.data_format,
                  name='fnode{}'.format(0))(x)
    outputs = efficientdet_arch_keras.FNode(feat_sizes[fnode_cfg_1['feat_level']]['height'],
                  feat_sizes[fnode_cfg_1['feat_level']]['width'],
                  fnode_cfg_1['inputs_offsets'],
                  config.fpn_num_filters,
                  config.apply_bn_for_resampling,
                  config.is_training_bn,
                  config.conv_after_downsample,
                  config.use_native_resize_op,
                  config.pooling_type,
                  config.conv_bn_act_pattern,
                  config.separable_conv,
                  config.act_type,
                  strategy=config.strategy,
                  weight_method=fpn_config.weight_method,
                  data_format=config.data_format,
                  name='fnode{}'.format(1))(x)
    model = tf2.keras.Model(inputs=inputs, outputs=outputs)
    preds = model(examples)

    self.assertEqual(len(preds), 7, msg="Expected that FNode will add two more node (P6', P7') to initial 5 (P3 - P7)")
    self.assertEqual(feat_sizes[fnode_cfg_1['feat_level']]['height'], preds[6].shape[1])
    self.assertEqual(feat_sizes[fnode_cfg_1['feat_level']]['width'], preds[6].shape[2])

  def test_fpncell_compile(self):
    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    inputs = [
          tf2.keras.Input(shape=[64, 64, 40]),
          tf2.keras.Input(shape=[32, 32, 112]),
          tf2.keras.Input(shape=[16, 16, 320]),
          tf2.keras.Input(shape=[8, 8, 64]),
          tf2.keras.Input(shape=[4, 4, 64]),
    ]

    outputs = efficientdet_arch_keras.FPNCell(feat_sizes, config, name='cell_{}'.format(0))(inputs)
    model = tf2.keras.Model(inputs=inputs, outputs=outputs)

    examples = [
          tf2.ones([1, 64, 64, 40]),
          tf2.ones([1, 32, 32, 112]),
          tf2.ones([1, 16, 16, 320]),
          tf2.ones([1, 8, 8, 64]),
          tf2.ones([1, 4, 4, 64]),
    ]
    preds = model(examples)
    self.assertEqual(len(preds), 13)

  def test_fpncells_compile(self):
    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    feat_sizes = utils.get_feat_sizes(config.image_size, config.max_level)
    inputs = [
          tf2.keras.Input(shape=[64, 64, 40]),
          tf2.keras.Input(shape=[32, 32, 112]),
          tf2.keras.Input(shape=[16, 16, 320]),
          tf2.keras.Input(shape=[8, 8, 64]),
          tf2.keras.Input(shape=[4, 4, 64]),
    ]

    outputs = efficientdet_arch_keras.FPNCells(feat_sizes, config, name='cell_{}'.format(0))(inputs)
    model = tf2.keras.Model(inputs=inputs, outputs=outputs)

    examples = [
          tf2.ones([1, 64, 64, 40]),
          tf2.ones([1, 32, 32, 112]),
          tf2.ones([1, 16, 16, 320]),
          tf2.ones([1, 8, 8, 64]),
          tf2.ones([1, 4, 4, 64]),
    ]
    preds = model(examples)
    self.assertEqual(len(preds), 5)

  def test_model_output(self):
    inputs_shape = [1, 512, 512, 3]
    config = hparams_config.get_efficientdet_config("efficientdet-d0")
    with tf.Session(graph=tf.Graph()) as sess:
      inputs = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      features, backbone_outputs = efficientdet_arch_keras.build_backbone(inputs, config)
      fpn_feats = efficientdet_arch_keras.build_feature_network(features, config)
      class_outputs1, box_outputs1 = efficientdet_arch_keras.build_class_and_box_outputs(fpn_feats, config)
      sess.run(tf.global_variables_initializer())
      class_output1, box_output1 = sess.run([class_outputs1, box_outputs1])
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      class_outputs2, box_outputs2 = legacy_arch.efficientdet(
          feats, 'efficientdet-d0')
      sess.run(tf.global_variables_initializer())
      class_output2, box_output2 = sess.run([class_outputs2, box_outputs2])

    for i in range(3, 8):
      self.assertAllEqual(class_output1[i - 3], class_output2[i])
      self.assertAllEqual(box_output1[i - 3], box_output2[i])

  def test_build_feature_network(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')

    with tf.Session(graph=tf.Graph()) as sess:
      tf.random.set_random_seed(SEED)
      inputs = [
            tf.ones([1, 512, 512, 3]),
            tf.ones([1, 256, 256, 16]),
            tf.ones([1, 128, 128, 24]),
            tf.ones([1, 64, 64, 40]),
            tf.ones([1, 32, 32, 112]),
            tf.ones([1, 16, 16, 320])
      ]
      
      outputs = efficientdet_arch_keras.build_feature_network(inputs, config)
      sess.run(tf.global_variables_initializer()) 
      new_feats1 = sess.run(outputs)

    with tf.Session(graph=tf.Graph()) as sess:
      inputs = {
          0: tf.ones([1, 512, 512, 3]),
          1: tf.ones([1, 256, 256, 16]),
          2: tf.ones([1, 128, 128, 24]),
          3: tf.ones([1, 64, 64, 40]),
          4: tf.ones([1, 32, 32, 112]),
          5: tf.ones([1, 16, 16, 320])
      }
      tf.random.set_random_seed(SEED)
      new_feats2 = legacy_arch.build_feature_network(inputs, config)
      sess.run(tf.global_variables_initializer())
      new_feats2 = sess.run(new_feats2)

    for i in range(config.min_level, config.max_level + 1):
      self.assertAllEqual(new_feats1[i - config.min_level], new_feats2[i])

  def _test_model_variables(self):
    with tf.Graph().as_default():
      feats = tf.random.uniform([1, 512, 512, 3])
      model = efficientdet_arch_keras.efficientdet('efficientdet-d0')
      model(feats)
      vars1 = [var.name for var in model.trainable_variables]
      vars3 = [var.name for var in model.variables]
    with tf.Graph().as_default():
      feats = tf.constant(feats)
      legacy_arch.efficientdet(feats, 'efficientdet-d0')
      vars2 = [var.name for var in tf.trainable_variables()]
      vars4 = [var.name for var in tf.global_variables()]
    vars1.sort()
    vars2.sort()
    self.assertEqual(vars1, vars2)
    vars3.sort()
    vars4.sort()
    self.assertEqual(vars3, vars4)

  def test_resample_feature_map(self):
    feat = tf.random.uniform([1, 16, 16, 320])
    for apply_bn in [True, False]:
      for is_training in [True, False]:
        for strategy in ['tpu', '']:
          with self.subTest(apply_bn=apply_bn,
                            is_training=is_training,
                            strategy=strategy):
            tf.random.set_random_seed(SEED)
            expect_result = legacy_arch.resample_feature_map(
                feat,
                name='resample_p0',
                target_height=8,
                target_width=8,
                target_num_channels=64,
                apply_bn=apply_bn,
                is_training=is_training,
                strategy=strategy)
            tf.random.set_random_seed(SEED)
            resample_layer = efficientdet_arch_keras.ResampleFeatureMap(
                name='resample_p0',
                target_height=8,
                target_width=8,
                target_num_channels=64,
                apply_bn=apply_bn,
                is_training=is_training,
                strategy=strategy)
            actual_result = resample_layer(feat)
            self.assertAllCloseAccordingToType(expect_result, actual_result)

  def _test_op_name(self):
    with tf.Graph().as_default():
      feat = tf.random.uniform([1, 16, 16, 320])
      resample_layer = efficientdet_arch_keras.ResampleFeatureMap(
          name='resample_p0',
          target_height=8,
          target_width=8,
          target_num_channels=64)
      resample_layer(feat)
      vars1 = [var.name for var in tf.trainable_variables()]

    with tf.Graph().as_default():
      feat = tf.random.uniform([1, 16, 16, 320])
      legacy_arch.resample_feature_map(feat,
                                       name='p0',
                                       target_height=8,
                                       target_width=8,
                                       target_num_channels=64)
      vars2 = [var.name for var in tf.trainable_variables()]

    self.assertEqual(vars1, vars2)


class EfficientDetVariablesNamesTest(tf.test.TestCase):

  def build_model(self, keras=False):
    with tf.Graph().as_default():
      config = hparams_config.get_efficientdet_config()
      inputs_shape = [1, 512, 512, 3]
      inputs = dict()
      for i in range(config.min_level, config.max_level + 1):
        inputs[i] = tf.ones(shape=inputs_shape, name='input', dtype=tf.float32)

      if not keras:
        legacy_arch.build_class_and_box_outputs(inputs, config)
      else:
        efficientdet_arch_keras.build_class_and_box_outputs(inputs, config)
      return [n.name for n in tf.global_variables()]

  def _test_graph_variables_name_compatibility(self):
    legacy_names = self.build_model(False)
    keras_names = self.build_model(True)

    self.assertEqual(legacy_names, keras_names)

  def test_output(self):
    config = hparams_config.get_efficientdet_config()
    inputs_shape = [1, 512, 512, 3]

    with tf.Session(graph=tf.Graph()) as sess:
      inputs = list()
      for i in range(config.min_level, config.max_level + 1):
        inputs.append(tf.ones(shape=inputs_shape, name='input', dtype=tf.float32))
      tf.random.set_random_seed(SEED)
      output1 = efficientdet_arch_keras.build_class_and_box_outputs(
          inputs, config)
      sess.run(tf.global_variables_initializer())
      class_output1, box_output1 = sess.run(output1)
    with tf.Session(graph=tf.Graph()) as sess:
      inputs = dict()
      for i in range(config.min_level, config.max_level + 1):
        inputs[i] = tf.ones(shape=inputs_shape, name='input', dtype=tf.float32)
      tf.random.set_random_seed(SEED)
      output2 = legacy_arch.build_class_and_box_outputs(inputs, config)
      sess.run(tf.global_variables_initializer())
      class_output2, box_output2 = sess.run(output2)
    for i in range(config.min_level, config.max_level + 1):
      self.assertAllEqual(class_output1[i - config.min_level], class_output2[i])
      self.assertAllEqual(box_output1[i - config.min_level], box_output2[i])

if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
