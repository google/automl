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

import efficientdet_arch as legacy_arch
import hparams_config
from keras import efficientdet_arch_keras

SEED = 111111


class KerasTest(tf.test.TestCase):

  def test_backbone(self):
    inputs_shape = [1, 512, 512, 3]
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      feats, _ = efficientdet_arch_keras.build_backbone(feats, config)
      sess.run(tf.global_variables_initializer())
      feats1 = sess.run(feats)
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      feats = legacy_arch.build_backbone(feats, config)
      sess.run(tf.global_variables_initializer())
      feats2 = sess.run(feats)
    for key in list(feats.keys()):
      self.assertAllEqual(feats1[key], feats2[key])

  def test_model_output(self):
    inputs_shape = [1, 512, 512, 3]
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      feats, _ = efficientdet_arch_keras.build_backbone(feats, config)
      feats = efficientdet_arch_keras.build_feature_network(feats, config)
      feats = efficientdet_arch_keras.build_class_and_box_outputs(feats, config)
      sess.run(tf.global_variables_initializer())
      class_output1, box_output1 = sess.run(feats)
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      feats = legacy_arch.build_backbone(feats, config)
      feats = legacy_arch.build_feature_network(feats, config)
      feats = legacy_arch.build_class_and_box_outputs(feats, config)
      sess.run(tf.global_variables_initializer())
      class_output2, box_output2 = sess.run(feats)
    for i in range(3, 8):
      self.assertAllEqual(class_output1[i], class_output2[i])
      self.assertAllEqual(box_output1[i], box_output2[i])

  def test_build_feature_network(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
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
      new_feats1 = efficientdet_arch_keras.build_feature_network(inputs, config)
      sess.run(tf.global_variables_initializer())
      new_feats1 = sess.run(new_feats1)
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
      self.assertAllEqual(new_feats1[i], new_feats2[i])

  def test_model_variables(self):
    # TODO(tanmingxing): Re-enable this code once pass internal tests.
    # feats = tf.ones([1, 512, 512, 3])
    # model = efficientdet_arch_keras.efficientdet('efficientdet-d0')
    # model(feats)
    # vars1 = sorted([var.name for var in model.trainable_variables])
    # vars2 = sorted([var.name for var in model.variables])
    with tf.Graph().as_default():
      feats = tf.ones([1, 512, 512, 3])
      model = efficientdet_arch_keras.efficientdet('efficientdet-d0')
      model(feats)
      vars3 = sorted([var.name for var in model.trainable_variables])
      vars4 = sorted([var.name for var in model.variables])
    with tf.Graph().as_default():
      feats = tf.ones([1, 512, 512, 3])
      legacy_arch.efficientdet(feats, 'efficientdet-d0')
      vars5 = sorted([var.name for var in tf.trainable_variables()])
      vars6 = sorted([var.name for var in tf.global_variables()])

    # self.assertEqual(vars1, vars3)
    self.assertEqual(vars3, vars5)
    # self.assertEqual(vars2, vars4)
    self.assertEqual(vars4, vars6)

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

  def test_var_names(self):
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
      config = hparams_config.get_efficientdet_config('efficientdet-d0')
      inputs_shape = [1, 512, 512, 3]
      inputs = dict()
      for i in range(config.min_level, config.max_level + 1):
        inputs[i] = tf.ones(shape=inputs_shape, name='input', dtype=tf.float32)

      if not keras:
        legacy_arch.build_class_and_box_outputs(inputs, config)
      else:
        efficientdet_arch_keras.build_class_and_box_outputs(inputs, config)
      return [n.name for n in tf.global_variables()]

  def test_graph_variables_name_compatibility(self):
    legacy_names = self.build_model(False)
    keras_names = self.build_model(True)

    self.assertEqual(legacy_names, keras_names)

  def test_output(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    inputs_shape = [1, 512, 512, 3]
    config.max_level = config.min_level + 1
    with tf.Session(graph=tf.Graph()) as sess:
      inputs = dict()
      for i in range(config.min_level, config.max_level + 1):
        inputs[i] = tf.ones(shape=inputs_shape, name='input', dtype=tf.float32)
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
      self.assertAllEqual(class_output1[i], class_output2[i])
      self.assertAllEqual(box_output1[i], box_output2[i])


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
