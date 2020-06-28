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
"""Tests for efficientdet_keras."""
import os
import tempfile
from absl import logging
import tensorflow.compat.v1 as tf

import efficientdet_arch as legacy_arch
import hparams_config
from keras import efficientdet_keras

SEED = 111111


class EfficientDetKerasTest(tf.test.TestCase):

  def test_backbone(self):
    inputs_shape = [1, 512, 512, 3]
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      feats = efficientdet_keras.build_backbone(feats, config)
      sess.run(tf.global_variables_initializer())
      keras_feats = sess.run(feats)
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      feats = legacy_arch.build_backbone(feats, config)
      sess.run(tf.global_variables_initializer())
      legacy_feats = sess.run(feats)
    for i, feat in enumerate(keras_feats):
      level = i + config.min_level
      self.assertAllClose(feat, legacy_feats[level])

  def test_model_output(self):
    inputs_shape = [1, 512, 512, 3]
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    tmp_ckpt = os.path.join(tempfile.mkdtemp(), 'ckpt')
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      model = efficientdet_keras.EfficientDetNet(config=config)
      outputs = model(feats)
      sess.run(tf.global_variables_initializer())
      keras_class_out, keras_box_out = sess.run(outputs)
      model.save_weights(tmp_ckpt)
    with tf.Session(graph=tf.Graph()) as sess:
      feats = tf.ones(inputs_shape)
      tf.random.set_random_seed(SEED)
      feats = legacy_arch.efficientdet(feats, config=config)
      sess.run(tf.global_variables_initializer())
      legacy_class_out, legacy_box_out = sess.run(feats)
    for i in range(3, 8):
      self.assertAllClose(
          keras_class_out[i - 3], legacy_class_out[i], rtol=1e-4, atol=1e-4)
      self.assertAllClose(
          keras_box_out[i - 3], legacy_box_out[i], rtol=1e-4, atol=1e-4)

    feats = tf.ones(inputs_shape)
    model = efficientdet_keras.EfficientDetNet(config=config)
    model.load_weights(tmp_ckpt)
    eager_class_out, eager_box_out = model(feats)
    for i in range(3, 8):
      self.assertAllClose(
        eager_class_out[i - 3], legacy_class_out[i], rtol=1e-4, atol=1e-4)
      self.assertAllClose(
        eager_box_out[i - 3], legacy_box_out[i], rtol=1e-4, atol=1e-4)

  def test_build_feature_network(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    with tf.Session(graph=tf.Graph()) as sess:
      inputs = [
          tf.ones([1, 64, 64, 40]),  # level 3
          tf.ones([1, 32, 32, 112]),  # level 4
          tf.ones([1, 16, 16, 320]),  # level 5
      ]
      tf.random.set_random_seed(SEED)
      new_feats1 = efficientdet_keras.build_feature_network(inputs, config)
      sess.run(tf.global_variables_initializer())
      keras_feats = sess.run(new_feats1)
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
      legacy_feats = sess.run(new_feats2)

    for i in range(config.min_level, config.max_level + 1):
      self.assertAllClose(keras_feats[i - config.min_level], legacy_feats[i])

  def test_model_variables(self):
    input_shape = (1, 512, 512, 3)
    model = efficientdet_keras.EfficientDetNet('efficientdet-d0')
    model.build(input_shape)
    eager_train_vars = sorted([var.name for var in model.trainable_variables])
    eager_model_vars = sorted([var.name for var in model.variables])
    with tf.Graph().as_default():
      feats = tf.ones([1, 512, 512, 3])
      model = efficientdet_keras.EfficientDetNet('efficientdet-d0')
      model.build(input_shape)
      keras_train_vars = sorted([var.name for var in model.trainable_variables])
      keras_model_vars = sorted([var.name for var in model.variables])
    with tf.Graph().as_default():
      feats = tf.ones([1, 512, 512, 3])
      legacy_arch.efficientdet(feats, 'efficientdet-d0')
      legacy_train_vars = sorted([var.name for var in tf.trainable_variables()])
      legacy_model_vars = sorted([var.name for var in tf.global_variables()])

    self.assertEqual(keras_train_vars, legacy_train_vars)
    self.assertEqual(keras_model_vars, legacy_model_vars)
    self.assertEqual(eager_train_vars, legacy_train_vars)
    self.assertEqual(eager_model_vars, legacy_model_vars)

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
            resample_layer = efficientdet_keras.ResampleFeatureMap(
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
      resample_layer = efficientdet_keras.ResampleFeatureMap(
          name='resample_p0',
          target_height=8,
          target_width=8,
          target_num_channels=64)
      resample_layer(feat)
      vars1 = sorted([var.name for var in tf.trainable_variables()])

    with tf.Graph().as_default():
      feat = tf.random.uniform([1, 16, 16, 320])
      legacy_arch.resample_feature_map(feat,
                                       name='p0',
                                       target_height=8,
                                       target_width=8,
                                       target_num_channels=64)
      vars2 = sorted([var.name for var in tf.trainable_variables()])

    self.assertEqual(vars1, vars2)


class EfficientDetVariablesNamesTest(tf.test.TestCase):

  def build_model(self, keras=False):
    with tf.Graph().as_default():
      config = hparams_config.get_efficientdet_config('efficientdet-d0')
      inputs_shape = [1, 512, 512, 3]
      legacy_inputs, keras_inputs = dict(), []
      for i in range(config.min_level, config.max_level + 1):
        keras_inputs.append(
            tf.ones(shape=inputs_shape, name='input', dtype=tf.float32))
        legacy_inputs[i] = keras_inputs[-1]

      if keras:
        efficientdet_keras.build_class_and_box_outputs(
            keras_inputs, config)
      else:
        legacy_arch.build_class_and_box_outputs(legacy_inputs, config)
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
      tf.random.set_random_seed(SEED)
      keras_inputs = []
      for i in range(config.min_level, config.max_level + 1):
        keras_inputs.append(
            tf.ones(shape=inputs_shape, name='input', dtype=tf.float32))

      output1 = efficientdet_keras.build_class_and_box_outputs(
          keras_inputs, config)
      sess.run(tf.global_variables_initializer())
      keras_class, keras_box = sess.run(output1)
    with tf.Session(graph=tf.Graph()) as sess:
      tf.random.set_random_seed(SEED)
      legacy_inputs = dict()
      for i in range(config.min_level, config.max_level + 1):
        legacy_inputs[i] = tf.ones(shape=inputs_shape,
                                   name='input',
                                   dtype=tf.float32)
      output2 = legacy_arch.build_class_and_box_outputs(legacy_inputs, config)
      sess.run(tf.global_variables_initializer())
      legacy_class, legacy_box = sess.run(output2)

    for i in range(config.min_level, config.max_level + 1):
      self.assertAllClose(keras_class[i - config.min_level], legacy_class[i])
      self.assertAllClose(keras_box[i - config.min_level], legacy_box[i])


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
