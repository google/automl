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
"""Tests for utils."""
import os
from absl import logging
import tensorflow.compat.v1 as tf

import utils


class UtilsTest(tf.test.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    self.model_dir = os.path.join(tf.test.get_temp_dir(), 'model_dir')

  def build_model(self):
    x = tf.Variable(1.0)
    y = tf.Variable(2.0)
    z = x + y
    return z

  def test_archive_ckpt(self):
    model_dir = os.path.join(tf.test.get_temp_dir(), 'model_dir')
    ckpt_path = os.path.join(model_dir, 'ckpt')
    self.build_model()
    saver = tf.train.Saver()
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.save(sess, ckpt_path)

    # Save checkpoint if the new objective is better.
    self.assertTrue(utils.archive_ckpt('eval1', 0.1, ckpt_path))
    logging.info(os.listdir(model_dir))
    self.assertTrue(tf.io.gfile.exists(os.path.join(model_dir, 'archive')))
    self.assertFalse(tf.io.gfile.exists(os.path.join(model_dir, 'backup')))

    # Save checkpoint if the new objective is better.
    self.assertTrue(utils.archive_ckpt('eval2', 0.2, ckpt_path))
    self.assertTrue(tf.io.gfile.exists(os.path.join(model_dir, 'archive')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(model_dir, 'backup')))

    # Skip checkpoint if the new objective is worse.
    self.assertFalse(utils.archive_ckpt('eval3', 0.1, ckpt_path))

    # Save checkpoint if the new objective is better.
    self.assertTrue(utils.archive_ckpt('eval4', 0.3, ckpt_path))

    # Save checkpoint if the new objective is equal.
    self.assertTrue(utils.archive_ckpt('eval5', 0.3, ckpt_path))
    self.assertTrue(tf.io.gfile.exists(os.path.join(model_dir, 'archive')))
    self.assertTrue(tf.io.gfile.exists(os.path.join(model_dir, 'backup')))

  def test_image_size(self):
    self.assertEqual(utils.parse_image_size('1280x640'), (640, 1280))
    self.assertEqual(utils.parse_image_size(1280), (1280, 1280))
    self.assertEqual(utils.parse_image_size((1280, 640)), (1280, 640))

  def test_get_feat_sizes(self):
    feats = utils.get_feat_sizes(640, 2)
    self.assertEqual(feats, [{
        'height': 640,
        'width': 640
    }, {
        'height': 320,
        'width': 320
    }, {
        'height': 160,
        'width': 160
    }])

    feats = utils.get_feat_sizes((640, 300), 2)
    self.assertEqual(feats, [{
        'height': 640,
        'width': 300,
    }, {
        'height': 320,
        'width': 150,
    }, {
        'height': 160,
        'width': 75,
    }])

  def test_precision_float16(self):
    def _model(inputs):
      x = tf.ones((4, 4, 4, 4), dtype='float32')
      conv = tf.keras.layers.Conv2D(filters=4, kernel_size=2, use_bias=False)
      a = tf.Variable(1.0)
      return tf.cast(a, inputs.dtype) * conv(x) * inputs

    x = tf.constant(2.0, dtype=tf.float32)  # input can be any type.
    out = utils.build_model_with_precision('mixed_float16', _model, x)
    # Variables should be float32.
    for v in tf.global_variables():
      self.assertIn(v.dtype, (tf.float32, tf.dtypes.as_dtype('float32_ref')))
    self.assertIs(out.dtype, tf.float16)  # output should be float16.


class ActivationTest(tf.test.TestCase):

  def test_swish(self):
    features = tf.constant([.5, 10])

    result = utils.activation_fn(features, 'swish')
    expected = features * tf.sigmoid(features)
    self.assertAllClose(result, expected)

    result = utils.activation_fn(features, 'swish_native')
    self.assertAllClose(result, expected)

  def test_hswish(self):
    features = tf.constant([.5, 10])
    result = utils.activation_fn(features, 'hswish')
    self.assertAllClose(result, [0.29166667, 10.0])

  def test_relu(self):
    features = tf.constant([.5, 10])
    result = utils.activation_fn(features, 'relu')
    self.assertAllClose(result, [0.5, 10])

  def test_relu6(self):
    features = tf.constant([.5, 10])
    result = utils.activation_fn(features, 'relu6')
    self.assertAllClose(result, [0.5, 6])

  def test_mish(self):
    features = tf.constant([.5, 10])
    result = utils.activation_fn(features, 'mish')
    self.assertAllClose(result, [0.37524524, 10.0])


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.disable_eager_execution()
  tf.test.main()
