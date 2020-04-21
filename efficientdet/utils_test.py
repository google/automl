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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


if __name__ == '__main__':
  tf.test.main()
