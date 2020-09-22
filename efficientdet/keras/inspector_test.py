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
r"""Tests for model inspect tool."""
import os
import shutil
import tempfile

from absl import flags
from absl import logging
import tensorflow as tf
from keras import inspector

FLAGS = flags.FLAGS


class ModelInspectTest(tf.test.TestCase):
  """Model inspect tests."""

  def setUp(self):
    super().setUp()
    sys_tempdir = tempfile.mkdtemp()
    self.tempdir = os.path.join(sys_tempdir, '_inspect_test')
    os.mkdir(self.tempdir)

    self.savedmodel_dir = os.path.join(self.tempdir, 'savedmodel')
    if os.path.exists(self.savedmodel_dir):
      shutil.rmtree(self.savedmodel_dir)

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_dry(self):
    FLAGS.mode = 'dry'
    FLAGS.export_ckpt = os.path.join(self.tempdir, 'model')
    inspector.main(None)
    self.assertIsNot(tf.train.get_checkpoint_state(self.tempdir), None)

  def test_infer(self):
    FLAGS.mode = 'infer'
    FLAGS.input_image = 'testdata/img1.jpg'
    FLAGS.output_image_dir = self.tempdir
    inspector.main(None)
    self.assertTrue(tf.io.gfile.exists(os.path.join(self.tempdir, '0.jpg')))

  def test_benchmark(self):
    FLAGS.mode = 'benchmark'
    inspector.main(None)
    self.assertFalse(tf.io.gfile.exists(os.path.join(self.tempdir, '0.jpg')))

  def test_export(self):
    FLAGS.mode = 'export'
    FLAGS.saved_model_dir = self.savedmodel_dir
    inspector.main(None)
    self.assertTrue(tf.saved_model.contains_saved_model(self.savedmodel_dir))


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
