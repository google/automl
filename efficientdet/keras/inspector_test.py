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
import subprocess

from absl import logging
import tensorflow as tf


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
    result = subprocess.run(
        ['python3', 'keras/inspector.py', '--mode=dry', '--ckpt_path=_'])
    self.assertEqual(result.returncode, 0)

  def test_infer(self):
    result = subprocess.run([
        'python3', 'keras/inspector.py', '--mode=infer', '--ckpt_path=_',
        '--input_image={}'.format('testdata/img1.jpg'),
        '--output_image_dir={}'.format('testdata')
    ])
    self.assertEqual(result.returncode, 0)

  def test_benchmark(self):
    result = subprocess.run([
        'python3', 'keras/inspector.py', '--mode=benchmark', '--ckpt_path=_',
        '--input_image={}'.format('testdata/img1.jpg'),
        '--output_image_dir={}'.format('testdata')
    ])
    self.assertEqual(result.returncode, 0)

  def test_export(self):
    result = subprocess.run([
        'python3', 'keras/inspector.py', '--mode=export', '--ckpt_path=_',
        '--saved_model_dir={}'.format(self.savedmodel_dir)
    ])
    self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
