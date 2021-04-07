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
from absl.testing import flagsaver
import numpy as np
from PIL import Image
import tensorflow as tf

from keras import inspector
FLAGS = flags.FLAGS


class InspectorTest(tf.test.TestCase):
  """Model inspect tests."""

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()
    FLAGS.model_dir = '_'

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  @flagsaver.flagsaver(mode='dry')
  def test_dry(self):
    FLAGS.export_ckpt = os.path.join(self.tempdir, 'model')
    inspector.main(None)
    self.assertIsNot(tf.train.get_checkpoint_state(self.tempdir), None)

  @flagsaver.flagsaver(mode='infer', saved_model_dir=None)
  def test_infer(self):
    test_image = np.random.randint(0, 244, (640, 720, 3)).astype(np.uint8)
    FLAGS.input_image = os.path.join(self.tempdir, 'img.jpg')
    Image.fromarray(test_image).save(FLAGS.input_image)
    FLAGS.output_image_dir = self.tempdir
    inspector.main(None)
    self.assertTrue(tf.io.gfile.exists(os.path.join(self.tempdir, '0.jpg')))

  @flagsaver.flagsaver(mode='benchmark', saved_model_dir=None)
  def test_benchmark(self):
    inspector.main(None)
    self.assertFalse(tf.io.gfile.exists(os.path.join(self.tempdir, '0.jpg')))

  @flagsaver.flagsaver(mode='export', tflite='FP32')
  def test_export(self):
    FLAGS.saved_model_dir = os.path.join(self.tempdir, 'savedmodel')
    tflite_path = os.path.join(FLAGS.saved_model_dir, 'fp32.tflite')
    inspector.main(None)
    self.assertTrue(tf.saved_model.contains_saved_model(FLAGS.saved_model_dir))
    self.assertTrue(tf.io.gfile.exists(tflite_path))


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
