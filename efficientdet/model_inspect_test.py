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
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import model_inspect
import utils
FLAGS = flags.FLAGS


class ModelInspectTest(tf.test.TestCase):
  """Model inspect tests."""

  def setUp(self):
    super(ModelInspectTest, self).setUp()
    sys_tempdir = tempfile.gettempdir()
    self.tempdir = os.path.join(sys_tempdir, '_inspect_test')
    os.mkdir(self.tempdir)

    np.random.seed(111)
    tf.random.set_random_seed(111)
    self.test_image = np.random.randint(0, 244, (640, 720, 3)).astype(np.uint8)

    self.savedmodel_dir = os.path.join(self.tempdir, 'savedmodel')
    if os.path.exists(self.savedmodel_dir):
      shutil.rmtree(self.savedmodel_dir)

    self.params = dict(
        model_name='efficientdet-d0',
        logdir=os.path.join(self.tempdir, 'logdir'),
        tensorrt=False,
        use_xla=False,
        ckpt_path='_',
        export_ckpt=None,
        saved_model_dir=self.savedmodel_dir,
        batch_size=1,
        hparams='')

  def tearDown(self):
    super(ModelInspectTest, self).tearDown()
    shutil.rmtree(self.tempdir)

  def test_dry_run(self):
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('dry')

  def test_freeze_model(self):
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('freeze')

  def test_bm(self):
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('bm')

  def test_eval_ckpt(self):
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('ckpt')

  def test_infer(self):
    outdir = os.path.join(self.tempdir, 'infer_imgout')
    os.mkdir(outdir)
    inspector = model_inspect.ModelInspector(**self.params)

    img_path = os.path.join(self.tempdir, 'img.jpg')
    Image.fromarray(self.test_image).save(img_path)

    self.assertFalse(os.path.exists(os.path.join(outdir, '0.jpg')))
    inspector.run_model('infer', input_image=img_path, output_image_dir=outdir)
    self.assertTrue(os.path.exists(os.path.join(outdir, '0.jpg')))

    out = np.sum(np.array(Image.open(os.path.join(outdir, '0.jpg'))))
    self.assertEqual(out // 10000000, 16)

  def test_saved_model(self):
    if tf.__version__ >= '2.3.0-dev20200521':
      self.params['tflite_path'] = os.path.join(self.savedmodel_dir, 'x.tflite')
    inspector = model_inspect.ModelInspector(**self.params)
    self.assertFalse(
        os.path.exists(os.path.join(self.savedmodel_dir, 'saved_model.pb')))
    inspector.run_model('saved_model')
    self.assertTrue(
        os.path.exists(os.path.join(self.savedmodel_dir, 'saved_model.pb')))
    self.assertTrue(
        os.path.exists(
            os.path.join(self.savedmodel_dir, 'efficientdet-d0_frozen.pb')))
    if self.params.get('tflite_path', None):
      self.assertTrue(
          os.path.exists(os.path.join(self.savedmodel_dir, 'x.tflite')))

  def test_saved_model_fp16(self):
    self.params['hparams'] = 'mixed_precision=true'
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('saved_model')
    self.assertTrue(
        os.path.exists(os.path.join(self.savedmodel_dir, 'saved_model.pb')))
    utils.set_precision_policy('float32')

  def test_saved_model_infer(self):
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('saved_model')

    outdir = os.path.join(self.tempdir, 'infer_imgout')
    os.mkdir(outdir)

    tf.reset_default_graph()
    self.assertFalse(os.path.exists(os.path.join(outdir, '0.jpg')))

    img_path = os.path.join(self.tempdir, 'img.jpg')
    Image.fromarray(self.test_image).save(img_path)
    inspector.run_model(
        'saved_model_infer', input_image=img_path, output_image_dir=outdir)
    self.assertTrue(os.path.exists(os.path.join(outdir, '0.jpg')))

    out = np.sum(np.array(Image.open(os.path.join(outdir, '0.jpg'))))
    self.assertEqual(out // 10000000, 16)

  def test_saved_model_infer_dynamic_batch(self):
    # Build saved model with dynamic batch size.
    self.params['batch_size'] = None
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('saved_model')

    outdir = os.path.join(self.tempdir, 'infer_imgout_dyn')
    os.mkdir(outdir)

    img_path = os.path.join(self.tempdir, 'img.jpg')
    Image.fromarray(self.test_image).save(img_path)
    test_image2 = np.random.randint(0, 244, (640, 320, 3)).astype(np.uint8)
    img2_path = os.path.join(self.tempdir, 'img2.jpg')
    Image.fromarray(test_image2).save(img2_path)

    # serve images with batch size 1.
    tf.reset_default_graph()
    self.params['batch_size'] = 1
    self.assertFalse(os.path.exists(os.path.join(outdir, '0.jpg')))
    inspector.run_model(
        'saved_model_infer', input_image=img_path, output_image_dir=outdir)
    self.assertTrue(os.path.exists(os.path.join(outdir, '0.jpg')))

    # serve images with batch size 2.
    tf.reset_default_graph()
    self.params['batch_size'] = 2
    self.assertFalse(os.path.exists(os.path.join(outdir, '1.jpg')))
    fname = img_path.replace('img.jpg', 'img*.jpg')
    inspector.run_model(
        'saved_model_infer', input_image=fname, output_image_dir=outdir)
    self.assertTrue(os.path.exists(os.path.join(outdir, '1.jpg')))

  def test_saved_model_graph_infer(self):
    inspector = model_inspect.ModelInspector(**self.params)
    inspector.run_model('saved_model')
    tf.reset_default_graph()

    # Use the frozen graph to do inference.
    inspector.saved_model_dir = os.path.join(self.params['saved_model_dir'],
                                             'efficientdet-d0_frozen.pb')
    outdir = os.path.join(self.tempdir, 'pb_infer_imgout')
    os.mkdir(outdir)

    img_path = os.path.join(self.tempdir, 'img.jpg')
    Image.fromarray(self.test_image).save(img_path)

    self.assertFalse(os.path.exists(os.path.join(outdir, '0.jpg')))
    inspector.run_model(
        'saved_model_infer', input_image=img_path, output_image_dir=outdir)
    self.assertTrue(os.path.exists(os.path.join(outdir, '0.jpg')))

    out = np.sum(np.array(Image.open(os.path.join(outdir, '0.jpg'))))
    self.assertEqual(out // 10000000, 16)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.disable_eager_execution()
  tf.test.main()
