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
r"""Inference test cases."""
import os
import tempfile
from absl import logging
import tensorflow as tf
import test_util
from keras import efficientdet_keras
from keras import inference


class InferenceTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(111111)
    model = efficientdet_keras.EfficientDetModel('efficientdet-d0')
    self.tmp_path = tempfile.mkdtemp()
    model.save_weights(os.path.join(self.tmp_path, 'model'))

  def test_export(self):
    saved_model_path = os.path.join(self.tmp_path, 'saved_model')
    driver = inference.ServingDriver('efficientdet-d0', self.tmp_path)
    driver.export(saved_model_path)
    has_saved_model = tf.saved_model.contains_saved_model(saved_model_path)
    self.assertAllEqual(has_saved_model, True)
    driver.load(saved_model_path)
    driver.load(os.path.join(saved_model_path, 'efficientdet-d0_frozen.pb'))

  def test_export_tflite_only_network(self):
    saved_model_path = os.path.join(self.tmp_path, 'saved_model')
    driver = inference.ServingDriver(
        'efficientdet-lite0', self.tmp_path, only_network=True)
    driver.export(saved_model_path, tflite='FP32')
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(saved_model_path, 'fp32.tflite')))
    tf.io.gfile.rmtree(saved_model_path)
    driver.export(saved_model_path, tflite='FP16')
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(saved_model_path, 'fp16.tflite')))
    tf.io.gfile.rmtree(saved_model_path)
    tfrecord_path = test_util.make_fake_tfrecord(self.get_temp_dir())
    driver.export(
        saved_model_path,
        tflite='INT8',
        file_pattern=[tfrecord_path],
        num_calibration_steps=1)
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(saved_model_path, 'int8.tflite')))

  def test_export_tflite_with_post_processing(self):
    saved_model_path = os.path.join(self.tmp_path, 'saved_model')
    driver = inference.ServingDriver(
        'efficientdet-lite0', self.tmp_path, only_network=False)
    driver.export(saved_model_path, tflite='FP32')
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(saved_model_path, 'fp32.tflite')))
    tf.io.gfile.rmtree(saved_model_path)
    tfrecord_path = test_util.make_fake_tfrecord(self.get_temp_dir())
    driver.export(
        saved_model_path,
        tflite='INT8',
        file_pattern=[tfrecord_path],
        num_calibration_steps=1)
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(saved_model_path, 'int8.tflite')))

  def test_inference(self):
    driver = inference.ServingDriver('efficientdet-d0', self.tmp_path)
    images = tf.ones((1, 512, 512, 3))
    boxes, scores, classes, valid_lens = driver.serve(images)
    self.assertEqual(tf.reduce_mean(boxes), 163.09)
    self.assertEqual(tf.reduce_mean(scores), 0.01000005)
    self.assertEqual(tf.reduce_mean(classes), 1)
    self.assertEqual(tf.reduce_mean(valid_lens), 100)
    self.assertEqual(boxes.shape, (1, 100, 4))
    self.assertEqual(scores.shape, (1, 100))
    self.assertEqual(classes.shape, (1, 100))
    self.assertEqual(valid_lens.shape, (1,))

  def test_inference_without_ema(self):
    driver = inference.ServingDriver('efficientdet-d0', self.tmp_path)
    driver.build({'moving_average_decay': 0})
    images = tf.ones((1, 512, 512, 3))
    boxes, scores, classes, valid_lens = driver.serve(images)
    self.assertEqual(tf.reduce_mean(boxes), 163.09)
    self.assertEqual(tf.reduce_mean(scores), 0.01000005)
    self.assertEqual(tf.reduce_mean(classes), 1)
    self.assertEqual(tf.reduce_mean(valid_lens), 100)
    self.assertEqual(boxes.shape, (1, 100, 4))
    self.assertEqual(scores.shape, (1, 100))
    self.assertEqual(classes.shape, (1, 100))
    self.assertEqual(valid_lens.shape, (1,))

  def test_network_inference(self):
    driver = inference.ServingDriver(
        'efficientdet-d0', self.tmp_path, only_network=True)
    images = tf.ones((1, 512, 512, 3))
    class_outputs, box_outputs = driver.serve(images)
    self.assertLen(class_outputs, 5)
    self.assertLen(box_outputs, 5)

  def test_inference_mixed_precision(self):
    driver = inference.ServingDriver('efficientdet-d0', self.tmp_path)
    driver.build({'mixed_precision': True})
    images = tf.ones((1, 512, 512, 3))
    boxes, scores, classes, valid_lens = driver.serve(images)
    policy = tf.keras.mixed_precision.experimental.global_policy()
    if policy.name == 'float32':
      self.assertEqual(tf.reduce_mean(boxes), 163.09)
      self.assertEqual(tf.reduce_mean(scores), 0.01000005)
      self.assertEqual(tf.reduce_mean(classes), 1)
      self.assertEqual(tf.reduce_mean(valid_lens), 100)
    elif policy.name == 'float16':
      pass
    elif policy.name == 'bfloat16':
      pass
    self.assertEqual(boxes.shape, (1, 100, 4))
    self.assertEqual(scores.shape, (1, 100))
    self.assertEqual(classes.shape, (1, 100))
    self.assertEqual(valid_lens.shape, (1,))


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
