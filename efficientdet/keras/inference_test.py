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
import tensorflow as tf
from keras import efficientdet_keras
from keras import inference


class InferenceTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    model = efficientdet_keras.EfficientDetModel('efficientdet-d0')
    self.tmp_path = tempfile.mkdtemp()
    model.save_weights(os.path.join(self.tmp_path, 'model'))

  def test_export(self):
    driver = inference.ServingDriver('efficientdet-d0', self.tmp_path)
    driver.export(self.tmp_path)
    has_saved_model = tf.saved_model.contains_saved_model(self.tmp_path)
    self.assertAllEqual(has_saved_model, True)

  def test_inference(self):
    driver = inference.ServingDriver('efficientdet-d0', self.tmp_path)
    images = tf.ones((1, 512, 512, 3))
    boxes, scores, classes, valid_lens = driver.serve(images)
    self.assertEqual(boxes.shape, (1, 100, 4))
    self.assertEqual(scores.shape, (1, 100))
    self.assertEqual(classes.shape, (1, 100))
    self.assertEqual(valid_lens.shape, (1,))
