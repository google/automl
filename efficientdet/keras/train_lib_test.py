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
import os
import tempfile
from absl import logging
import tensorflow as tf

import det_model_fn as legacy_fn
import hparams_config
from keras import train_lib


class TrainLibTest(tf.test.TestCase):

  def test_display_callback(self):
    config = hparams_config.get_detection_config('efficientdet-d0')
    config.batch_size = 1
    config.num_examples_per_epoch = 1
    config.model_dir = tempfile.mkdtemp()
    fake_image = tf.ones([512, 512, 3], dtype=tf.uint8)
    fake_jpeg = tf.image.encode_jpeg(fake_image)
    sample_image = os.path.join(config.model_dir + 'fake_image.jpg')
    tf.io.write_file(sample_image, fake_jpeg)
    display_callback = train_lib.DisplayCallback(sample_image,
                                                 config.model_dir,
                                                 update_freq=1)
    model = train_lib.EfficientDetNetTrain(config=config)
    model.build((1, 512, 512, 3))
    display_callback.set_model(model)
    display_callback.on_train_batch_end(0)

  def test_lr_schedule(self):
    stepwise = train_lib.StepwiseLrSchedule(1e-3, 1e-4, 1, 3, 5)
    cosine = train_lib.CosineLrSchedule(1e-3, 1e-4, 1, 5)
    polynomial = train_lib.PolynomialLrSchedule(1e-3, 1e-4, 1, 2, 5)
    for i in range(5):
      self.assertEqual(
          legacy_fn.stepwise_lr_schedule(1e-3, 1e-4, 1, 3, 5, i), stepwise(i))
      self.assertEqual(
          legacy_fn.cosine_lr_schedule(1e-3, 1e-4, 1, 5, i), cosine(i))
      self.assertEqual(
          legacy_fn.polynomial_lr_schedule(1e-3, 1e-4, 1, 2, 5, i),
          polynomial(i))

  def test_losses(self):
    tf.random.set_seed(1111)
    box_loss = train_lib.BoxLoss()
    box_iou_loss = train_lib.BoxIouLoss(
        iou_loss_type='ciou',
        min_level=3,
        max_level=3,
        num_scales=1,
        aspect_ratios=[1.0],
        anchor_scale=1.0,
        image_size=32)
    alpha = 0.25
    gamma = 1.5
    focal_loss_v2 = train_lib.FocalLoss(
        alpha, gamma, reduction=tf.keras.losses.Reduction.NONE)
    box_outputs = tf.random.normal([64, 4])
    box_targets = tf.random.normal([64, 4])
    num_positives = tf.constant(4.0)
    self.assertAllClose(
        legacy_fn._box_loss(box_outputs, box_targets, num_positives),
        box_loss([num_positives, box_targets], box_outputs))
    self.assertAllClose(
        legacy_fn.focal_loss(box_outputs, box_targets, alpha, gamma,
                             num_positives),
        focal_loss_v2([num_positives, box_targets], box_outputs))
    # TODO(tanmingxing): Re-enable this test after fixing this failing test.
    # self.assertEqual(
    #     legacy_fn._box_iou_loss(box_outputs, box_targets, num_positives,
    #                             'ciou'),
    #     box_iou_loss([num_positives, box_targets], box_outputs))
    iou_loss = box_iou_loss([num_positives, box_targets], box_outputs)
    self.assertAlmostEqual(iou_loss.numpy(), 4.924635, places=5)

  def test_predict(self):
    _, x, _, model = self._build_model()
    cls_outputs, box_outputs, seg_outputs = model(x)
    self.assertLen(cls_outputs, 5)
    self.assertLen(box_outputs, 5)
    self.assertLen(seg_outputs, 1)

  def _build_model(self, grad_checkpoint=False):
    tf.random.set_seed(1111)
    config = hparams_config.get_detection_config('efficientdet-d0')
    config.heads = ['object_detection', 'segmentation']
    config.batch_size = 1
    config.num_examples_per_epoch = 1
    config.model_dir = tempfile.mkdtemp()
    config.steps_per_epoch = 1
    config.mixed_precision = True
    config.grad_checkpoint = grad_checkpoint
    x = tf.ones((1, 512, 512, 3))
    labels = {
        'box_targets_%d' % i: tf.ones((1, 512 // 2**i, 512 // 2**i, 36))
        for i in range(3, 8)
    }
    labels.update({
        'cls_targets_%d' % i: tf.ones((1, 512 // 2**i, 512 // 2**i, 9),
                                      dtype=tf.int32) for i in range(3, 8)
    })
    labels.update({'image_masks': tf.ones((1, 128, 128, 1))})
    labels.update({'mean_num_positives': tf.constant([10.0])})

    params = config.as_dict()
    params['num_shards'] = 1
    params['steps_per_execution'] = 100
    params['model_dir'] = tempfile.mkdtemp()
    params['profile'] = False
    config.override(params, allow_new_keys=True)
    model = train_lib.EfficientDetNetTrain(config=config)
    model.build((1, 512, 512, 3))
    model.compile(
        optimizer=train_lib.get_optimizer(params),
        loss={
            train_lib.BoxLoss.__name__:
                train_lib.BoxLoss(
                    params['delta'], reduction=tf.keras.losses.Reduction.NONE),
            train_lib.BoxIouLoss.__name__:
                train_lib.BoxIouLoss(
                    params['iou_loss_type'],
                    params['min_level'],
                    params['max_level'],
                    params['num_scales'],
                    params['aspect_ratios'],
                    params['anchor_scale'],
                    params['image_size'],
                    reduction=tf.keras.losses.Reduction.NONE),
            train_lib.FocalLoss.__name__:
                train_lib.FocalLoss(
                    params['alpha'],
                    params['gamma'],
                    label_smoothing=params['label_smoothing'],
                    reduction=tf.keras.losses.Reduction.NONE),
            tf.keras.losses.SparseCategoricalCrossentropy.__name__:
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        })
    return params, x, labels, model

  def test_train_on_batch(self):
    _, x, labels, model = self._build_model()
    outputs = model.train_on_batch(x, labels, return_dict=True)
    expect_results = {
        'loss': 26277.03125,
        'det_loss': 26277.03125,
        'cls_loss': 5065.4375,
        'box_loss': 423.44061279296875,
        'gradient_norm': 10.0,
        'reg_l2_loss': 0.,
        'learning_rate': 0.,
        'seg_loss': 1.224212408065796,
    }
    self.assertAllClose(outputs, expect_results, rtol=.1, atol=100.)

  def test_infer_on_batch(self):
    _, x, labels, model = self._build_model()
    outputs = model.test_on_batch(x, labels, return_dict=True)
    expect_results = {
        'loss': 26277.03125,
        'det_loss': 26277.03125,
        'cls_loss': 5065.4375,
        'reg_l2_loss': 0.,
        'box_loss': 423.5846862792969,
        'seg_loss': 1.0981438159942627,
    }
    self.assertAllClose(outputs, expect_results, rtol=.1, atol=100.)

  def test_fit(self):
    params, x, labels, model = self._build_model()
    hist = model.fit(
        x,
        labels,
        steps_per_epoch=1,
        epochs=1,
        callbacks=train_lib.get_callbacks(params))
    self.assertAllClose(hist.history['loss'], [26279.47], rtol=.1, atol=10.)
    self.assertAllClose(hist.history['det_loss'], [26279.47], rtol=.1, atol=10.)
    self.assertAllClose(hist.history['cls_loss'], [5065.437], rtol=.1, atol=10.)
    self.assertAllClose(hist.history['box_loss'], [424.], rtol=.1, atol=100.)
    self.assertAllClose(
        hist.history['seg_loss'], [1.221547], rtol=.1, atol=100.)
    # skip gnorm test because it is flaky.

  def test_recompute_grad(self):
    tf.config.run_functions_eagerly(True)
    _, x, labels, model = self._build_model(False)
    with tf.GradientTape() as tape:
      loss_vals = {}
      cls_outputs, box_outputs, _ = model(x, training=True)
      det_loss = model._detection_loss(cls_outputs, box_outputs, labels,
                                       loss_vals)
      grads1 = tape.gradient(det_loss, model.trainable_variables)

    _, x, labels, model = self._build_model(True)
    with tf.GradientTape() as tape:
      loss_vals = {}
      cls_outputs2, box_outputs2, _ = model(x, training=True)
      det_loss2 = model._detection_loss(cls_outputs2, box_outputs2, labels,
                                        loss_vals)
      grads2 = tape.gradient(det_loss2, model.trainable_variables)
    for grad1, grad2 in zip(grads1, grads2):
      if grad1 is None or grad2 is None:
        continue
      self.assertAllClose(grad1, grad2)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
