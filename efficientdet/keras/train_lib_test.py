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

from absl.testing import parameterized
from absl import logging

import tensorflow as tf

import hparams_config
from keras.train_lib import StepwiseLrSchedule, CosineLrSchedule, PolynomialLrSchedule, BoxLoss, BoxIouLoss, FocalLoss, EfficientDetNetTrain, get_optimizer
from det_model_fn import stepwise_lr_schedule, cosine_lr_schedule, polynomial_lr_schedule, _box_loss


class TrainLibTest(tf.test.TestCase, parameterized.TestCase):

  def test_lr_schedule(self):
    stepwise = StepwiseLrSchedule(1e-3, 1e-4, 1, 3, 5)
    cosine = CosineLrSchedule(1e-3, 1e-4, 1, 5)
    polynomial = PolynomialLrSchedule(1e-3, 1e-4, 1, 2, 5)
    for i in range(5):
      self.assertEqual(stepwise_lr_schedule(1e-3, 1e-4, 1, 3, 5, i),
                       stepwise(i))
      self.assertEqual(cosine_lr_schedule(1e-3, 1e-4, 1, 5, i), cosine(i))
      self.assertEqual(polynomial_lr_schedule(1e-3, 1e-4, 1, 2, 5, i),
                       polynomial(i))

  def test_losses(self):
    box_loss = BoxLoss()
    box_outputs = tf.ones([10])
    box_targets = tf.zeros([10])
    num_positives = 4.0
    self.assertEqual(_box_loss(box_outputs, box_targets, num_positives),
                     box_loss([num_positives, box_targets], box_outputs))

  def test_train(self):
    config = hparams_config.get_detection_config('efficientdet-d0')
    config.batch_size = 1
    config.num_examples_per_epoch = 1
    x = tf.ones((1, 512, 512, 3))
    labels = {
        'box_targets_%d' % i: tf.ones((1, 512 // 2**i, 512 // 2**i, 36))
        for i in range(3, 8)
    }
    labels.update({
        'cls_targets_%d' % i: tf.ones((1, 512 // 2**i, 512 // 2**i, 9),
                                      dtype=tf.int32) for i in range(3, 8)
    })
    labels.update({'mean_num_positives': 100.0})

    model = EfficientDetNetTrain(config=config)
    params = config.as_dict()
    model.compile(optimizer=get_optimizer(params),
                  loss={
                      "box_loss":
                          BoxLoss(params['delta'],
                                  reduction=tf.keras.losses.Reduction.NONE),
                      "box_iou_loss":
                          BoxIouLoss(params['iou_loss_type'],
                                     reduction=tf.keras.losses.Reduction.NONE),
                      "class_loss":
                          FocalLoss(params['alpha'],
                                    params['gamma'],
                                    label_smoothing=params['label_smoothing'],
                                    reduction=tf.keras.losses.Reduction.NONE)
                  })
    outputs = model.train_on_batch(x, labels)
    outputs = model.test_on_batch(x, labels)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
