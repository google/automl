# Copyright 2021 Google Research. All Rights Reserved.
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

import tensorflow as tf
import utils


class UtilsTest(tf.test.TestCase):

  def test_constant_lr(self):
    constant_schedule = utils.WarmupLearningRateSchedule(
        1.0, lr_decay_type='constant', warmup_epochs=None)

    lr = constant_schedule(10)
    self.assertAllClose(lr, 1.0)

  def test_linear_lr(self):
    linear_schedule = utils.WarmupLearningRateSchedule(
        1.0, total_steps=10, lr_decay_type='linear', warmup_epochs=None)

    lr = linear_schedule(0)
    self.assertAllClose(lr, 1.0)

    lr = linear_schedule(5)
    self.assertAllClose(lr, 0.5)

    lr = linear_schedule(10)
    self.assertAllClose(lr, 0.0)

  def test_cosine_lr(self):
    cosine_schedule = utils.WarmupLearningRateSchedule(
        1.0, total_steps=10, lr_decay_type='cosine', warmup_epochs=None)

    lr = cosine_schedule(4)
    self.assertAllClose(lr, 0.654508)

    lr = cosine_schedule(5)
    self.assertAllClose(lr, 0.5)

    lr = cosine_schedule(6)
    self.assertAllClose(lr, 0.345491)

  def test_exponential_lr(self):
    exponential_schedule = utils.WarmupLearningRateSchedule(
        1.0,
        total_steps=100,
        steps_per_epoch=10,
        decay_epochs=2,
        decay_factor=0.5,
        lr_decay_type='exponential',
        warmup_epochs=None)

    lr = exponential_schedule(5)
    self.assertAllClose(lr, 1.0)

    lr = exponential_schedule(25)
    self.assertAllClose(lr, 0.5)

    lr = exponential_schedule(70)
    self.assertAllClose(lr, 0.125)

  def test_warmup(self):
    warmup_schedule = utils.WarmupLearningRateSchedule(
        1.0,
        total_steps=100,
        steps_per_epoch=10,
        warmup_epochs=2,
        lr_decay_type='constant')

    lr = warmup_schedule(5)
    self.assertAllClose(lr, 0.25)

    lr = warmup_schedule(35)
    self.assertAllClose(lr, 1.0)


if __name__ == '__main__':
  tf.test.main()
