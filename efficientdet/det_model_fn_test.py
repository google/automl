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
"""Tests for det_model_fn."""
import tensorflow as tf

import det_model_fn


class FocalLossTest(tf.test.TestCase):

    def test_focal_loss(self):
        y_pred = tf.random.uniform([4, 32, 32, 90])
        y_true = tf.ones([4, 32, 32, 90])
        alpha, gamma, n = 0.25, 1.5, 100
        legacy_output = det_model_fn.legacy_focal_loss(y_pred, y_true, alpha,
                                                       gamma, n)
        new_output = det_model_fn.focal_loss(y_pred, y_true, alpha, gamma, n)
        self.assertAllClose(legacy_output, new_output)

    def test_focal_loss_with_label_smoothing(self):
        y_pred = tf.random.uniform([4, 32, 32, 2])

        # A binary classification target [0.0, 1.0] becomes [.1, .9]
        #  with smoothing .2
        y_true = tf.ones([4, 32, 32, 2]) * [0.0, 1.0]
        y_true_presmoothed = tf.ones([4, 32, 32, 2]) * [0.1, 0.9]

        alpha, gamma, n = 0.25, 1.5, 100

        presmoothed = det_model_fn.focal_loss(y_pred, y_true_presmoothed, alpha,
                                              gamma, n, 0)
        unsmoothed = det_model_fn.focal_loss(y_pred, y_true, alpha, gamma, n,
                                             0.2)

        self.assertAllClose(presmoothed, unsmoothed)


if __name__ == '__main__':
    tf.test.main()
