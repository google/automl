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
from absl import logging
from absl.testing import parameterized
import tensorflow as tf

import utils
from keras import util_keras


class KerasUtilTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('train_local', True, ''), ('eval_local', False, ''),
      ('train_tpu', True, 'tpu'), ('eval_tpu', False, 'tpu'))
  def test_batch_norm(self, is_training, strategy):
    inputs = tf.random.uniform([8, 40, 40, 3])
    expect_results = utils.batch_norm_act(
        inputs, is_training, None, strategy=strategy)

    # Call batch norm layer with is_training parameter.
    bn_layer = util_keras.build_batch_norm(is_training, strategy=strategy)
    self.assertAllClose(expect_results, bn_layer(inputs, is_training))


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
