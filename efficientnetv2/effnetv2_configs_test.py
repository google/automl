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
"""Tests fo effnetv2_configs."""
from absl import logging
import tensorflow as tf
import effnetv2_configs


class EffnetV2ConfigsTest(tf.test.TestCase):

  def test_model_config(self):
    cfg = effnetv2_configs.get_model_config('efficientnet-b0')
    self.assertEqual(cfg.model.model_name, 'efficientnet-b0')

    cfg = effnetv2_configs.get_model_config('efficientnetv2-s')
    self.assertEqual(cfg.model.model_name, 'efficientnetv2-s')


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()

