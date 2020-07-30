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
"""Tests for fpn_configs."""
from absl import logging
import tensorflow as tf
from keras import fpn_configs


class FpnConfigTest(tf.test.TestCase):

  def test_bifpn_l3l7(self):
    p1 = fpn_configs.bifpn_config(3, 7, None)
    # pyformat: disable
    self.assertEqual(
        p1.nodes,
        [
            {'feat_level': 6, 'inputs_offsets': [3, 4]},
            {'feat_level': 5, 'inputs_offsets': [2, 5]},
            {'feat_level': 4, 'inputs_offsets': [1, 6]},
            {'feat_level': 3, 'inputs_offsets': [0, 7]},
            {'feat_level': 4, 'inputs_offsets': [1, 7, 8]},
            {'feat_level': 5, 'inputs_offsets': [2, 6, 9]},
            {'feat_level': 6, 'inputs_offsets': [3, 5, 10]},
            {'feat_level': 7, 'inputs_offsets': [4, 11]},
        ])
    # pyformat: enable

  def test_bifpn_l2l7(self):
    p = fpn_configs.bifpn_config(2, 7, None)

    # pyformat: disable
    self.assertEqual(
        p.nodes,
        [
            {'feat_level': 6, 'inputs_offsets': [4, 5]},
            {'feat_level': 5, 'inputs_offsets': [3, 6]},
            {'feat_level': 4, 'inputs_offsets': [2, 7]},
            {'feat_level': 3, 'inputs_offsets': [1, 8]},
            {'feat_level': 2, 'inputs_offsets': [0, 9]},
            {'feat_level': 3, 'inputs_offsets': [1, 9, 10]},
            {'feat_level': 4, 'inputs_offsets': [2, 8, 11]},
            {'feat_level': 5, 'inputs_offsets': [3, 7, 12]},
            {'feat_level': 6, 'inputs_offsets': [4, 6, 13]},
            {'feat_level': 7, 'inputs_offsets': [5, 14]},
        ])
    # pyformat: enable

  def test_qufpn_dynamic_l3l7(self):
    p = fpn_configs.qufpn_config(3, 7, None)

    # pyformat: disable
    # pylint: disable=line-too-long
    self.assertEqual(
        p.nodes,
        [
            {'feat_level': 6, 'inputs_offsets': [3, 4], 'weight_method': 'fastattn'},
            {'feat_level': 5, 'inputs_offsets': [2, 5], 'weight_method': 'fastattn'},
            {'feat_level': 4, 'inputs_offsets': [1, 6], 'weight_method': 'fastattn'},
            {'feat_level': 3, 'inputs_offsets': [0, 7], 'weight_method': 'fastattn'},
            {'feat_level': 4, 'inputs_offsets': [1, 7, 8], 'weight_method': 'fastattn'},
            {'feat_level': 5, 'inputs_offsets': [2, 6, 9], 'weight_method': 'fastattn'},
            {'feat_level': 6, 'inputs_offsets': [3, 5, 10], 'weight_method': 'fastattn'},
            {'feat_level': 7, 'inputs_offsets': [4, 11], 'weight_method': 'fastattn'},
            {'feat_level': 4, 'inputs_offsets': [1, 0], 'weight_method': 'fastattn'},
            {'feat_level': 5, 'inputs_offsets': [2, 13], 'weight_method': 'fastattn'},
            {'feat_level': 6, 'inputs_offsets': [3, 14], 'weight_method': 'fastattn'},
            {'feat_level': 7, 'inputs_offsets': [4, 15], 'weight_method': 'fastattn'},
            {'feat_level': 6, 'inputs_offsets': [3, 15, 16], 'weight_method': 'fastattn'},
            {'feat_level': 5, 'inputs_offsets': [2, 14, 17], 'weight_method': 'fastattn'},
            {'feat_level': 4, 'inputs_offsets': [1, 13, 18], 'weight_method': 'fastattn'},
            {'feat_level': 3, 'inputs_offsets': [0, 19], 'weight_method': 'fastattn'},
            {'feat_level': 7, 'inputs_offsets': [12, 16], 'weight_method': 'fastattn'},
            {'feat_level': 6, 'inputs_offsets': [11, 17], 'weight_method': 'fastattn'},
            {'feat_level': 5, 'inputs_offsets': [10, 18], 'weight_method': 'fastattn'},
            {'feat_level': 4, 'inputs_offsets': [9, 19], 'weight_method': 'fastattn'},
            {'feat_level': 3, 'inputs_offsets': [8, 20], 'weight_method': 'fastattn'},
        ])
    # pylint: enable=line-too-long
    # pyformat: enable


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
