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
"""Tests for efficientdet_arch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import efficientdet_arch
import hparams_config
import utils


class EfficientDetArchTest(tf.test.TestCase):

  def build_model(self,
                  model_name,
                  isize,
                  is_training=False,
                  data_format='channels_last'):
    if isinstance(isize, int):
      isize = (isize, isize)
    if data_format == 'channels_first':
      inputs_shape = [1, 3, isize[0], isize[1]]
    else:
      inputs_shape = [1, isize[0], isize[1], 3]
    inputs = tf.placeholder(tf.float32, name='input', shape=inputs_shape)
    efficientdet_arch.efficientdet(
        inputs,
        model_name=model_name,
        is_training_bn=is_training,
        use_bfloat16=False,
        image_size=isize,
        data_format=data_format)
    return utils.num_params_flops(False)

  def test_efficientdet_d0(self):
    self.assertSequenceEqual((3880067, 2535978423),
                             self.build_model('efficientdet-d0', 512))

  def test_efficientdet_d0_channel_first(self):
    self.assertSequenceEqual(
        (3880067, 2534780343),
        self.build_model('efficientdet-d0', 512, data_format='channels_first'))

  def test_efficientdet_d0_511_513(self):
    self.assertSequenceEqual((3880067, 2613160475),
                             self.build_model('efficientdet-d0', (511, 513)))

  def test_efficientdet_d1(self):
    self.assertSequenceEqual((6625898, 6102772568),
                             self.build_model('efficientdet-d1', 640))

  def test_efficientdet_d1_640_1280(self):
    self.assertSequenceEqual((6625898, 12197893743),
                             self.build_model('efficientdet-d1', (640, 1280)))

  def test_efficientdet_d2(self):
    self.assertSequenceEqual((8097039, 10997043492),
                             self.build_model('efficientdet-d2', 768))

  def test_efficientdet_d3(self):
    self.assertSequenceEqual((12032296, 24890171439),
                             self.build_model('efficientdet-d3', 896))

  def test_efficientdet_d4(self):
    self.assertSequenceEqual((20723675, 55185040717),
                             self.build_model('efficientdet-d4', 1024))

  def test_efficientdet_d5(self):
    self.assertSequenceEqual((33653315, 135387474989),
                             self.build_model('efficientdet-d5', 1280))

  def test_efficientdet_d6(self):
    self.assertSequenceEqual((51871782, 225584339747),
                             self.build_model('efficientdet-d6', 1280))

  def test_efficientdet_d7(self):
    self.assertSequenceEqual((51871782, 324815496167),
                             self.build_model('efficientdet-d7', 1536))


class BackboneTest(tf.test.TestCase):

  def test_backbone_feats(self):
    config = hparams_config.get_efficientdet_config('efficientdet-d0')
    images = tf.ones([4, 224, 224, 3])
    feats = efficientdet_arch.build_backbone(images, config)
    self.assertEqual(list(feats.keys()), [0, 1, 2, 3, 4, 5])
    self.assertEqual(feats[0].shape, [4, 224, 224, 3])
    self.assertEqual(feats[5].shape, [4, 7, 7, 320])


class BiFPNTest(tf.test.TestCase):

  def test_bifpn_dynamic_l3l7(self):
    p1 = efficientdet_arch.bifpn_dynamic_config(3, 7, None)
    p2 = efficientdet_arch.bifpn_fa_config()
    self.assertEqual(p1.weight_method, p2.weight_method)
    self.assertEqual(p1.nodes, p2.nodes)

  def test_bifpn_dynamic_l2l7(self):
    p = efficientdet_arch.bifpn_dynamic_config(2, 7, None)

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


if __name__ == '__main__':
  tf.test.main()
