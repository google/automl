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
"""Tests for efficientnet_model."""
from absl import logging
import tensorflow.compat.v1 as tf

import utils
from backbone import efficientnet_model


class ModelTest(tf.test.TestCase):

  def test_bottleneck_block(self):
    """Test for creating a model with bottleneck block arguments."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.batch_norm_class(False))
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=3,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=True,
            strides=[2, 2],
            conv_type=0,
            fused_conv=0,
            super_pixel=0)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    outputs = model(images, training=True)
    self.assertEqual((10, 10), outputs[0].shape)

  def test_fused_bottleneck_block(self):
    """Test for creating a model with fused bottleneck block arguments."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.TpuBatchNormalization)
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=3,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=True,
            strides=[2, 2],
            conv_type=0,
            fused_conv=1,
            super_pixel=0)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    outputs = model(images, training=True)
    self.assertEqual((10, 10), outputs[0].shape)

  def test_bottleneck_block_with_superpixel_layer(self):
    """Test for creating a model with fused bottleneck block arguments."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.TpuBatchNormalization)
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=3,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=True,
            strides=[2, 2],
            conv_type=0,
            fused_conv=0,
            super_pixel=1)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    outputs = model(images, training=True)
    self.assertEqual((10, 10), outputs[0].shape)

  def test_bottleneck_block_with_superpixel_tranformation(self):
    """Test for creating a model with fused bottleneck block arguments."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.TpuBatchNormalization)
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=3,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=True,
            strides=[2, 2],
            conv_type=0,
            fused_conv=0,
            super_pixel=2)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    outputs = model(images, training=True)
    self.assertEqual((10, 10), outputs[0].shape)

  def test_se_block(self):
    """Test for creating a model with SE block arguments."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.TpuBatchNormalization)
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=3,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=False,
            strides=[2, 2],
            se_ratio=0.8,
            conv_type=0,
            fused_conv=0,
            super_pixel=0)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    outputs = model(images, training=True)
    self.assertEqual((10, 10), outputs[0].shape)

  def test_variables(self):
    """Test for variables in blocks to be included in `model.variables`."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.TpuBatchNormalization)
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=3,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=False,
            strides=[2, 2],
            se_ratio=0.8,
            conv_type=0,
            fused_conv=0,
            super_pixel=0)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    _ = model(images, training=True)
    var_names = {var.name for var in model.variables}
    self.assertIn('model/blocks_0/conv2d/kernel:0', var_names)

  def test_reduction_endpoint_with_single_block_with_sp(self):
    """Test reduction point with single block/layer."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.TpuBatchNormalization)
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=1,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=False,
            strides=[2, 2],
            se_ratio=0.8,
            conv_type=0,
            fused_conv=0,
            super_pixel=1)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    _ = model(images, training=True)
    self.assertIn('reduction_1', model.endpoints)
    # single block should have one and only one reduction endpoint
    self.assertNotIn('reduction_2', model.endpoints)

  def test_reduction_endpoint_with_single_block_without_sp(self):
    """Test reduction point with single block/layer."""
    images = tf.zeros((10, 128, 128, 3), dtype=tf.float32)
    global_params = efficientnet_model.GlobalParams(
        1.0,
        1.0,
        0,
        'channels_last',
        num_classes=10,
        batch_norm=utils.TpuBatchNormalization)
    blocks_args = [
        efficientnet_model.BlockArgs(
            kernel_size=3,
            num_repeat=1,
            input_filters=3,
            output_filters=6,
            expand_ratio=6,
            id_skip=False,
            strides=[2, 2],
            se_ratio=0.8,
            conv_type=0,
            fused_conv=0,
            super_pixel=0)
    ]
    model = efficientnet_model.Model(blocks_args, global_params)
    _ = model(images, training=True)
    self.assertIn('reduction_1', model.endpoints)
    # single block should have one and only one reduction endpoint
    self.assertNotIn('reduction_2', model.endpoints)

if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
