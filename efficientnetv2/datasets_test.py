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
"""Tests datasets."""
import tensorflow as tf
import datasets


class ImagenetInputTest(tf.test.TestCase):

  def test_imagenet(self):
    ds_class = datasets.get_dataset_class('imagenet')
    ds = ds_class(
        is_training=False,
        data_dir='null',
        cache=False,
        image_size=224,
        image_dtype=None,
        augname=None,
        mixup_alpha=0,
        ra_num_layers=2,
        ra_magnitude=20)
    params = {'batch_size': 2}
    for _, labels in ds.input_fn(params):
      label = labels['label']
      self.assertAllClose(label[:, 0:4], [[0, 0, 0, 0], [0, 0, 0, 0]])
      break

  def test_imagenet21k(self):
    ds_class = datasets.get_dataset_class('imagenet21k')
    ds = ds_class(
        is_training=False,
        data_dir='null',
        cache=False,
        image_size=224,
        image_dtype=None,
        augname=None,
        mixup_alpha=0,
        ra_num_layers=2,
        ra_magnitude=20)
    params = {'batch_size': 2}
    for _, labels in ds.input_fn(params):
      label = labels['label']
      self.assertAllClose(label[:, 0:4], [[0, 0, 1, 1], [0, 0, 1, 1]])
      break


class DatasetConfigTest(tf.test.TestCase):

  def test_dataset_config(self):
    cfg = datasets.get_dataset_config('cifar10ft')
    self.assertEqual(cfg.data.ds_name, 'cifar10')


if __name__ == '__main__':
  tf.test.main()
