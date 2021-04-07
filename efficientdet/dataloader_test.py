# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data loader and processing test cases."""

import tensorflow as tf

import dataloader
import hparams_config
import test_util

from keras import anchors
from object_detection import tf_example_decoder


class DataloaderTest(tf.test.TestCase):

  def test_parser(self):
    tf.random.set_seed(111111)
    params = hparams_config.get_detection_config('efficientdet-d0').as_dict()
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    example_decoder = tf_example_decoder.TfExampleDecoder(
        regenerate_source_id=params['regenerate_source_id'])
    tfrecord_path = test_util.make_fake_tfrecord(self.get_temp_dir())
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    value = next(iter(dataset))
    reader = dataloader.InputReader(tfrecord_path, True)
    result = reader.dataset_parser(value, example_decoder, anchor_labeler,
                                   params)
    self.assertEqual(len(result), 11)


if __name__ == '__main__':
  tf.test.main()
