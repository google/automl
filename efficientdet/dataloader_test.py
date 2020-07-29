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
import os
import tempfile
import tensorflow as tf

import dataloader
import hparams_config
from dataset import tfrecord_util
from keras import anchors
from object_detection import tf_example_decoder


class DataloaderTest(tf.test.TestCase):

  def _make_fake_tfrecord(self):
    tfrecord_path = os.path.join(tempfile.mkdtemp(), 'test.tfrecords')
    writer = tf.io.TFRecordWriter(tfrecord_path)
    encoded_jpg = tf.io.encode_jpeg(tf.ones([512, 512, 3], dtype=tf.uint8))
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                    tfrecord_util.int64_feature(512),
                'image/width':
                    tfrecord_util.int64_feature(512),
                'image/filename':
                    tfrecord_util.bytes_feature('test_file_name.jpg'.encode(
                        'utf8')),
                'image/source_id':
                    tfrecord_util.bytes_feature('123456'.encode('utf8')),
                'image/key/sha256':
                    tfrecord_util.bytes_feature('qwdqwfw12345'.encode('utf8')),
                'image/encoded':
                    tfrecord_util.bytes_feature(encoded_jpg.numpy()),
                'image/format':
                    tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                    tfrecord_util.float_list_feature([0.1]),
                'image/object/bbox/xmax':
                    tfrecord_util.float_list_feature([0.1]),
                'image/object/bbox/ymin':
                    tfrecord_util.float_list_feature([0.2]),
                'image/object/bbox/ymax':
                    tfrecord_util.float_list_feature([0.2]),
                'image/object/class/text':
                    tfrecord_util.bytes_list_feature(['test'.encode('utf8')]),
                'image/object/class/label':
                    tfrecord_util.int64_list_feature([1]),
                'image/object/difficult':
                    tfrecord_util.int64_list_feature([]),
                'image/object/truncated':
                    tfrecord_util.int64_list_feature([]),
                'image/object/view':
                    tfrecord_util.bytes_list_feature([]),
            }))
    writer.write(example.SerializeToString())
    return tfrecord_path

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
    tfrecord_path = self._make_fake_tfrecord()
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    value = next(iter(dataset))
    reader = dataloader.InputReader(tfrecord_path, True)
    result = reader.dataset_parser(value, example_decoder, anchor_labeler,
                                   params)
    self.assertEqual(len(result), 11)


if __name__ == '__main__':
  tf.test.main()
