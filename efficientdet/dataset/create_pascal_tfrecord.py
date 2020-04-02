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
r"""Convert PASCAL dataset to TFRecord.

Example usage:
    python create_pascal_tfrecord.py  --data_dir=/tmp/VOCdevkit  \
        --year=VOC2012  --output_path=/tmp/pascal
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from dataset import tfrecord_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord and json.')
flags.DEFINE_string('label_map_json_path', None,
                    'Path to label map json file with a dictionary.')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_integer('num_shards', 100, 'Number of shards for output file.')
flags.DEFINE_integer('num_images', None, 'Max number of imags to process.')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

pascal_label_map_dict = {
    'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
    'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
    'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
    'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20,
}


def filename_to_int(filename):
  """Convert a string to a integer."""
  filename = filename.replace('_', '')
  filename = os.path.splitext(os.path.basename(filename))[0]
  return int(filename)


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages',
                       ann_json_dict=None):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running tfrecord_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.
    ann_json_dict: annotation json dictionary.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])
  image_id = filename_to_int(data['filename'])
  if ann_json_dict:
    image = {
        'file_name': data['filename'],
        'height': height,
        'width': width,
        'id': image_id,
    }
    ann_json_dict['images'].append(image)
    ann_box_id = 1

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

      if ann_json_dict:
        abs_xmin = int(obj['bndbox']['xmin'])
        abs_ymin = int(obj['bndbox']['ymin'])
        abs_xmax = int(obj['bndbox']['xmax'])
        abs_ymax = int(obj['bndbox']['ymax'])
        abs_width = abs_xmax - abs_xmin
        abs_height = abs_ymax - abs_ymin
        ann = {
            'area': abs_width * abs_height,
            'iscrowd': 0,
            'image_id': image_id,
            'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
            'category_id': label_map_dict[obj['name']],
            'id': ann_box_id,
            'ignore': 0,
            'segmentation': [],
        }
        ann_json_dict['annotations'].append(ann)
        ann_box_id += 1

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': tfrecord_util.int64_feature(height),
      'image/width': tfrecord_util.int64_feature(width),
      'image/filename': tfrecord_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': tfrecord_util.bytes_feature(
          str(image_id).encode('utf8')),
      'image/key/sha256': tfrecord_util.bytes_feature(key.encode('utf8')),
      'image/encoded': tfrecord_util.bytes_feature(encoded_jpg),
      'image/format': tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': tfrecord_util.float_list_feature(xmin),
      'image/object/bbox/xmax': tfrecord_util.float_list_feature(xmax),
      'image/object/bbox/ymin': tfrecord_util.float_list_feature(ymin),
      'image/object/bbox/ymax': tfrecord_util.float_list_feature(ymax),
      'image/object/class/text': tfrecord_util.bytes_list_feature(classes_text),
      'image/object/class/label': tfrecord_util.int64_list_feature(classes),
      'image/object/difficult': tfrecord_util.int64_list_feature(difficult_obj),
      'image/object/truncated': tfrecord_util.int64_list_feature(truncated),
      'image/object/view': tfrecord_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))
  if not FLAGS.output_path:
    raise ValueError('output_path cannot be empty.')

  data_dir = FLAGS.data_dir
  years = ['VOC2007', 'VOC2012']
  if FLAGS.year != 'merged':
    years = [FLAGS.year]

  logging.info('writing to output path: %s', FLAGS.output_path)
  writers = [
      tf.python_io.TFRecordWriter(
          FLAGS.output_path + '-%05d-of-%05d.tfrecord' % (i, FLAGS.num_shards))
      for i in range(FLAGS.num_shards)
  ]

  if FLAGS.label_map_json_path:
    with tf.io.gfile.GFile(FLAGS.label_map_json_path, 'rb') as f:
      label_map_dict = json.load(f)
  else:
    label_map_dict = pascal_label_map_dict

  for year in years:
    ann_json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }
    for class_name, class_id in label_map_dict.items():
      cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(cls)

    logging.info('Reading from PASCAL %s dataset.', year)
    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                 'aeroplane_' + FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
    examples_list = tfrecord_util.read_examples_list(examples_path)
    for idx, example in enumerate(examples_list):
      if FLAGS.num_images and idx >= FLAGS.num_images:
        break
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))
      path = os.path.join(annotations_dir, example + '.xml')
      with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = tfrecord_util.recursive_parse_xml_to_dict(xml)['annotation']

      tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances,
                                      ann_json_dict=ann_json_dict)
      writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())

  for writer in writers:
    writer.close()

  with tf.io.gfile.GFile(FLAGS.output_path + '.json', 'w') as f:
    json.dump(ann_json_dict, f)


if __name__ == '__main__':
  tf.app.run()
