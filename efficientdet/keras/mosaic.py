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
# =============================================================================
"""Mosaic data augmentation"""
import tensorflow as tf


def crop_image_and_box(image, size):
  limit = tf.shape(image) - size + 1
  offset = tf.random.uniform(
      [3], dtype=limit.dtype, minval=0, maxval=limit.dtype.max) % limit
  crop_image = tf.slice(image, offset, size)
  return crop_image, offset


def clip_box(box, clazz, min, max):
  ymin = tf.clip_by_value(box[:, 0], min[0], max[0])
  xmin = tf.clip_by_value(box[:, 1], min[1], max[1])
  ymax = tf.clip_by_value(box[:, 2], min[0], max[0])
  xmax = tf.clip_by_value(box[:, 3], min[1], max[1])
  cliped_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
  mask = tf.logical_or(
      tf.not_equal(xmax - xmin, 0), tf.not_equal(ymax - ymin, 0))
  final_boxes = tf.boolean_mask(cliped_boxes, mask, axis=0)
  final_classes = tf.boolean_mask(clazz, mask, axis=0)
  return final_boxes, final_classes


def calculate_boxes_offset(box, offset_y, offset_x):
  return tf.stack([
      box[:, 0] + offset_y, box[:, 1] + offset_x, box[:, 2] + offset_y,
      box[:, 3] + offset_x
  ],
                  axis=1)


def mosaic(images, boxes, classes, size):
  with tf.name_scope('mosaic'):
    y = tf.cast(
        tf.random.uniform([], size[0] / 4, size[0] * 3 / 4), dtype=tf.int32)
    x = tf.cast(
        tf.random.uniform([], size[1] / 4, size[1] * 3 / 4), dtype=tf.int32)

    temp_size = [y, x, 3]
    crop_image1, offset = crop_image_and_box(images[0], temp_size)
    box = calculate_boxes_offset(boxes[0], -offset[0], -offset[1])
    boxes[0], classes[0] = clip_box(box, classes[0], [0, 0], [y, x])

    temp_size = [size[0] - y, x, 3]
    crop_image2, offset = crop_image_and_box(images[1], temp_size)
    box = calculate_boxes_offset(boxes[1], y - offset[0], -offset[1])
    boxes[1], classes[1] = clip_box(box, classes[1], [y, 0], [size[0], x])

    temp_size = [y, size[1] - x, 3]
    crop_image3, offset = crop_image_and_box(images[2], temp_size)
    box = calculate_boxes_offset(boxes[2], -offset[0], x - offset[1])
    boxes[2], classes[2] = clip_box(box, classes[2], [0, x], [y, size[1]])

    temp_size = [size[0] - y, size[1] - x, 3]
    crop_image4, offset = crop_image_and_box(images[3], temp_size)
    box = calculate_boxes_offset(boxes[3], y - offset[0], x - offset[1])
    boxes[3], classes[3] = clip_box(box, classes[3], [y, x], size[:2])

    temp1 = tf.concat([crop_image1, crop_image2], axis=0)
    temp2 = tf.concat([crop_image3, crop_image4], axis=0)
    final = tf.concat([temp1, temp2], axis=1)
    return final, tf.concat(boxes, axis=0), tf.concat(classes, axis=0)
