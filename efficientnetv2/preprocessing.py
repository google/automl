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
"""ImageNet preprocessing."""
from absl import logging
import tensorflow as tf
import autoaugment
import preprocess_legacy


def preprocess_for_train(image,
                         image_size,
                         augname=None,
                         ra_num_layers=None,
                         ra_magnitude=None,
                         transformations=None):
  """Preprocesses the given image for train."""
  transformations = transformations or 'crop|flip'
  # crop
  if 'crop' in transformations:
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(0.05, 1.0),
        min_object_covered=0,
        use_image_if_no_bounding_boxes=True)
    image = tf.slice(image, begin, size)
  # resize
  image.set_shape([None, None, 3])
  image = tf.image.resize(image, [image_size, image_size])
  # flip
  if 'flip' in transformations:
    image = tf.image.random_flip_left_right(image)

  if augname:
    logging.info('Apply AutoAugment policy %s', augname)
    input_image_type = image.dtype
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    image = autoaugment.distort_image(image, augname, ra_num_layers,
                                      ra_magnitude)
    image = tf.cast(image, dtype=input_image_type)

  return image


def preprocess_for_eval(image, image_size, transformations=None):
  """Process image for eval."""
  transformations = transformations or ('crop' if image_size < 320 else '')
  if 'crop' in transformations:
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    ratio = image_size / (image_size + 32)  # for imagenet.
    crop_size = tf.cast(
        (ratio * tf.cast(tf.minimum(height, width), tf.float32)), tf.int32)
    y, x = (height - crop_size) // 2, (width - crop_size) // 2
    image = tf.image.crop_to_bounding_box(image, y, x, crop_size, crop_size)
  image.set_shape([None, None, 3])
  return tf.image.resize(image, [image_size, image_size])


def preprocess_for_finetune(image,
                            image_size,
                            is_training,
                            augname=None,
                            ra_num_layers=None,
                            ra_magnitude=None):
  """Preprocessor for finetuning."""
  logging.info('using preprocessing for finetuning.')
  # normalize
  mean, std = 0.5, 0.5
  image = (tf.cast(image, tf.float32) / 255.0 - mean) / std
  image.set_shape([None, None, 3])
  image = tf.image.resize(image, [image_size, image_size])

  if is_training:
    #  Random flip.
    image = tf.image.random_flip_left_right(image)
    if 'autoaug' in augname or 'randaug' in augname:
      # autoaug requires uint8 image values in [0, 255]
      image = ((image * std) + mean) * 255
      image = tf.cast(tf.clip_by_value(image, 0.0, 255.0), dtype=tf.uint8)
      if 'autoaug' in augname:
        image = autoaugment.distort_image(image, 'autoaug', ra_num_layers,
                                          ra_magnitude)
      if 'randaug' in augname:
        image = autoaugment.distort_image(image, 'randaug', ra_num_layers,
                                          ra_magnitude)
      image = (tf.cast(image, tf.float32) / 255.0 - mean) / std

    if augname == 'ft' or 'cutout' in augname:
      # Only apply cutout in default (replacing with random values).
      replace = tf.random.uniform(image.shape, 0.0, 1.0, image.dtype)
      image = autoaugment.cutout(
          image, pad_size=image_size // 4, replace=replace)
  return image


def preprocess_image(image,
                     image_size,
                     is_training,
                     image_dtype=None,
                     augname=None,
                     ra_num_layers=2,
                     ra_magnitude=15):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image binary of arbitrary size.
    image_size: int, image size after cropping or resizing.
    is_training: `bool` for whether the preprocessing is for training.
    image_dtype: image dtype. If None, default to tf.float32.
    augname: `string`, name of augmentation, 'autoaug' or 'randaug'.
    ra_num_layers: 'int', if RandAug is used, what should the number of layers
      be. See autoaugment.py for detailed description.
    ra_magnitude: 'int', if RandAug is used, what should the magnitude be. See
      autoaugment.py for detailed description.

  Returns:
    A preprocessed image `Tensor` with value range of [-1, 1].
  """
  if augname and augname.startswith('effnetv1_'):
    # Legacy efficientnet v1 preprocessing (for compitability and comparison).
    # Difference: (1) mean/stddev; (2) biubic; (3) cropping.
    augname = augname[len('effnetv1_'):]
    return preprocess_legacy.preprocess_image(image, image_size, is_training,
                                              image_dtype, augname,
                                              ra_num_layers, ra_magnitude)

  is_raw = (image.dtype == tf.string)
  image = tf.image.decode_image(image, channels=3) if is_raw else image

  if augname and augname.startswith('ft'):
    image = preprocess_for_finetune(image, image_size, is_training, augname,
                                    ra_num_layers, ra_magnitude)
  else:
    if is_training:
      image = preprocess_for_train(image, image_size, augname, ra_num_layers,
                                   ra_magnitude)
    else:
      image = preprocess_for_eval(image, image_size)
    image = (image - 128.0) / 128.0  # normalize to [-1, 1]
  return tf.cast(image, dtype=image_dtype or tf.float32)
