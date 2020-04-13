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
"""Data loader and processing.

This module is borrowed from TPU RetinaNet implementation:
https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/anchors.py
"""

import tensorflow.compat.v1 as tf

import anchors
from object_detection import preprocessor
from object_detection import tf_example_decoder

MAX_NUM_INSTANCES = 100


class InputProcessor(object):
  """Base class of Input processor."""

  def __init__(self, image, output_size):
    """Initializes a new `InputProcessor`.

    Args:
      image: The input image before processing.
      output_size: The output image size after calling resize_and_crop_image
        function.
    """
    self._image = image
    if isinstance(output_size, int):
      self._output_size = (output_size, output_size)
    else:
      self._output_size = output_size
    # Parameters to control rescaling and shifting during preprocessing.
    # Image scale defines scale from original image to scaled image.
    self._image_scale = tf.constant(1.0)
    # The integer height and width of scaled image.
    self._scaled_height = tf.shape(image)[0]
    self._scaled_width = tf.shape(image)[1]
    # The x and y translation offset to crop scaled image to the output size.
    self._crop_offset_y = tf.constant(0)
    self._crop_offset_x = tf.constant(0)

  def normalize_image(self):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    self._image = tf.image.convert_image_dtype(self._image, dtype=tf.float32)
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    self._image -= offset

    scale = tf.constant([0.229, 0.224, 0.225])
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    self._image /= scale

  def set_training_random_scale_factors(self, scale_min, scale_max):
    """Set the parameters for multiscale training."""
    # Select a random scale factor.
    random_scale_factor = tf.random_uniform([], scale_min, scale_max)
    scaled_size_y = tf.to_int32(random_scale_factor * self._output_size[0])
    scaled_size_x = tf.to_int32(random_scale_factor * self._output_size[1])

    # Recompute the accurate scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    image_scale_y = tf.to_float(scaled_size_y) / tf.to_float(height)
    image_scale_x = tf.to_float(scaled_size_x) / tf.to_float(width)
    image_scale = tf.minimum(image_scale_x, image_scale_y)

    # Select non-zero random offset (x, y) if scaled image is larger than
    # self._output_size.
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    offset_y = tf.to_float(scaled_height - self._output_size[0])
    offset_x = tf.to_float(scaled_width - self._output_size[1])
    offset_y = tf.maximum(0.0, offset_y) * tf.random_uniform([], 0, 1)
    offset_x = tf.maximum(0.0, offset_x) * tf.random_uniform([], 0, 1)
    offset_y = tf.to_int32(offset_y)
    offset_x = tf.to_int32(offset_x)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    self._crop_offset_x = offset_x
    self._crop_offset_y = offset_y

  def set_scale_factors_to_output_size(self):
    """Set the parameters to resize input image to self._output_size."""
    # Compute the scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    image_scale_y = tf.to_float(self._output_size[0]) / tf.to_float(height)
    image_scale_x = tf.to_float(self._output_size[1]) / tf.to_float(width)
    image_scale = tf.minimum(image_scale_x, image_scale_y)
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width

  def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
    """Resize input image and crop it to the self._output dimension."""
    scaled_image = tf.image.resize_images(
        self._image, [self._scaled_height, self._scaled_width], method=method)
    scaled_image = scaled_image[
        self._crop_offset_y:self._crop_offset_y + self._output_size[0],
        self._crop_offset_x:self._crop_offset_x + self._output_size[1], :]
    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, self._output_size[0], self._output_size[1])
    return output_image


class DetectionInputProcessor(InputProcessor):
  """Input processor for object detection."""

  def __init__(self, image, output_size, boxes=None, classes=None):
    InputProcessor.__init__(self, image, output_size)
    self._boxes = boxes
    self._classes = classes

  def random_horizontal_flip(self):
    """Randomly flip input image and bounding boxes."""
    self._image, self._boxes = preprocessor.random_horizontal_flip(
        self._image, boxes=self._boxes)

  def clip_boxes(self, boxes):
    """Clip boxes to fit in an image."""
    boxes = tf.where(tf.less(boxes, 0), tf.zeros_like(boxes), boxes)
    boxes = tf.where(tf.greater(boxes, self._output_size[0] - 1),
                     (self._output_size[1] - 1) * tf.ones_like(boxes), boxes)
    return boxes

  def resize_and_crop_boxes(self):
    """Resize boxes and crop it to the self._output dimension."""
    boxlist = preprocessor.box_list.BoxList(self._boxes)
    boxes = preprocessor.box_list_scale(
        boxlist, self._scaled_height, self._scaled_width).get()
    # Adjust box coordinates based on the offset.
    box_offset = tf.stack([self._crop_offset_y, self._crop_offset_x,
                           self._crop_offset_y, self._crop_offset_x,])
    boxes -= tf.to_float(tf.reshape(box_offset, [1, 4]))
    # Clip the boxes.
    boxes = self.clip_boxes(boxes)
    # Filter out ground truth boxes that are all zeros.
    indices = tf.where(tf.not_equal(tf.reduce_sum(boxes, axis=1), 0))
    boxes = tf.gather_nd(boxes, indices)
    classes = tf.gather_nd(self._classes, indices)
    return boxes, classes

  @property
  def image_scale(self):
    # Return image scale from original image to scaled image.
    return self._image_scale

  @property
  def image_scale_to_original(self):
    # Return image scale from scaled image to original image.
    return 1.0 / self._image_scale

  @property
  def offset_x(self):
    return self._crop_offset_x

  @property
  def offset_y(self):
    return self._crop_offset_y


def pad_to_fixed_size(data, pad_value, output_shape):
  """Pad data to a fixed length at the first dimension.

  Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

  Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
  """
  max_num_instances = output_shape[0]
  dimension = output_shape[1]
  data = tf.reshape(data, [-1, dimension])
  num_instances = tf.shape(data)[0]
  assert_length = tf.Assert(
      tf.less_equal(num_instances, max_num_instances), [num_instances])
  with tf.control_dependencies([assert_length]):
    pad_length = max_num_instances - num_instances
  paddings = pad_value * tf.ones([pad_length, dimension])
  padded_data = tf.concat([data, paddings], axis=0)
  padded_data = tf.reshape(padded_data, output_shape)
  return padded_data


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, is_training, use_fake_data=False):
    self._file_pattern = file_pattern
    self._is_training = is_training
    self._use_fake_data = use_fake_data
    self._max_num_instances = MAX_NUM_INSTANCES

  def __call__(self, params):
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        image: Image tensor that is preprocessed to have normalized value and
          fixed dimension [image_height, image_width, 3]
        cls_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors]. The height_l and width_l
          represent the dimension of class logits at l-th level.
        box_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        num_positives: Number of positive anchors in the image.
        source_id: Source image id. Default value -1 if the source id is empty
          in the groundtruth annotation.
        image_scale: Scale of the processed image to the original image.
        boxes: Groundtruth bounding box annotations. The box is represented in
          [y1, x1, y2, x2] format. The tensor is padded with -1 to the fixed
          dimension [self._max_num_instances, 4].
        is_crowds: Groundtruth annotations to indicate if an annotation
          represents a group of instances by value {0, 1}. The tensor is
          padded with 0 to the fixed dimension [self._max_num_instances].
        areas: Groundtruth areas annotations. The tensor is padded with -1
          to the fixed dimension [self._max_num_instances].
        classes: Groundtruth classes annotations. The tensor is padded with -1
          to the fixed dimension [self._max_num_instances].
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        source_id = data['source_id']
        image = data['image']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        areas = data['groundtruth_area']
        is_crowds = data['groundtruth_is_crowd']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

        if params['skip_crowd_during_training'] and self._is_training:
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)

        # NOTE: The autoaugment method works best when used alongside the
        # standard horizontal flipping of images along with size jittering
        # and normalization.
        if params.get('autoaugment_policy', None) and self._is_training:
          from aug import autoaugment  # pylint: disable=g-import-not-at-top
          image, boxes = autoaugment.distort_image_with_autoaugment(
              image, boxes, params['autoaugment_policy'])

        input_processor = DetectionInputProcessor(
            image, params['image_size'], boxes, classes)
        input_processor.normalize_image()
        if self._is_training and params['input_rand_hflip']:
          input_processor.random_horizontal_flip()
        if self._is_training:
          input_processor.set_training_random_scale_factors(
              params['train_scale_min'], params['train_scale_max'])
        else:
          input_processor.set_scale_factors_to_output_size()
        image = input_processor.resize_and_crop_image()
        boxes, classes = input_processor.resize_and_crop_boxes()

        # Assign anchors.
        (cls_targets, box_targets,
         num_positives) = anchor_labeler.label_anchors(boxes, classes)

        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        # Pad groundtruth data for evaluation.
        image_scale = input_processor.image_scale_to_original
        boxes *= image_scale
        is_crowds = tf.cast(is_crowds, dtype=tf.float32)
        boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
        is_crowds = pad_to_fixed_size(is_crowds, 0,
                                      [self._max_num_instances, 1])
        areas = pad_to_fixed_size(areas, -1, [self._max_num_instances, 1])
        classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])
        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)
        return (image, cls_targets, box_targets, num_positives, source_id,
                image_scale, boxes, is_crowds, areas, classes)

    batch_size = params['batch_size']
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=self._is_training)
    if self._is_training:
      dataset = dataset.repeat()

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            _prefetch_dataset, cycle_length=32, sloppy=self._is_training))
    if self._is_training:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=64)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _process_example(images, cls_targets, box_targets, num_positives,
                         source_ids, image_scales, boxes, is_crowds, areas,
                         classes):
      """Processes one batch of data."""
      labels = {}
      # Count num_positives in a batch.
      num_positives_batch = tf.reduce_mean(num_positives)
      labels['mean_num_positives'] = tf.reshape(
          tf.tile(tf.expand_dims(num_positives_batch, 0), [
              batch_size,
          ]), [batch_size, 1])

      for level in range(params['min_level'], params['max_level'] + 1):
        labels['cls_targets_%d' % level] = cls_targets[level]
        labels['box_targets_%d' % level] = box_targets[level]
      # Concatenate groundtruth annotations to a tensor.
      groundtruth_data = tf.concat([boxes, is_crowds, areas, classes], axis=2)
      labels['source_ids'] = source_ids
      labels['groundtruth_data'] = groundtruth_data
      labels['image_scales'] = image_scales
      return images, labels

    dataset = dataset.map(_process_example)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      dataset = dataset.take(1).cache().repeat()
    return dataset
