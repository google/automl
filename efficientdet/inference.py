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
r"""Inference related utilities."""

from __future__ import absolute_import
from __future__ import division
# gtype import
from __future__ import print_function

import copy
import functools
import os
import time

from absl import logging
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from typing import Text, Dict, Any, List, Tuple, Union
import yaml

import anchors
import dataloader
import det_model_fn
import hparams_config
import utils
from visualize import vis_utils

coco_id_mapping = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}  # pyformat: disable


def image_preprocess(image, image_size: Union[int, Tuple[int, int]]):
  """Preprocess image for inference.

  Args:
    image: input image, can be a tensor or a numpy arary.
    image_size: single integer of image size for square image or tuple of two
      integers, in the format of (image_height, image_width).

  Returns:
    (image, scale): a tuple of processed image and its scale.
  """
  input_processor = dataloader.DetectionInputProcessor(image, image_size)
  input_processor.normalize_image()
  input_processor.set_scale_factors_to_output_size()
  image = input_processor.resize_and_crop_image()
  image_scale = input_processor.image_scale_to_original
  return image, image_scale


@tf.autograph.to_graph
def batch_image_files_decode(image_files):
  raw_images = tf.TensorArray(tf.uint8, size=0, dynamic_size=True)
  for i in tf.range(tf.shape(image_files)[0]):
    image = tf.io.decode_image(image_files[i])
    image.set_shape([None, None, None])
    raw_images = raw_images.write(i, image)
  return raw_images.stack()


def batch_image_preprocess(raw_images,
                           image_size: Union[int, Tuple[int, int]],
                           batch_size: int = None):
  """Preprocess batched images for inference.

  Args:
    raw_images: a list of images, each image can be a tensor or a numpy arary.
    image_size: single integer of image size for square image or tuple of two
      integers, in the format of (image_height, image_width).
    batch_size: if None, use map_fn to deal with dynamic batch size.

  Returns:
    (image, scale): a tuple of processed images and scales.
  """
  if not batch_size:
    # map_fn is a little bit slower due to some extra overhead.
    map_fn = functools.partial(image_preprocess, image_size=image_size)
    images, scales = tf.map_fn(
        map_fn, raw_images, dtype=(tf.float32, tf.float32), back_prop=False)
    return (images, scales)

  # If batch size is known, use a simple loop.
  scales, images = [], []
  for i in range(batch_size):
    image, scale = image_preprocess(raw_images[i], image_size)
    scales.append(scale)
    images.append(image)
  images = tf.stack(images)
  scales = tf.stack(scales)
  return (images, scales)


def build_inputs(image_path_pattern: Text, image_size: Union[int, Tuple[int,
                                                                        int]]):
  """Read and preprocess input images.

  Args:
    image_path_pattern: a path to indicate a single or multiple files.
    image_size: single integer of image size for square image or tuple of two
      integers, in the format of (image_height, image_width).

  Returns:
    (raw_images, images, scales): raw images, processed images, and scales.

  Raises:
    ValueError if image_path_pattern doesn't match any file.
  """
  raw_images, images, scales = [], [], []
  for f in tf.io.gfile.glob(image_path_pattern):
    image = Image.open(f)
    raw_images.append(image)
    image, scale = image_preprocess(image, image_size)
    images.append(image)
    scales.append(scale)
  if not images:
    raise ValueError(
        'Cannot find any images for pattern {}'.format(image_path_pattern))
  return raw_images, tf.stack(images), tf.stack(scales)


def build_model(model_name: Text, inputs: tf.Tensor, **kwargs):
  """Build model for a given model name.

  Args:
    model_name: the name of the model.
    inputs: an image tensor or a numpy array.
    **kwargs: extra parameters for model builder.

  Returns:
    (cls_outputs, box_outputs): the outputs for class and box predictions.
    Each is a dictionary with key as feature level and value as predictions.
  """
  model_arch = det_model_fn.get_model_arch(model_name)
  cls_outputs, box_outputs = utils.build_model_with_precision(
      kwargs.get('precision', None), model_arch, inputs, model_name, **kwargs)
  if kwargs.get('precision', None):
    # Post-processing has multiple places with hard-coded float32.
    # TODO(tanmingxing): Remove them once post-process can adpat to dtypes.
    cls_outputs = {k: tf.cast(v, tf.float32) for k, v in cls_outputs.items()}
    box_outputs = {k: tf.cast(v, tf.float32) for k, v in box_outputs.items()}
  return cls_outputs, box_outputs


def restore_ckpt(sess, ckpt_path, ema_decay=0.9998, export_ckpt=None):
  """Restore variables from a given checkpoint.

  Args:
    sess: a tf session for restoring or exporting models.
    ckpt_path: the path of the checkpoint. Can be a file path or a folder path.
    ema_decay: ema decay rate. If None or zero or negative value, disable ema.
    export_ckpt: whether to export the restored model.
  """
  sess.run(tf.global_variables_initializer())
  if tf.io.gfile.isdir(ckpt_path):
    ckpt_path = tf.train.latest_checkpoint(ckpt_path)
  if ema_decay > 0:
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    ema_vars = utils.get_ema_vars()
    var_dict = ema.variables_to_restore(ema_vars)
    ema_assign_op = ema.apply(ema_vars)
  else:
    var_dict = utils.get_ema_vars()
    ema_assign_op = None
  tf.train.get_or_create_global_step()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(var_dict, max_to_keep=1)
  saver.restore(sess, ckpt_path)

  if export_ckpt:
    print('export model to {}'.format(export_ckpt))
    if ema_assign_op is not None:
      sess.run(ema_assign_op)
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    saver.save(sess, export_ckpt)


def det_post_process_combined(params, cls_outputs, box_outputs, scales,
                              min_score_thresh, max_boxes_to_draw):
  """A combined version of det_post_process with dynamic batch size support."""
  batch_size = tf.shape(list(cls_outputs.values())[0])[0]
  cls_outputs_all = []
  box_outputs_all = []
  # Concatenates class and box of all levels into one tensor.
  for level in range(params['min_level'], params['max_level'] + 1):
    if params['data_format'] == 'channels_first':
      cls_outputs[level] = tf.transpose(cls_outputs[level], [0, 2, 3, 1])
      box_outputs[level] = tf.transpose(box_outputs[level], [0, 2, 3, 1])

    cls_outputs_all.append(tf.reshape(
        cls_outputs[level],
        [batch_size, -1, params['num_classes']]))
    box_outputs_all.append(tf.reshape(
        box_outputs[level], [batch_size, -1, 4]))
  cls_outputs_all = tf.concat(cls_outputs_all, 1)
  box_outputs_all = tf.concat(box_outputs_all, 1)

  # Create anchor_label for picking top-k predictions.
  eval_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                 params['num_scales'], params['aspect_ratios'],
                                 params['anchor_scale'], params['image_size'])
  anchor_boxes = eval_anchors.boxes
  scores = tf.math.sigmoid(cls_outputs_all)
  # apply bounding box regression to anchors
  boxes = anchors.decode_box_outputs_tf(box_outputs_all, anchor_boxes)
  boxes = tf.expand_dims(boxes, axis=2)
  scales = tf.expand_dims(scales, axis=-1)
  nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
      tf.image.combined_non_max_suppression(
          boxes,
          scores,
          max_boxes_to_draw,
          max_boxes_to_draw,
          score_threshold=min_score_thresh,
          clip_boxes=False))
  del valid_detections  # to be used in futue.

  image_ids = tf.cast(
      tf.tile(
          tf.expand_dims(tf.range(batch_size), axis=1), [1, max_boxes_to_draw]),
      dtype=tf.float32)
  image_size = params['image_size']
  ymin = tf.clip_by_value(nmsed_boxes[..., 0], 0, image_size[0]) * scales
  xmin = tf.clip_by_value(nmsed_boxes[..., 1], 0, image_size[1]) * scales
  ymax = tf.clip_by_value(nmsed_boxes[..., 2], 0, image_size[0]) * scales
  xmax = tf.clip_by_value(nmsed_boxes[..., 3], 0, image_size[1]) * scales

  detection_list = [
      # Format: (image_ids, ymin, xmin, ymax, xmax, score, class)
      image_ids, ymin, xmin, ymax, xmax, nmsed_scores,
      tf.cast(nmsed_classes + 1, tf.float32)
  ]
  detections = tf.stack(detection_list, axis=2, name='detections')
  return detections


def det_post_process(params: Dict[Any, Any], cls_outputs: Dict[int, tf.Tensor],
                     box_outputs: Dict[int, tf.Tensor], scales: List[float],
                     min_score_thresh, max_boxes_to_draw):
  """Post preprocessing the box/class predictions.

  Args:
    params: a parameter dictionary that includes `min_level`, `max_level`,
      `batch_size`, and `num_classes`.
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in [batch_size, height, width,
      num_anchors * 4].
    scales: a list of float values indicating image scale.
    min_score_thresh: A float representing the threshold for deciding when to
      remove boxes based on score.
    max_boxes_to_draw: Max number of boxes to draw.

  Returns:
    detections_batch: a batch of detection results. Each detection is a tensor
      with each row representing [image_id, ymin, xmin, ymax, xmax, score, class].
  """
  if not params['batch_size']:
    # Use combined version for dynamic batch size.
    return det_post_process_combined(params, cls_outputs, box_outputs, scales,
                                     min_score_thresh, max_boxes_to_draw)

  # TODO(tanmingxing): refactor the code to make it more explicity.
  outputs = {
      'cls_outputs_all': [None],
      'box_outputs_all': [None],
      'indices_all': [None],
      'classes_all': [None]
  }
  det_model_fn.add_metric_fn_inputs(params, cls_outputs, box_outputs, outputs,
                                    -1)

  # Create anchor_label for picking top-k predictions.
  eval_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                 params['num_scales'], params['aspect_ratios'],
                                 params['anchor_scale'], params['image_size'])
  anchor_labeler = anchors.AnchorLabeler(eval_anchors, params['num_classes'])

  # Add all detections for each input image.
  detections_batch = []
  for index in range(params['batch_size']):
    cls_outputs_per_sample = outputs['cls_outputs_all'][index]
    box_outputs_per_sample = outputs['box_outputs_all'][index]
    indices_per_sample = outputs['indices_all'][index]
    classes_per_sample = outputs['classes_all'][index]
    detections = anchor_labeler.generate_detections(
        cls_outputs_per_sample,
        box_outputs_per_sample,
        indices_per_sample,
        classes_per_sample,
        image_id=[index],
        image_scale=[scales[index]],
        image_size=params['image_size'],
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw,
        disable_pyfun=params.get('disable_pyfun'))
    if params['batch_size'] > 1:
      # pad to fixed length if batch size > 1.
      padding_size = max_boxes_to_draw - tf.shape(detections)[0]
      detections = tf.pad(detections, [[0, padding_size], [0, 0]])
    detections_batch.append(detections)
  return tf.stack(detections_batch, name='detections')


def visualize_image(image,
                    boxes,
                    classes,
                    scores,
                    id_mapping,
                    min_score_thresh=anchors.MIN_SCORE_THRESH,
                    max_boxes_to_draw=anchors.MAX_DETECTIONS_PER_IMAGE,
                    line_thickness=2,
                    **kwargs):
  """Visualizes a given image.

  Args:
    image: a image with shape [H, W, C].
    boxes: a box prediction with shape [N, 4] ordered [ymin, xmin, ymax, xmax].
    classes: a class prediction with shape [N].
    scores: A list of float value with shape [N].
    id_mapping: a dictionary from class id to name.
    min_score_thresh: minimal score for showing. If claass probability is below
      this threshold, then the object will not show up.
    max_boxes_to_draw: maximum bounding box to draw.
    line_thickness: how thick is the bounding box line.
    **kwargs: extra parameters.

  Returns:
    output_image: an output image with annotated boxes and classes.
  """
  category_index = {k: {'id': k, 'name': id_mapping[k]} for k in id_mapping}
  img = np.array(image)
  vis_utils.visualize_boxes_and_labels_on_image_array(
      img,
      boxes,
      classes,
      scores,
      category_index,
      min_score_thresh=min_score_thresh,
      max_boxes_to_draw=max_boxes_to_draw,
      line_thickness=line_thickness,
      **kwargs)
  return img


def parse_label_id_mapping(
    label_id_mapping: Union[Text, Dict[int, Text]] = None) -> Dict[int, Text]:
  """Parse label id mapping from a string or a yaml file.

  The label_id_mapping is a dict that maps class id to its name, such as:

    {
      1: "person",
      2: "dog"
    }

  Args:
    label_id_mapping:

  Returns:
    A dictionary with key as integer id and value as a string of name.
  """
  if label_id_mapping is None:
    return coco_id_mapping

  if isinstance(label_id_mapping, dict):
    label_id_dict = label_id_mapping
  elif isinstance(label_id_mapping, str):
    with tf.io.gfile.GFile(label_id_mapping) as f:
      label_id_dict = yaml.load(f, Loader=yaml.FullLoader)
  else:
    raise TypeError('label_id_mapping must be a dict or a yaml filename, '
                    'containing a mapping from class ids to class names.')

  return label_id_dict


def visualize_image_prediction(image,
                               prediction,
                               disable_pyfun=True,
                               label_id_mapping=None,
                               **kwargs):
  """Viusalize detections on a given image.

  Args:
    image: Image content in shape of [height, width, 3].
    prediction: a list of vector, with each vector has the format of [image_id,
      ymin, xmin, ymax, xmax, score, class].
    disable_pyfun: disable pyfunc for faster post processing.
    label_id_mapping: a map from label id to name.
    **kwargs: extra parameters for vistualization, such as min_score_thresh,
      max_boxes_to_draw, and line_thickness.

  Returns:
    a list of annotated images.
  """
  boxes = prediction[:, 1:5]
  classes = prediction[:, 6].astype(int)
  scores = prediction[:, 5]

  if not disable_pyfun:
    # convert [x, y, width, height] to [y, x, height, width]
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

  label_id_mapping = label_id_mapping or coco_id_mapping

  return visualize_image(image, boxes, classes, scores, label_id_mapping,
                         **kwargs)


class ServingDriver(object):
  """A driver for serving single or batch images.

  This driver supports serving with image files or arrays, with configurable
  batch size.

  Example 1. Serving streaming image contents:

    driver = inference.ServingDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=1)
    driver.build()
    for m in image_iterator():
      predictions = driver.serve_files([m])
      driver.visualize(m, predictions[0])
      # m is the new image with annotated boxes.

  Example 2. Serving batch image contents:

    imgs = []
    for f in ['/tmp/1.jpg', '/tmp/2.jpg']:
      imgs.append(np.array(Image.open(f)))

    driver = inference.ServingDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=len(imgs))
    driver.build()
    predictions = driver.serve_images(imgs)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], predictions[i])

  Example 3: another way is to use SavedModel:

    # step1: export a model.
    driver = inference.ServingDriver('efficientdet-d0', '/tmp/efficientdet-d0')
    driver.build()
    driver.export('/tmp/saved_model_path')

    # step2: Serve a model.
    with tf.Session() as sess:
      tf.saved_model.load(sess, ['serve'], self.saved_model_dir)
      raw_images = []
      for f in tf.io.gfile.glob('/tmp/images/*.jpg'):
        raw_images.append(np.array(PIL.Image.open(f)))
      detections = sess.run('detections:0', {'image_arrays:0': raw_images})
      driver = inference.ServingDriver(
        'efficientdet-d0', '/tmp/efficientdet-d0')
      driver.visualize(raw_images[0], detections[0])
      PIL.Image.fromarray(raw_images[0]).save(output_image_path)
  """

  def __init__(self,
               model_name: Text,
               ckpt_path: Text,
               batch_size: int = 1,
               use_xla: bool = False,
               min_score_thresh: float = None,
               max_boxes_to_draw: float = None,
               line_thickness: int = None,
               model_params: Dict[Text, Any] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      batch_size: batch size for inference.
      use_xla: Whether run with xla optimization.
      min_score_thresh: minimal score threshold for filtering predictions.
      max_boxes_to_draw: the maximum number of boxes per image.
      line_thickness: the line thickness for drawing boxes.
      model_params: model parameters for overriding the config.
    """
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.batch_size = batch_size

    self.params = hparams_config.get_detection_config(model_name).as_dict()

    if model_params:
      self.params.update(model_params)
    self.params.update(dict(is_training_bn=False))
    self.label_id_mapping = parse_label_id_mapping(
        self.params.get('label_id_mapping', None))

    self.signitures = None
    self.sess = None
    self.disable_pyfun = True
    self.use_xla = use_xla

    self.min_score_thresh = min_score_thresh or anchors.MIN_SCORE_THRESH
    self.max_boxes_to_draw = (
        max_boxes_to_draw or anchors.MAX_DETECTIONS_PER_IMAGE)
    self.line_thickness = line_thickness

  def __del__(self):
    if self.sess:
      self.sess.close()

  def _build_session(self):
    sess_config = tf.ConfigProto()
    if self.use_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_2)
    return tf.Session(config=sess_config)

  def build(self, params_override=None):
    """Build model and restore checkpoints."""
    params = copy.deepcopy(self.params)
    if params_override:
      params.update(params_override)

    if not self.sess:
      self.sess = self._build_session()
    with self.sess.graph.as_default():
      image_files = tf.placeholder(tf.string, name='image_files', shape=[None])
      raw_images = batch_image_files_decode(image_files)
      raw_images = tf.identity(raw_images, name='image_arrays')
      images, scales = batch_image_preprocess(raw_images, params['image_size'],
                                              self.batch_size)
      if params['data_format'] == 'channels_first':
        images = tf.transpose(images, [0, 3, 1, 2])
      class_outputs, box_outputs = build_model(self.model_name, images,
                                               **params)
      params.update(
          dict(batch_size=self.batch_size, disable_pyfun=self.disable_pyfun))
      detections = det_post_process(params, class_outputs, box_outputs, scales,
                                    self.min_score_thresh,
                                    self.max_boxes_to_draw)

      restore_ckpt(
          self.sess,
          self.ckpt_path,
          ema_decay=self.params['moving_average_decay'],
          export_ckpt=None)

    self.signitures = {
        'image_files': image_files,
        'image_arrays': raw_images,
        'prediction': detections,
    }
    return self.signitures

  def visualize(self, image, prediction, **kwargs):
    """Visualize prediction on image."""
    return visualize_image_prediction(
        image,
        prediction,
        disable_pyfun=self.disable_pyfun,
        label_id_mapping=self.label_id_mapping,
        **kwargs)

  def serve_files(self, image_files: List[Text]):
    """Serve a list of input image files.

    Args:
      image_files: a list of image files with shape [1] and type string.

    Returns:
      A list of detections.
    """
    if not self.sess:
      self.build()
    predictions = self.sess.run(
        self.signitures['prediction'],
        feed_dict={self.signitures['image_files']: image_files})
    return predictions

  def benchmark(self, image_arrays, trace_filename=None):
    """Benchmark inference latency/throughput.

    Args:
      image_arrays: a list of images in numpy array format.
      trace_filename: If None, specify the filename for saving trace.
    """
    if not self.sess:
      self.build()

    # init session
    self.sess.run(
        self.signitures['prediction'],
        feed_dict={self.signitures['image_arrays']: image_arrays})

    start = time.perf_counter()
    for _ in range(10):
      self.sess.run(
          self.signitures['prediction'],
          feed_dict={self.signitures['image_arrays']: image_arrays})
    end = time.perf_counter()
    inference_time = (end - start) / 10

    print('Per batch inference time: ', inference_time)
    print('FPS: ', self.batch_size / inference_time)

    if trace_filename:
      run_options = tf.RunOptions()
      run_options.trace_level = tf.RunOptions.FULL_TRACE
      run_metadata = tf.RunMetadata()
      self.sess.run(
          self.signitures['prediction'],
          feed_dict={self.signitures['image_arrays']: image_arrays},
          options=run_options,
          run_metadata=run_metadata)
      with tf.io.gfile.GFile(trace_filename, 'w') as trace_file:
        from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))

  def serve_images(self, image_arrays):
    """Serve a list of image arrays.

    Args:
      image_arrays: A list of image content with each image has shape [height,
        width, 3] and uint8 type.

    Returns:
      A list of detections.
    """
    if not self.sess:
      self.build()
    predictions = self.sess.run(
        self.signitures['prediction'],
        feed_dict={self.signitures['image_arrays']: image_arrays})
    return predictions

  def load(self, saved_model_dir_or_frozen_graph: Text):
    """Load the model using saved model or a frozen graph."""
    if not self.sess:
      self.sess = self._build_session()
    self.signitures = {
        'image_files': 'image_files:0',
        'image_arrays': 'image_arrays:0',
        'prediction': 'detections:0',
    }

    # Load saved model if it is a folder.
    if tf.io.gfile.isdir(saved_model_dir_or_frozen_graph):
      return tf.saved_model.load(self.sess, ['serve'],
                                 saved_model_dir_or_frozen_graph)

    # Load a frozen graph.
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(saved_model_dir_or_frozen_graph, 'rb') as f:
      graph_def.ParseFromString(f.read())
    return tf.import_graph_def(graph_def, name='')

  def freeze(self):
    """Freeze the graph."""
    output_names = [self.signitures['prediction'].op.name]
    graphdef = tf.graph_util.convert_variables_to_constants(
        self.sess, self.sess.graph_def, output_names)
    return graphdef

  def to_tflite(self, saved_model_dir):
    """Convert to tflite."""
    input_name = self.signitures['image_arrays'].op.name
    input_shapes = {input_name: [None, *self.params['image_size'], 3]}
    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir,
        input_arrays=[input_name],
        input_shapes=input_shapes,
        output_arrays=[self.signitures['prediction'].op.name])
    converter.experimental_new_converter = True
    supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_ops = supported_ops
    return converter.convert()

  def export(self, output_dir, frozen_pb=True, tflite_path=None):
    """Export a saved model."""
    signitures = self.signitures
    signature_def_map = {
        'serving_default':
            tf.saved_model.predict_signature_def(
                {signitures['image_arrays'].name: signitures['image_arrays']},
                {signitures['prediction'].name: signitures['prediction']}),
        'serving_base64':
            tf.saved_model.predict_signature_def(
                {signitures['image_files'].name: signitures['image_files']},
                {signitures['prediction'].name: signitures['prediction']}),
    }
    b = tf.saved_model.Builder(output_dir)
    b.add_meta_graph_and_variables(
        self.sess,
        tags=['serve'],
        signature_def_map=signature_def_map,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        clear_devices=True)
    b.save()
    logging.info('Model saved at %s', output_dir)

    # also save freeze pb file.
    if frozen_pb:
      graphdef = self.freeze()
      pb_path = os.path.join(output_dir, self.model_name + '_frozen.pb')
      tf.io.gfile.GFile(pb_path, 'wb').write(graphdef.SerializeToString())
      logging.info('Free graph saved at %s', pb_path)

    if tflite_path:
      ver = tf.__version__
      if ver < '2.2.0-dev20200501' or ('dev' not in ver and ver < '2.2.0-rc4'):
        raise ValueError('TFLite requires TF 2.2.0rc4 or laterr version.')
      tflite_model = self.to_tflite(output_dir)
      with tf.io.gfile.GFile(tflite_path, 'wb') as f:
        f.write(tflite_model)


class InferenceDriver(object):
  """A driver for doing batch inference.

  Example usage:

   driver = inference.InferenceDriver('efficientdet-d0', '/tmp/efficientdet-d0')
   driver.inference('/tmp/*.jpg', '/tmp/outputdir')

  """

  def __init__(self,
               model_name: Text,
               ckpt_path: Text,
               model_params: Dict[Text, Any] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      model_params: model parameters for overriding the config.
    """
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.params = hparams_config.get_detection_config(model_name).as_dict()
    if model_params:
      self.params.update(model_params)
    self.params.update(dict(is_training_bn=False))
    self.label_id_mapping = parse_label_id_mapping(
        self.params.get('label_id_mapping', None))

    self.disable_pyfun = True

  def inference(self, image_path_pattern: Text, output_dir: Text, **kwargs):
    """Read and preprocess input images.

    Args:
      image_path_pattern: Image file pattern such as /tmp/img*.jpg
      output_dir: the directory for output images. Output images will be named
        as 0.jpg, 1.jpg, ....
      **kwargs: extra parameters for for vistualization, such as
        min_score_thresh, max_boxes_to_draw, and line_thickness.

    Returns:
      Annotated image.
    """
    params = copy.deepcopy(self.params)
    with tf.Session() as sess:
      # Buid inputs and preprocessing.
      raw_images, images, scales = build_inputs(image_path_pattern,
                                                params['image_size'])
      if params['data_format'] == 'channels_first':
        images = tf.transpose(images, [0, 3, 1, 2])
      # Build model.
      class_outputs, box_outputs = build_model(self.model_name, images,
                                               **self.params)
      restore_ckpt(
          sess,
          self.ckpt_path,
          ema_decay=self.params['moving_average_decay'],
          export_ckpt=None)

      # for postprocessing.
      params.update(
          dict(batch_size=len(raw_images), disable_pyfun=self.disable_pyfun))

      # Build postprocessing.
      detections_batch = det_post_process(
          params,
          class_outputs,
          box_outputs,
          scales,
          min_score_thresh=kwargs.get('min_score_thresh',
                                      anchors.MIN_SCORE_THRESH),
          max_boxes_to_draw=kwargs.get('max_boxes_to_draw',
                                       anchors.MAX_DETECTIONS_PER_IMAGE))
      predictions = sess.run(detections_batch)
      # Visualize results.
      for i, prediction in enumerate(predictions):
        img = visualize_image_prediction(
            raw_images[i],
            prediction,
            disable_pyfun=self.disable_pyfun,
            label_id_mapping=self.label_id_mapping,
            **kwargs)
        output_image_path = os.path.join(output_dir, str(i) + '.jpg')
        Image.fromarray(img).save(output_image_path)
        logging.info('writing file to %s', output_image_path)

      return predictions
