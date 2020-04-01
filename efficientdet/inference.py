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
import os
from absl import logging
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from typing import Text, Dict, Any, List

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


def image_preprocess(image, image_size: int):
  """Preprocess image for inference.

  Args:
    image: input image, can be a tensor or a numpy arary.
    image_size: integer of image size.

  Returns:
    (image, scale): a tuple of processed image and its scale.
  """
  input_processor = dataloader.DetectionInputProcessor(image, image_size)
  input_processor.normalize_image()
  input_processor.set_scale_factors_to_output_size()
  image = input_processor.resize_and_crop_image()
  image_scale = input_processor.image_scale_to_original
  return image, image_scale


def build_inputs(image_path_pattern: Text, image_size: int):
  """Read and preprocess input images.

  Args:
    image_path_pattern: a path to indicate a single or multiple files.
    image_size: a single integer for image width and height.

  Returns:
    (raw_images, images, scales): raw images, processed images, and scales.
  """
  raw_images, images, scales = [], [], []
  for f in tf.io.gfile.glob(image_path_pattern):
    image = Image.open(f)
    raw_images.append(image)
    image, scale = image_preprocess(image, image_size)
    images.append(image)
    scales.append(scale)
  return raw_images, tf.stack(images), tf.stack(scales)


def build_model(model_name: Text, inputs: tf.Tensor, **kwargs):
  """Build model for a given model name.

  Args:
    model_name: the name of the model.
    inputs: an image tensor or a numpy array.
    **kwargs: extra parameters for model builder.

  Returns:
    (class_outputs, box_outputs): the outputs for class and box predictions.
    Each is a dictionary with key as feature level and value as predictions.
  """
  model_arch = det_model_fn.get_model_arch(model_name)
  class_outputs, box_outputs = model_arch(inputs, model_name, **kwargs)
  return class_outputs, box_outputs


def restore_ckpt(sess, ckpt_path, enable_ema=True, export_ckpt=None):
  """Restore variables from a given checkpoint.

  Args:
    sess: a tf session for restoring or exporting models.
    ckpt_path: the path of the checkpoint. Can be a file path or a folder path.
    enable_ema: whether reload ema values or not.
    export_ckpt: whether to export the restored model.
  """
  sess.run(tf.global_variables_initializer())
  if tf.io.gfile.isdir(ckpt_path):
    ckpt_path = tf.train.latest_checkpoint(ckpt_path)
  if enable_ema:
    ema = tf.train.ExponentialMovingAverage(decay=0.0)
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


def det_post_process(params: Dict[Any, Any],
                     cls_outputs: Dict[int, tf.Tensor],
                     box_outputs: Dict[int, tf.Tensor],
                     scales: List[float],
                     min_score_thresh=0.2,
                     max_boxes_to_draw=50):
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
      with each row representing [image_id, x, y, width, height, score, class].
  """
  # TODO(tanmingxing): refactor the code to make it more explicity.
  outputs = {
      'cls_outputs_all': [None],
      'box_outputs_all': [None],
      'indices_all': [None],
      'classes_all': [None]
  }
  det_model_fn.add_metric_fn_inputs(params, cls_outputs, box_outputs, outputs)

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
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw,
        disable_pyfun=params.get('disable_pyfun'))
    detections_batch.append(detections)
  return tf.stack(detections_batch, name='detections')


def visualize_image(image,
                    boxes,
                    classes,
                    scores,
                    id_mapping,
                    min_score_thresh=0.2,
                    max_boxes_to_draw=50,
                    line_thickness=4,
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
    predictions = driver.serve_files(imgs)
    for i in imgs:
      driver.visualize(imgs[0], predictions[0])

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
               image_size: int = None,
               batch_size: int = 1,
               num_classes: int = None,
               label_id_mapping: Dict[int, Text] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      image_size: user specified image size. If None, use the default image size
        defined by model_name.
      batch_size: batch size for inference.
      num_classes: number of classes. If None, use the default COCO classes.
      label_id_mapping: a dictionary from id to name. If None, use the default
        coco_id_mapping (with 90 classes).
    """
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.batch_size = batch_size
    self.label_id_mapping = label_id_mapping or coco_id_mapping

    self.params = hparams_config.get_detection_config(self.model_name).as_dict()
    self.params.update(dict(is_training_bn=False, use_bfloat16=False))
    if image_size:
      self.params.update(dict(image_size=image_size))
    if num_classes:
      self.params.update(dict(num_classes=num_classes))

    self.signitures = None
    self.sess = None
    self.disable_pyfun = True

  def build(self,
            params_override=None,
            min_score_thresh=0.2,
            max_boxes_to_draw=50):
    """Build model and restore checkpoints."""
    params = copy.deepcopy(self.params)
    if params_override:
      params.update(params_override)

    image_files = tf.placeholder(tf.string, name='image_files', shape=(None))
    image_size = params['image_size']
    raw_images = []
    for i in range(self.batch_size):
      image = tf.io.decode_image(image_files[i])
      image.set_shape([None, None, None])
      raw_images.append(image)
    raw_images = tf.stack(raw_images, name='image_arrays')

    scales, images = [], []
    for i in range(self.batch_size):
      image, scale = image_preprocess(raw_images[i], image_size)
      scales.append(scale)
      images.append(image)
    scales = tf.stack(scales)
    images = tf.stack(images)
    class_outputs, box_outputs = build_model(self.model_name, images, **params)
    params.update(
        dict(batch_size=self.batch_size, disable_pyfun=self.disable_pyfun))
    detections = det_post_process(
        params,
        class_outputs,
        box_outputs,
        scales,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw)

    if not self.sess:
      self.sess = tf.Session()
    restore_ckpt(self.sess, self.ckpt_path, enable_ema=True, export_ckpt=None)

    self.signitures = {
        'image_files': image_files,
        'image_arrays': raw_images,
        'prediction': detections,
    }
    return self.signitures

  def visualize(self, image, predictions, **kwargs):
    """Visualize predictions on image.

    Args:
      image: Image content in shape of [height, width, 3].
      predictions: a list of vector, with each vector has the format of
        [image_id, x, y, width, height, score, class].
      **kwargs: extra parameters for for vistualization, such as
        min_score_thresh, max_boxes_to_draw, and line_thickness.

    Returns:
      annotated image.
    """
    boxes = predictions[:, 1:5]
    classes = predictions[:, 6].astype(int)
    scores = predictions[:, 5]

    # This is not needed if disable_pyfun=True
    # convert [x, y, width, height] to [ymin, xmin, ymax, xmax]
    # TODO(tanmingxing): make this convertion more efficient.
    if not self.disable_pyfun:
      boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

    boxes[:, 2:4] += boxes[:, 0:2]
    return visualize_image(image, boxes, classes, scores, self.label_id_mapping,
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

  def export(self, output_dir):
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


class InferenceDriver(object):
  """A driver for doing batch inference.

  Example usage:

   driver = inference.InferenceDriver('efficientdet-d0', '/tmp/efficientdet-d0')
   driver.inference('/tmp/*.jpg', '/tmp/outputdir')

  """

  def __init__(self,
               model_name: Text,
               ckpt_path: Text,
               image_size: int = None,
               num_classes: int = None,
               label_id_mapping: Dict[int, Text] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      image_size: user specified image size. If None, use the default image size
        defined by model_name.
      num_classes: number of classes. If None, use the default COCO classes.
      label_id_mapping: a dictionary from id to name. If None, use the default
        coco_id_mapping (with 90 classes).
    """
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.label_id_mapping = label_id_mapping or coco_id_mapping

    self.params = hparams_config.get_detection_config(self.model_name).as_dict()
    self.params.update(dict(is_training_bn=False, use_bfloat16=False))
    if image_size:
      self.params.update(dict(image_size=image_size))
    if num_classes:
      self.params.update(dict(num_classes=num_classes))
    self.disable_pyfun = False

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

      # Build model.
      class_outputs, box_outputs = build_model(self.model_name, images,
                                               **self.params)
      restore_ckpt(sess, self.ckpt_path, enable_ema=True, export_ckpt=None)
      # for postprocessing.
      params.update(
          dict(batch_size=len(raw_images), disable_pyfun=self.disable_pyfun))

      # Build postprocessing.
      detections_batch = det_post_process(
          params,
          class_outputs,
          box_outputs,
          scales,
          min_score_thresh=kwargs.get('min_score_thresh', 0.2),
          max_boxes_to_draw=kwargs.get('max_boxes_to_draw', 50))
      outputs_np = sess.run(detections_batch)
      # Visualize results.
      for i, output_np in enumerate(outputs_np):
        # output_np has format [image_id, y, x, height, width, score, class]
        boxes = output_np[:, 1:5]
        classes = output_np[:, 6].astype(int)
        scores = output_np[:, 5]

        # This is not needed if disable_pyfun=True
        # convert [x, y, width, height] to [ymin, xmin, ymax, xmax]
        # TODO(tanmingxing): make this convertion more efficient.
        if not self.disable_pyfun:
          boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

        boxes[:, 2:4] += boxes[:, 0:2]
        img = visualize_image(raw_images[i], boxes, classes, scores,
                              self.label_id_mapping, **kwargs)
        output_image_path = os.path.join(output_dir, str(i) + '.jpg')
        Image.fromarray(img).save(output_image_path)
        logging.info('writing file to %s', output_image_path)

      return outputs_np
