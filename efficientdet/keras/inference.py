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
import copy
import os
import time
from typing import Text, Dict, Any, List
from absl import logging
import numpy as np
import tensorflow as tf

import hparams_config
import utils
from keras import util_keras
from keras import label_util
from keras import postprocess
from visualize import vis_utils
from keras.efficientdet_keras import EfficientDetModel


def det_post_process(params: Dict[Any, Any], cls_outputs: Dict[int, tf.Tensor],
                     box_outputs: Dict[int, tf.Tensor], scales: List[float]):
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

  Returns:
    detections_batch: a batch of detection results. Each detection is a tensor
      with each row as [image_id, ymin, xmin, ymax, xmax, score, class].
  """
  if params.get('combined_nms', None):
    # Use combined version for dynamic batch size.
    nms_boxes, nms_scores, nms_classes, _ = postprocess.postprocess_combined(
        params, cls_outputs, box_outputs, scales)
  else:
    nms_boxes, nms_scores, nms_classes, _ = postprocess.postprocess_global(
        params, cls_outputs, box_outputs, scales)

  batch_size = tf.shape(cls_outputs[params['min_level']])[0]
  img_ids = tf.expand_dims(
      tf.cast(tf.range(0, batch_size), nms_scores.dtype), -1)
  detections = [
      img_ids * tf.ones_like(nms_scores),
      nms_boxes[:, :, 0],
      nms_boxes[:, :, 1],
      nms_boxes[:, :, 2],
      nms_boxes[:, :, 3],
      nms_scores,
      nms_classes,
  ]
  return tf.stack(detections, axis=-1, name='detections')


def visualize_image(image,
                    boxes,
                    classes,
                    scores,
                    label_map=None,
                    min_score_thresh=0.01,
                    max_boxes_to_draw=1000,
                    line_thickness=2,
                    **kwargs):
  """Visualizes a given image.

  Args:
    image: a image with shape [H, W, C].
    boxes: a box prediction with shape [N, 4] ordered [ymin, xmin, ymax, xmax].
    classes: a class prediction with shape [N].
    scores: A list of float value with shape [N].
    label_map: a dictionary from class id to name.
    min_score_thresh: minimal score for showing. If claass probability is below
      this threshold, then the object will not show up.
    max_boxes_to_draw: maximum bounding box to draw.
    line_thickness: how thick is the bounding box line.
    **kwargs: extra parameters.

  Returns:
    output_image: an output image with annotated boxes and classes.
  """
  label_map = label_util.get_label_map(label_map or 'coco')
  category_index = {k: {'id': k, 'name': label_map[k]} for k in label_map}
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


class ExportModel(tf.Module):

  def __init__(self, model):
    super().__init__()
    self.model = model

  @tf.function
  def __call__(self, imgs):
    return self.model(imgs, training=False, post_mode='global')


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
      boxes, scores, classes, _ = tf.nest.map_structure(np.array, predictions)
      driver.visualize(m, boxes[0], scores[0], classes[0])
      # m is the new image with annotated boxes.

  Example 2. Serving batch image contents:

    imgs = []
    for f in ['/tmp/1.jpg', '/tmp/2.jpg']:
      imgs.append(np.array(Image.open(f)))

    driver = inference.ServingDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=len(imgs))
    driver.build()
    predictions = driver.serve_images(imgs)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, predictions)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], boxes[i], scores[i], classes[i])

  Example 3: another way is to use SavedModel:

    # step1: export a model.
    driver = inference.ServingDriver('efficientdet-d0', '/tmp/efficientdet-d0')
    driver.build()
    driver.export('/tmp/saved_model_path')

    # step2: Serve a model.
    driver.load(self.saved_model_dir)
    raw_images = []
    for f in tf.io.gfile.glob('/tmp/images/*.jpg'):
      raw_images.append(np.array(PIL.Image.open(f)))
    detections = driver.serve_images(raw_images)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], boxes[i], scores[i], classes[i])
  """

  def __init__(self,
               model_name: Text,
               ckpt_path: Text,
               batch_size: int = 1,
               min_score_thresh: float = None,
               max_boxes_to_draw: float = None,
               model_params: Dict[Text, Any] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      batch_size: batch size for inference.
      use_xla: Whether run with xla optimization.
      min_score_thresh: minimal score threshold for filtering predictions.
      max_boxes_to_draw: the maximum number of boxes per image.
      model_params: model parameters for overriding the config.
    """
    super().__init__()
    self.model_name = model_name
    self.ckpt_path = ckpt_path
    self.batch_size = batch_size

    self.params = hparams_config.get_detection_config(model_name).as_dict()

    if model_params:
      self.params.update(model_params)
    self.params.update(dict(is_training_bn=False))
    self.label_map = self.params.get('label_map', None)

    self.model = None

    self.min_score_thresh = min_score_thresh
    self.max_boxes_to_draw = max_boxes_to_draw
    mixed_precision = self.params.get('mixed_precision', None)
    precision = utils.get_precision(
        self.params.get('strategy', None), mixed_precision)
    policy = tf.keras.mixed_precision.experimental.Policy(precision)
    tf.keras.mixed_precision.experimental.set_policy(policy)

  def build(self, params_override=None):
    """Build model and restore checkpoints."""
    params = copy.deepcopy(self.params)
    if params_override:
      params.update(params_override)
    config = hparams_config.get_efficientdet_config(self.model_name)
    config.override(params)
    self.model = EfficientDetModel(config=config)
    image_size = utils.parse_image_size(params['image_size'])
    self.model.build((self.batch_size, *image_size, 3))
    util_keras.restore_ckpt(self.model, self.ckpt_path, params['moving_average_decay'])

  def visualize(self, image, boxes, classes, scores, **kwargs):
    """Visualize prediction on image."""
    return visualize_image(image, boxes, classes.astype(int), scores,
                           self.label_map, **kwargs)

  def benchmark(self, image_arrays, bm_runs=10, trace_filename=None):
    """Benchmark inference latency/throughput.

    Args:
      image_arrays: a list of images in numpy array format.
      bm_runs: Number of benchmark runs.
      trace_filename: If None, specify the filename for saving trace.
    """
    if not self.model:
      self.build()

    @tf.function
    def test_func(image_arrays):
      return self.model(image_arrays)

    test_func(image_arrays)

    start = time.perf_counter()
    for _ in range(bm_runs):
      test_func(image_arrays)
    end = time.perf_counter()
    inference_time = (end - start) / bm_runs

    print('Per batch inference time: ', inference_time)
    print('FPS: ', self.batch_size / inference_time)

    if trace_filename:
      options = tf.profiler.experimental.ProfilerOptions()
      tf.profiler.experimental.start(trace_filename, options)
      test_func(image_arrays)
      tf.profiler.experimental.stop()

  def serve_files(self, image_files: List[Text]):
    """Serve a list of input image files.

    Args:
      image_files: a list of image files with shape [1] and type string.

    Returns:
      A list of detections.
    """

    def _map_fn(file):
      image = tf.image.decode_image(file)
      image.set_shape((None, None, 3))
      return image

    images = tf.map_fn(_map_fn, image_files, dtype=tf.uint8)
    return self.serve_images(images)

  def serve_images(self, image_arrays):
    """Serve a list of image arrays.

    Args:
      image_arrays: A list of image content with each image has shape [height,
        width, 3] and uint8 type.

    Returns:
      A list of detections.
    """
    if not self.model:
      self.build()
    return self.model(image_arrays)

  def load(self, saved_model_dir_or_frozen_graph: Text):
    """Load the model using saved model or a frozen graph."""

    # Load saved model if it is a folder.
    if tf.saved_model.contains_saved_model(saved_model_dir_or_frozen_graph):
      self.model = tf.saved_model.load(saved_model_dir_or_frozen_graph)
      return
    # Load a frozen graph.
    graph_def = tf.Graph().as_graph_def()
    with tf.io.gfile.GFile(saved_model_dir_or_frozen_graph, 'rb') as f:
      graph_def.ParseFromString(f.read())
    self.model = tf.import_graph_def(graph_def, name='')

  def freeze(self, func):
    """Freeze the graph."""
    from tensorflow.python.framework.convert_to_constants \
      import convert_variables_to_constants_v2_as_graph # pylint: disable=g-direct-tensorflow-import
    _, graphdef = convert_variables_to_constants_v2_as_graph(func)
    return graphdef

  def export(self,
             output_dir: Text,
             tflite_path: Text = None,
             tensorrt: Text = None):
    """Export a saved model, frozen graph, and potential tflite/tensorrt model.

    Args:
      output_dir: the output folder for saved model.
      tflite_path: the path for saved tflite file.
      tensorrt: If not None, must be {'FP32', 'FP16', 'INT8'}.
    """
    if not self.model:
      self.build()
    export_model = ExportModel(self.model)
    tf.saved_model.save(
        export_model,
        output_dir,
        signatures=export_model.__call__.get_concrete_function(
            tf.TensorSpec(
                shape=[None, None, None, 3], dtype=tf.uint8, name="images")))
    logging.info('Model saved at %s', output_dir)

    # also save freeze pb file.
    graphdef = self.freeze(
        export_model.__call__.get_concrete_function(
            tf.TensorSpec(
                shape=[None, None, None, 3], dtype=tf.uint8, name="images")))
    proto_path = tf.io.write_graph(
        graphdef, output_dir, self.model_name + '_frozen.pb', as_text=False)
    logging.info('Frozen graph saved at %s', proto_path)

    if tflite_path:
      converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
      converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
      tflite_model = converter.convert()

      tf.io.gfile.GFile(tflite_path, 'wb').write(tflite_model)
      logging.info('TFLite is saved at %s', tflite_path)

    if tensorrt:
      trt_path = os.path.join(output_dir, 'tensorrt_' + tensorrt.lower())
      conversion_params = tf.experimental.tensorrt.ConversionParams(
          precision_mode=tensorrt.upper())
      converter = tf.experimental.tensorrt.Converter(
          output_dir, conversion_params=conversion_params)
      converter.convert()
      converter.save(trt_path)
      logging.info('TensorRT model is saved at %s', trt_path)
