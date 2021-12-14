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
import multiprocessing
import os
import time
from typing import Text, Dict, Any, Optional
from absl import logging
import numpy as np
import tensorflow as tf

import dataloader
import hparams_config
import utils
from tf2 import efficientdet_keras
from tf2 import label_util
from tf2 import util_keras
from tf2 import postprocess
from visualize import vis_utils

DEFAULT_SCALE, DEFAULT_ZERO_POINT = 0, 0


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
  img = vis_utils.visualize_boxes_and_labels_on_image_array(
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
  """Model to be exported as SavedModel/TFLite format."""

  def __init__(self, model):
    super().__init__()
    self.model = model
    self.model.optimizer = None
    self.model._delete_tracking('optimizer')

  @tf.function
  def predict(self, imgs):
    return tf.nest.flatten(
        self.model(imgs, training=False, pre_mode=None, post_mode=None))

  @tf.function
  def tflite(self, imgs):
    return self.model(imgs, training=False, pre_mode=None, post_mode='tflite')

  @tf.function
  def __call__(self, imgs):
    return self.model(imgs, training=False)


class ServingDriver:
  """A driver for serving single or batch images.

  This driver supports serving with image files or arrays, with configurable
  batch size.

  Example 1. Serving batch image contents:

    imgs = []
    for f in ['/tmp/1.jpg', '/tmp/2.jpg']:
      imgs.append(np.array(Image.open(f)))

    driver = infer_lib.KerasDriver(
      'efficientdet-d0', '/tmp/efficientdet-d0', batch_size=len(imgs))
    predictions = driver.serve(imgs)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, predictions)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], boxes[i], scores[i], classes[i])

  Example 2: another way is to use SavedModel:

    # step1: export a model.
    driver = infer_lib.KerasDriver('efficientdet-d0', '/tmp/efficientdet-d0')
    driver.export('/tmp/saved_model_path')

    # step2: Serve a model.
    driver = infer_lib.SavedModelDriver('efficientdet-d0', '/tmp/efficientdet-d0')
    raw_images = []
    for f in tf.io.gfile.glob('/tmp/images/*.jpg'):
      raw_images.append(np.array(PIL.Image.open(f)))
    detections = driver.serve(raw_images)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections)
    for i in range(len(imgs)):
      driver.visualize(imgs[i], boxes[i], scores[i], classes[i])
  """

  @classmethod
  def create(cls, model_dir, debug, saved_model_dir, *args, **kwargs):
    if saved_model_dir:
      if saved_model_dir.endswith('tflite'):
        driver = TfliteDriver(saved_model_dir, *args, **kwargs)
      else:
        driver = SavedModelDriver(saved_model_dir, *args, **kwargs)
    else:
      driver = KerasDriver(model_dir, debug, *args, **kwargs)
    return driver

  def __init__(self,
               model_name: Text,
               batch_size: int = 1,
               only_network: bool = False,
               model_params: Optional[Dict[Text, Any]] = None):
    """Initialize the inference driver.

    Args:
      model_name: target model name, such as efficientdet-d0.
      batch_size: batch size for inference.
      only_network: only use the network without pre/post processing.
      model_params: model parameters for overriding the config.
    """
    super().__init__()
    self.model_name = model_name
    self.batch_size = batch_size
    self.only_network = only_network

    self.params = hparams_config.get_detection_config(model_name).as_dict()

    if model_params:
      self.params.update(model_params)
    self.params.update(dict(is_training_bn=False))
    self.label_map = self.params.get('label_map', None)

    self._model = None

  def visualize(self, image, boxes, classes, scores, **kwargs):
    """Visualize prediction on image."""
    return visualize_image(image, boxes, classes.astype(int), scores,
                           self.label_map, **kwargs)

  def _benchmark(self, image_arrays, test_func, bm_runs=10,
                 trace_filename=None):
    """Benchmark inference latency/throughput.

    Args:
      image_arrays: a list of images in numpy array format.
      bm_runs: Number of benchmark runs.
      trace_filename: If None, specify the filename for saving trace.
    """
    for _ in range(3):  # warmup 3 runs.
      test_func(image_arrays)

    start = time.perf_counter()
    for _ in range(bm_runs):
      test_func(image_arrays)
    end = time.perf_counter()
    inference_time = (end - start) / bm_runs

    print('Per batch inference time: ', inference_time)
    print('FPS: ', (self.batch_size or 1) / inference_time)

    if trace_filename:
      options = tf.profiler.experimental.ProfilerOptions()
      tf.profiler.experimental.start(trace_filename, options)
      test_func(image_arrays)
      tf.profiler.experimental.stop()

  def predict(self, image_arrays):
    """Feed image arrays to TF/TFLite model without extra preprocess or postprocess.

    Args:
      image_arrays: Image tensors with dtype tf.float32 or tf.uint8.

    Returns:
      Model outputs.
    """
    raise NotImplemented

  def _preprocess(self, image_arrays):

    def map_fn(image):
      input_processor = dataloader.DetectionInputProcessor(
          image, self.params['image_size'])
      input_processor.normalize_image(self.params['mean_rgb'],
                                      self.params['stddev_rgb'])
      input_processor.set_scale_factors_to_output_size()
      image = input_processor.resize_and_crop_image()
      image_scale = input_processor.image_scale_to_original
      return image, image_scale

    if self.batch_size:
      outputs = [map_fn(image_arrays[i]) for i in range(self.batch_size)]
      return [tf.stop_gradient(tf.stack(y)) for y in zip(*outputs)]

    return tf.vectorized_map(map_fn, image_arrays)

  def _postprocess(self, outputs, scales):
    det_outputs = postprocess.postprocess_global(self.params, outputs[0],
                                                 outputs[1], scales)
    return det_outputs + tuple(outputs[2:])

  def serve(self, image_arrays):
    """Serve a list of image arrays.

    Args:
      image_arrays: A list of image content with each image has shape [height,
        width, 3] and uint8 type.

    Returns:
      A list of detections.
    """
    raise NotImplemented

  def _get_model_and_spec(self, tflite=None):
    """Get model instance and export spec."""
    export_model = ExportModel(self.model)
    if self.only_network or tflite:
      batch_size = self.batch_size
      if tflite and not batch_size:
        batch_size = 1
      image_size = utils.parse_image_size(self.params['image_size'])
      spec = tf.TensorSpec(
          shape=[batch_size, *image_size, 3], dtype=tf.float32, name='images')
      return export_model, spec
    spec = tf.TensorSpec(
        shape=[self.batch_size, None, None, 3], dtype=tf.uint8, name='images')
    return export_model, spec


class SavedModelDriver(ServingDriver):

  def __init__(self, saved_model_dir_or_frozen_graph: Text, *args, **kwargs):
    """ Initialize the SavedModelDriver.

    Args:
      saved_model_dir_or_frozen_graph: Saved model or frozen graph path.
    """
    # Load saved model if it is a folder.
    super().__init__(*args, **kwargs)
    if tf.saved_model.contains_saved_model(saved_model_dir_or_frozen_graph):
      self.model = tf.saved_model.load(saved_model_dir_or_frozen_graph)
      return

    # Load a frozen graph.
    def wrap_frozen_graph(graph_def, inputs, outputs):
      # https://www.tensorflow.org/guide/migrate
      imports_graph_def_fn = lambda: tf.import_graph_def(graph_def, name='')
      wrapped_import = tf.compat.v1.wrap_function(imports_graph_def_fn, [])
      import_graph = wrapped_import.graph
      return wrapped_import.prune(
          tf.nest.map_structure(import_graph.as_graph_element, inputs),
          tf.nest.map_structure(import_graph.as_graph_element, outputs))

    graph_def = tf.Graph().as_graph_def()
    with tf.io.gfile.GFile(saved_model_dir_or_frozen_graph, 'rb') as f:
      graph_def.ParseFromString(f.read())

    self.model = wrap_frozen_graph(
        graph_def,
        inputs='images:0',
        outputs=['Identity:0', 'Identity_1:0', 'Identity_2:0', 'Identity_3:0'])

  def benchmark(self, image_arrays, bm_runs=10, trace_filename=None):
    self._benchmark(image_arrays, self.predict, bm_runs, trace_filename)

  def serve(self, image_arrays):
    if self.only_network:
      image_arrays, scales = self._preprocess(image_arrays)
    outputs = self.predict(image_arrays)
    if self.only_network:
      outputs = self._postprocess(outputs, scales)
    return outputs

  def predict(self, image_arrays):
    if self.only_network:
      outputs = tuple(self.model.predict(image_arrays))
    else:
      outputs = self.model(image_arrays)
    return outputs


class TfliteDriver(ServingDriver):

  def __init__(self, tflite_path: Text, *args, **kwargs):
    """ Initialize the inference driver.

    Args:
      tflite_path: tensorflow lite model path.
    """
    super().__init__(*args, **kwargs)
    self.model = tf.lite.Interpreter(
        tflite_path, num_threads=multiprocessing.cpu_count())

  def benchmark(self, image_arrays, bm_runs=10, trace_filename=None):
    image_arrays = np.array(image_arrays)
    self._benchmark(image_arrays, self.predict, bm_runs, trace_filename)

  def serve(self, image_arrays):
    image_arrays, scales = self._preprocess(image_arrays)
    outputs = self.predict(image_arrays)
    if self.only_network:
      outputs = [outputs[:5], outputs[5:]]
      boxes, scores, classes, val_indices = self._postprocess(outputs, scales)
    else:
      val_indices, scores, classes, boxes = outputs
      height, width = utils.parse_image_size(self.params['image_size'])
      normalize_factor = tf.constant([height, width, height, width],
                                     dtype=tf.float32)
      boxes *= normalize_factor * scales
      classes += postprocess.CLASS_OFFSET
    return boxes, scores, classes, val_indices

  def predict(self, image_arrays):
    input_details = self.model.get_input_details()
    output_details = self.model.get_output_details()
    input_detail = input_details[0]
    if input_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
      scale, zero_point = input_detail['quantization']
      image_arrays = image_arrays / scale + zero_point
      image_arrays = tf.cast(image_arrays, dtype=input_detail['dtype'])

    signature = list(self.model.get_signature_list().keys())[0]
    infer_fn = self.model.get_signature_runner(signature)
    outputs = infer_fn(images=image_arrays)

    def get_output(output_detail, output_tensor):
      if output_detail['quantization'] != (DEFAULT_SCALE, DEFAULT_ZERO_POINT):
        # Dequantize the output
        scale, zero_point = output_detail['quantization']
        output_tensor = output_tensor.astype(np.float32)
        output_tensor = (output_tensor - zero_point) * scale
      return output_tensor

    outputs = [
        get_output(output_detail, output)
        for output_detail, output in zip(output_details, outputs.values())
    ]
    if self.only_network:
      outputs = tuple(outputs)
    return outputs


class KerasDriver(ServingDriver):

  def __init__(self, ckpt_path, debug, *args, **kwargs):
    """ Initialize the inference driver.

    Args:
      ckpt_path: checkpoint path, such as /tmp/efficientdet-d0/.
      debug: bool, if true, run in debug mode.
    """
    super().__init__(*args, **kwargs)
    params = copy.deepcopy(self.params)
    config = hparams_config.get_efficientdet_config(self.model_name)
    config.override(params)
    precision = utils.get_precision(config.strategy, config.mixed_precision)
    policy = tf.keras.mixed_precision.Policy(precision)
    tf.keras.mixed_precision.set_global_policy(policy)
    self.model = efficientdet_keras.EfficientDetModel(config=config)
    image_size = utils.parse_image_size(config.image_size)
    self.model.build((self.batch_size, *image_size, 3))
    util_keras.restore_ckpt(
        self.model, ckpt_path, config.moving_average_decay, skip_mismatch=False)
    self.debug = debug
    if debug:
      tf.config.run_functions_eagerly(debug)

  def serve(self, image_arrays):
    if self.only_network:
      image_arrays, scales = self._preprocess(image_arrays)
    outputs = self.predict(image_arrays)
    if self.only_network:
      outputs = self._postprocess(outputs, scales)
    return outputs

  def predict(self, image_arrays):
    if self.only_network:
      outputs = tuple(self.model(image_arrays, pre_mode=None, post_mode=None))  # pylint: disable=not-callable
    else:
      outputs = self.model(image_arrays)  # pylint: disable=not-callable
    return outputs

  def freeze(self, func):
    """Freeze the graph."""
    # pylint: disable=g-import-not-at-top,disable=g-direct-tensorflow-import
    from tensorflow.python.framework.convert_to_constants \
      import convert_variables_to_constants_v2_as_graph
    _, graphdef = convert_variables_to_constants_v2_as_graph(func)
    return graphdef

  def _create_representative_dataset(self, file_pattern, num_calibration_steps):
    config = hparams_config.get_efficientdet_config(self.model_name)
    config.override(self.params)
    ds = dataloader.InputReader(
        file_pattern,
        is_training=False,
        max_instances_per_image=config.max_instances_per_image)(
            config, batch_size=self.batch_size)

    def representative_dataset_gen():
      for image, _ in ds.take(num_calibration_steps):
        yield [image]

    return representative_dataset_gen

  def benchmark(self, image_arrays, bm_runs=10, trace_filename=None):
    _, spec = self._get_model_and_spec()

    @tf.function(input_signature=[spec])
    def test_func(image_arrays):
      return self.predict(image_arrays)

    self._benchmark(image_arrays, test_func, bm_runs, trace_filename)

  def export(self,
             output_dir: Optional[Text] = None,
             tensorrt: Optional[Text] = None,
             tflite: Optional[Text] = None,
             file_pattern: Optional[Text] = None,
             num_calibration_steps: int = 500):
    """Export a saved model, frozen graph, and potential tflite/tensorrt model.

    Args:
      output_dir: the output folder for saved model.
      tensorrt: If not None, must be {'FP32', 'FP16', 'INT8'}.
      tflite: Type for post-training quantization.
      file_pattern: Glob for tfrecords, e.g. coco/val-*.tfrecord.
      num_calibration_steps: Number of post-training quantization calibration
        steps to run.
    """
    export_model, input_spec = self._get_model_and_spec(False)
    _, tflite_input_spec = self._get_model_and_spec(True)

    if output_dir:
      tf.saved_model.save(
          export_model,
          output_dir,
          signatures={
              tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  export_model.__call__.get_concrete_function(input_spec),
              'tflite':
                  export_model.tflite.get_concrete_function(tflite_input_spec),
              'predict':
                  export_model.predict.get_concrete_function(tflite_input_spec)
          },
          options=tf.saved_model.SaveOptions(
              function_aliases={'serve': export_model.__call__}))
      logging.info('Model saved at %s', output_dir)

      # also save freeze pb file.
      if self.only_network:
        call_fn = export_model.predict
      else:
        call_fn = export_model.__call__
      graphdef = self.freeze(call_fn.get_concrete_function(input_spec))
      proto_path = tf.io.write_graph(
          graphdef, output_dir, self.model_name + '_frozen.pb', as_text=False)
      logging.info('Frozen graph saved at %s', proto_path)

    if tflite:
      input_spec = tflite_input_spec
      # from_saved_model supports advanced converter features like op fusing.
      signature_key = 'predict' if self.only_network else 'tflite'
      converter = tf.lite.TFLiteConverter.from_saved_model(
          output_dir, [signature_key])
      if tflite == 'FP32':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float32]
      elif tflite == 'FP16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
      elif tflite == 'INT8':
        # Enables MLIR-based post-training quantization.
        converter.experimental_new_quantizer = True
        if file_pattern:
          representative_dataset_gen = self._create_representative_dataset(
              file_pattern, num_calibration_steps)
        elif self.debug:  # Used for debugging, can remove later.
          logging.warning(
              'Use real representative dataset instead of fake ones.')
          num_calibration_steps = 10

          def representative_dataset_gen():  # rewrite this for real data.
            for _ in range(num_calibration_steps):
              yield [tf.ones(input_spec.shape, dtype=input_spec.dtype)]
        else:
          raise ValueError(
              'Please specific --file_pattern before export INT8 tflite model.'
              ' If you just want to debug the export process, add --debug')

        converter.representative_dataset = representative_dataset_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.uint8
        # TFLite's custom NMS op isn't supported by post-training quant,
        # so we add TFLITE_BUILTINS as well.
        supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.target_spec.supported_ops = supported_ops

      else:
        raise ValueError(f'Invalid tflite {tflite}: must be FP32, FP16, INT8.')

      tflite_path = os.path.join(output_dir, tflite.lower() + '.tflite')
      tflite_model = converter.convert()
      tf.io.gfile.GFile(tflite_path, 'wb').write(tflite_model)
      logging.info('TFLite is saved at %s', tflite_path)

    if tensorrt:
      trt_path = os.path.join(output_dir, 'tensorrt_' + tensorrt.lower())
      conversion_params = tf.experimental.tensorrt.ConversionParams(
          max_workspace_size_bytes=(2 << 20),
          maximum_cached_engines=1,
          precision_mode=tensorrt.upper())
      converter = tf.experimental.tensorrt.Converter(
          output_dir, conversion_params=conversion_params)
      if tensorrt == "INT8" and file_pattern:
        representative_dataset_gen = self._create_representative_dataset(
            file_pattern, num_calibration_steps)
      else:
        representative_dataset_gen = None
      converter.convert(calibration_input_fn=representative_dataset_gen)
      converter.save(trt_path)
      logging.info('TensorRT model is saved at %s', trt_path)
