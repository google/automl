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
r"""Tool to inspect a model."""

from __future__ import absolute_import
from __future__ import division
# gtype import
from __future__ import print_function

import os
import time

from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from typing import Text, Tuple, List

import det_model_fn
import hparams_config
import inference
import utils


flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model.')
flags.DEFINE_string('logdir', '/tmp/deff/', 'log directory.')
flags.DEFINE_string('runmode', 'dry', 'Run mode: {freeze, bm, dry}')
flags.DEFINE_string('trace_filename', None, 'Trace file name.')
flags.DEFINE_integer('num_classes', 90, 'Number of classes.')
flags.DEFINE_string('input_image_size', None, 'Size of input image. Enter a'
                    'single integer if the image height is equal to the width;'
                    'Otherwise, enter two integers seprated by a "x".'
                    'e.g. "1280x640" if width=1280 and height=640.')
flags.DEFINE_integer('threads', 0, 'Number of threads.')
flags.DEFINE_integer('bm_runs', 10, 'Number of benchmark runs.')
flags.DEFINE_string('tensorrt', None, 'TensorRT mode: {None, FP32, FP16, INT8}')
flags.DEFINE_bool('use_batch_nms', False, 'use tf.image.combined_non_max_suppression to do nms')
flags.DEFINE_bool('delete_logdir', True, 'Whether to delete logdir.')
flags.DEFINE_bool('freeze', False, 'Freeze graph.')
flags.DEFINE_bool('xla', False, 'Run with xla optimization.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference.')

flags.DEFINE_string('ckpt_path', None, 'checkpoint dir used for eval.')
flags.DEFINE_string('export_ckpt', None, 'Path for exporting new models.')
flags.DEFINE_bool('enable_ema', True, 'Use ema variables for eval.')
flags.DEFINE_string('data_format', None, 'data format, e.g., channel_last.')

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')

# For video.
flags.DEFINE_string('input_video', None, 'Input video path for inference.')
flags.DEFINE_string('output_video', None,
                    'Output video path. If None, play it online instead.')

# For visualization.
flags.DEFINE_integer('line_thickness', None, 'Line thickness for box.')
flags.DEFINE_integer('max_boxes_to_draw', None, 'Max number of boxes to draw.')
flags.DEFINE_float('min_score_thresh', None, 'Score threshold to show box.')

# For saved model.
flags.DEFINE_string('saved_model_dir', '/tmp/saved_model',
                    'Folder path for saved model.')

FLAGS = flags.FLAGS


class ModelInspector(object):
  """A simple helper class for inspecting a model."""

  def __init__(self,
               model_name: Text,
               image_size: Text,
               num_classes: int,
               logdir: Text,
               tensorrt: Text = False,
               use_xla: bool = False,
               ckpt_path: Text = None,
               enable_ema: bool = True,
               export_ckpt: Text = None,
               saved_model_dir: Text = None,
               data_format: Text = None,
               batch_size: int = 1,
               use_batch_nms=False):
    self.model_name = model_name
    self.model_params = hparams_config.get_detection_config(model_name)
    self.logdir = logdir
    self.tensorrt = tensorrt
    self.use_xla = use_xla
    self.ckpt_path = ckpt_path
    self.enable_ema = enable_ema
    self.export_ckpt = export_ckpt
    self.saved_model_dir = saved_model_dir
    self.use_batch_nms = use_batch_nms

    if image_size is None:
      image_size = hparams_config.get_detection_config(model_name).image_size
      image_size = (image_size, image_size)
    elif 'x' in image_size:
      # image_size is in format of WIDTHxHEIGHT
      width, height = image_size.split('x')
      image_size = (int(height), int(width))
    else:
      # image_size is integer, witht the same width and height.
      image_size = (int(image_size), int(image_size))

    self.model_overrides = {
        'image_size': image_size,
        'num_classes': num_classes
    }

    if data_format:
      self.model_overrides.update(dict(data_format=data_format))

    # A few fixed parameters.
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.data_format = data_format
    self.inputs_shape = [self.batch_size, image_size[0], image_size[1], 3]
    self.labels_shape = [self.batch_size, self.num_classes]
    self.image_size = image_size

  def build_model(self, inputs: tf.Tensor,
                  is_training: bool = False) -> List[tf.Tensor]:
    """Build model with inputs and labels and print out model stats."""
    logging.info('start building model')
    model_arch = det_model_fn.get_model_arch(self.model_name)
    cls_outputs, box_outputs = model_arch(
        inputs,
        model_name=self.model_name,
        is_training_bn=is_training,
        use_bfloat16=False,
        **self.model_overrides)

    print('backbone+fpn+box params/flops = {:.6f}M, {:.9f}B'.format(
        *utils.num_params_flops()))

    # Write to tfevent for tensorboard.
    train_writer = tf.summary.FileWriter(self.logdir)
    train_writer.add_graph(tf.get_default_graph())
    train_writer.flush()

    all_outputs = list(cls_outputs.values()) + list(box_outputs.values())
    return all_outputs

  def export_saved_model(self, **kwargs):
    """Export a saved model for inference."""
    tf.enable_resource_variables()
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        enable_ema=self.enable_ema,
        use_xla=self.use_xla,
        data_format=self.data_format,
        **kwargs)
    driver.build(params_override=self.model_overrides)
    driver.export(self.saved_model_dir)

  def saved_model_inference(self, image_path_pattern, output_dir, **kwargs):
    """Perform inference for the given saved model."""
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        enable_ema=self.enable_ema,
        use_xla=self.use_xla,
        data_format=self.data_format,
        **kwargs)
    driver.load(self.saved_model_dir)

    batch_size = self.batch_size
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    num_batches = len(all_files) // batch_size

    for i in range(num_batches):
      batch_files = all_files[i * batch_size: (i + 1) * batch_size]
      raw_images = [np.array(Image.open(f)) for f in batch_files]
      detections_bs = driver.serve_images(raw_images)
      for j in range(len(raw_images)):
        img = driver.visualize(raw_images[j], detections_bs[j], **kwargs)
        img_id = str(i * batch_size + j)
        output_image_path = os.path.join(output_dir, img_id + '.jpg')
        Image.fromarray(img).save(output_image_path)
        logging.info('writing file to %s', output_image_path)

  def saved_model_benchmark(self, image_path_pattern, **kwargs):
    """Perform inference for the given saved model."""
    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        batch_size=self.batch_size,
        enable_ema=self.enable_ema,
        use_xla=self.use_xla,
        data_format=self.data_format,
        **kwargs)
    driver.load(self.saved_model_dir)
    raw_images = []
    all_files = list(tf.io.gfile.glob(image_path_pattern))
    if len(all_files) < self.batch_size:
      all_files = all_files * (self.batch_size // len(all_files) + 1)
    raw_images = [np.array(Image.open(f)) for f in all_files[:self.batch_size]]
    driver.benchmark(raw_images, FLAGS.trace_filename)

  def saved_model_video(self, video_path: Text, output_video: Text, **kwargs):
    """Perform video inference for the given saved model."""
    import cv2  # pylint: disable=g-import-not-at-top

    driver = inference.ServingDriver(
        self.model_name,
        self.ckpt_path,
        enable_ema=self.enable_ema,
        use_xla=self.use_xla)
    driver.load(self.saved_model_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      print('Error opening input video: {}'.format(video_path))

    out_ptr = None
    if output_video:
      frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
      out_ptr = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(
          'm', 'p', '4', 'v'), 25, (frame_width, frame_height))

    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()
      if not ret:
        break

      raw_frames = [np.array(frame)]
      detections_bs = driver.serve_images(raw_frames)
      new_frame = driver.visualize(raw_frames[0], detections_bs[0], **kwargs)

      if out_ptr:
        # write frame into output file.
        out_ptr.write(new_frame)
      else:
        # show the frame online, mainly used for real-time speed test.
        cv2.imshow('Frame', new_frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  def inference_single_image(self, image_image_path, output_dir, **kwargs):
    driver = inference.InferenceDriver(self.model_name, self.ckpt_path,
                                       self.image_size, self.num_classes,
                                       self.enable_ema, self.data_format,
                                       use_batch_nms=self.use_batch_nms)
    driver.inference(image_image_path, output_dir, **kwargs)

  def build_and_save_model(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs, is_training=False)

      # Run the model
      inputs_val = np.random.rand(*self.inputs_shape).astype(float)
      labels_val = np.zeros(self.labels_shape).astype(np.int64)
      labels_val[:, 0] = 1
      sess.run(tf.global_variables_initializer())
      # Run a single train step.
      sess.run(outputs, feed_dict={inputs: inputs_val})
      all_saver = tf.train.Saver(save_relative_paths=True)
      all_saver.save(sess, os.path.join(self.logdir, self.model_name))

      tf_graph = os.path.join(self.logdir, self.model_name + '_train.pb')
      with tf.io.gfile.GFile(tf_graph, 'wb') as f:
        f.write(sess.graph_def.SerializeToString())

  def restore_model(self, sess, ckpt_path, enable_ema=True, export_ckpt=None):
    """Restore variables from a given checkpoint."""
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint(ckpt_path)
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
    saver.restore(sess, checkpoint)

    if export_ckpt:
      print('export model to {}'.format(export_ckpt))
      if ema_assign_op is not None:
        sess.run(ema_assign_op)
      saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
      saver.save(sess, export_ckpt)

  def eval_ckpt(self):
    """build and save the model into self.logdir."""
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build model with inputs and labels.
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      self.build_model(inputs, is_training=False)
      self.restore_model(
          sess, self.ckpt_path, self.enable_ema, self.export_ckpt)

  def freeze_model(self) -> Tuple[Text, Text]:
    """Freeze model and convert them into tflite and tf graph."""
    with tf.Graph().as_default(), tf.Session() as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      outputs = self.build_model(inputs, is_training=False)

      checkpoint = tf.train.latest_checkpoint(self.logdir)
      logging.info('Loading checkpoint: %s', checkpoint)
      saver = tf.train.Saver()

      # Restore the Variables from the checkpoint and frozen the Graph.
      saver.restore(sess, checkpoint)

      output_node_names = [node.op.name for node in outputs]
      graphdef = tf.graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_node_names)

      tf_graph = os.path.join(self.logdir, self.model_name + '_frozen.pb')
      tf.io.gfile.GFile(tf_graph, 'wb').write(graphdef.SerializeToString())

    return graphdef

  def benchmark_model(self, warmup_runs, bm_runs, num_threads,
                      trace_filename=None):
    """Benchmark model."""
    if self.tensorrt:
      print('Using tensorrt ', self.tensorrt)
      self.build_and_save_model()
      graphdef = self.freeze_model()

    if num_threads > 0:
      print('num_threads for benchmarking: {}'.format(num_threads))
      sess_config = tf.ConfigProto(
          intra_op_parallelism_threads=num_threads,
          inter_op_parallelism_threads=1)
    else:
      sess_config = tf.ConfigProto()

    # rewriter_config_pb2.RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.dependency_optimization = 2
    if self.use_xla:
      sess_config.graph_options.optimizer_options.global_jit_level = (
          tf.OptimizerOptions.ON_2)

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
      inputs = tf.placeholder(tf.float32, name='input', shape=self.inputs_shape)
      output = self.build_model(inputs, is_training=False)

      img = np.random.uniform(size=self.inputs_shape)

      sess.run(tf.global_variables_initializer())
      if self.tensorrt:
        fetches = [inputs.name] + [i.name for i in output]
        goutput = self.convert_tr(graphdef, fetches)
        inputs, output = goutput[0], goutput[1:]

      if not self.use_xla:
        # Don't use tf.group because XLA removes the whole graph for tf.group.
        output = tf.group(*output)

      for i in range(warmup_runs):
        start_time = time.time()
        sess.run(output, feed_dict={inputs: img})
        print('Warm up: {} {:.4f}s'.format(i, time.time() - start_time))

      print('Start benchmark runs total={}'.format(bm_runs))
      start = time.perf_counter()
      for i in range(bm_runs):
        sess.run(output, feed_dict={inputs: img})
      end = time.perf_counter()
      inference_time = (end - start) / 10
      print('Per batch inference time: ', inference_time)
      print('FPS: ', self.batch_size / inference_time)

      if trace_filename:
        run_options = tf.RunOptions()
        run_options.trace_level = tf.RunOptions.FULL_TRACE
        run_metadata = tf.RunMetadata()
        sess.run(output, feed_dict={inputs: img},
                 options=run_options, run_metadata=run_metadata)
        logging.info('Dumping trace to %s', trace_filename)
        trace_dir = os.path.dirname(trace_filename)
        if not tf.io.gfile.exists(trace_dir):
          tf.io.gfile.makedirs(trace_dir)
        with tf.io.gfile.GFile(trace_filename, 'w') as trace_file:
          from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          trace_file.write(
              trace.generate_chrome_trace_format(show_memory=True))

  def convert_tr(self, graph_def, fetches):
    """Convert to TensorRT."""
    from tensorflow.python.compiler.tensorrt import trt  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    converter = trt.TrtGraphConverter(
        nodes_blacklist=[t.split(':')[0] for t in fetches],
        input_graph_def=graph_def,
        precision_mode=self.tensorrt)
    infer_graph = converter.convert()
    goutput = tf.import_graph_def(infer_graph, return_elements=fetches)
    return goutput

  def run_model(self, runmode, threads=0):
    """Run the model on devices."""
    if runmode == 'dry':
      self.build_and_save_model()
    elif runmode == 'freeze':
      self.build_and_save_model()
      self.freeze_model()
    elif runmode == 'ckpt':
      self.eval_ckpt()
    elif runmode == 'saved_model_benchmark':
      self.saved_model_benchmark(FLAGS.input_image)
    elif runmode in ('infer', 'saved_model', 'saved_model_infer',
                     'saved_model_video'):
      config_dict = {}
      if FLAGS.line_thickness:
        config_dict['line_thickness'] = FLAGS.line_thickness
      if FLAGS.max_boxes_to_draw:
        config_dict['max_boxes_to_draw'] = FLAGS.max_boxes_to_draw
      if FLAGS.min_score_thresh:
        config_dict['min_score_thresh'] = FLAGS.min_score_thresh

      if runmode == 'infer':
        self.inference_single_image(
            FLAGS.input_image, FLAGS.output_image_dir, **config_dict)
      elif runmode == 'saved_model':
        self.export_saved_model(**config_dict)
      elif runmode == 'saved_model_infer':
        self.saved_model_inference(
            FLAGS.input_image, FLAGS.output_image_dir, **config_dict)
      elif runmode == 'saved_model_video':
        self.saved_model_video(
            FLAGS.input_video, FLAGS.output_video, **config_dict)
    elif runmode == 'bm':
      self.benchmark_model(warmup_runs=5, bm_runs=FLAGS.bm_runs,
                           num_threads=threads,
                           trace_filename=FLAGS.trace_filename)


def main(_):
  if tf.io.gfile.exists(FLAGS.logdir) and FLAGS.delete_logdir:
    logging.info('Deleting log dir ...')
    tf.io.gfile.rmtree(FLAGS.logdir)

  inspector = ModelInspector(
      model_name=FLAGS.model_name,
      image_size=FLAGS.input_image_size,
      num_classes=FLAGS.num_classes,
      logdir=FLAGS.logdir,
      tensorrt=FLAGS.tensorrt,
      use_xla=FLAGS.xla,
      ckpt_path=FLAGS.ckpt_path,
      enable_ema=FLAGS.enable_ema,
      export_ckpt=FLAGS.export_ckpt,
      saved_model_dir=FLAGS.saved_model_dir,
      data_format=FLAGS.data_format,
      batch_size=FLAGS.batch_size,
      use_batch_nms=FLAGS.use_batch_nms)
  inspector.run_model(FLAGS.runmode, FLAGS.threads)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.disable_v2_behavior()
  tf.app.run(main)
