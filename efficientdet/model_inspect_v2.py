import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow as tf

import hparams_config
from keras import inference
import utils

flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model.')
flags.DEFINE_string('mode', 'infer',
                    'Run mode: {dry, infer, export, benchmark}')
flags.DEFINE_string('trace_filename', None, 'Trace file name.')

flags.DEFINE_integer('bm_runs', 10, 'Number of benchmark runs.')
flags.DEFINE_string('tensorrt', None, 'TensorRT mode: {None, FP32, FP16, INT8}')
flags.DEFINE_bool('use_xla', False, 'Run with xla optimization.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference.')

flags.DEFINE_string('ckpt_path', None, 'checkpoint dir used for eval.')

flags.DEFINE_string(
    'hparams', '', 'Comma separated k=v pairs of hyperparameters or a module'
    ' containing attributes to use as hyperparameters.')

flags.DEFINE_string('input_image', None, 'Input image path for inference.')
flags.DEFINE_string('output_image_dir', None, 'Output dir for inference.')

# For video.
flags.DEFINE_string('input_video', None, 'Input video path for inference.')
flags.DEFINE_string('output_video', None,
                    'Output video path. If None, play it online instead.')

# For visualization.
flags.DEFINE_integer('max_boxes_to_draw', 100, 'Max number of boxes to draw.')
flags.DEFINE_float('min_score_thresh', 0.4, 'Score threshold to show box.')
flags.DEFINE_string('nms_method', 'hard', 'nms method, hard or gaussian.')

# For saved model.
flags.DEFINE_string('saved_model_dir', '/tmp/saved_model',
                    'Folder path for saved model.')
flags.DEFINE_string('tflite_path', None, 'Path for exporting tflite file.')
flags.DEFINE_bool('debug', False, 'Debug mode.')

FLAGS = flags.FLAGS


def main(_):
  tf.config.experimental_run_functions_eagerly(FLAGS.debug)
  devices = tf.config.list_physical_devices('GPU')
  for device in devices:
    tf.config.experimental.set_memory_growth(device, True)

  model_config = hparams_config.get_detection_config(FLAGS.model_name)
  model_config.override(FLAGS.hparams)  # Add custom overrides
  model_config.is_training_bn = False
  model_config.image_size = utils.parse_image_size(model_config.image_size)

  # A hack to make flag consistent with nms configs.
  if FLAGS.min_score_thresh:
    model_config.nms_configs.score_thresh = FLAGS.min_score_thresh
  if FLAGS.nms_method:
    model_config.nms_configs.method = FLAGS.nms_method
  if FLAGS.max_boxes_to_draw:
    model_config.nms_configs.max_output_size = FLAGS.max_boxes_to_draw

  model_params = model_config.as_dict()
  driver = inference.ServingDriver(FLAGS.model_name, FLAGS.ckpt_path,
                                   FLAGS.batch_size or None,
                                   FLAGS.min_score_thresh,
                                   FLAGS.max_boxes_to_draw, model_params)
  if FLAGS.mode == 'export':
    if tf.io.gfile.exists(FLAGS.saved_model_dir):
      tf.io.gfile.rmtree(FLAGS.saved_model_dir)
    driver.export(FLAGS.saved_model_dir, FLAGS.tflite_path, FLAGS.tensorrt)
  elif FLAGS.mode == 'infer':
    if tf.saved_model.contains_saved_model(FLAGS.saved_model_dir):
      driver.load(FLAGS.saved_model_dir)
    image_file = tf.io.read_file(FLAGS.input_image)
    image_file = tf.expand_dims(image_file, axis=0)
    detections_bs = driver.serve_files(image_file)
    boxes, scores, classes, _ = tf.nest.map_structure(np.array, detections_bs)
    raw_image = Image.open(FLAGS.input_image)
    img = driver.visualize(
        raw_image,
        boxes[0],
        classes[0],
        scores[0],
        min_score_thresh=model_config.nms_configs.score_thresh,
        max_boxes_to_draw=model_config.nms_configs.max_output_size)
    output_image_path = os.path.join(FLAGS.output_image_dir, '0.jpg')
    Image.fromarray(img).save(output_image_path)
    print('writing file to %s' % output_image_path)
  elif FLAGS.mode == 'benchmark':
    if tf.saved_model.contains_saved_model(FLAGS.saved_model_dir):
      driver.load(FLAGS.saved_model_dir)
    image_file = tf.io.read_file(FLAGS.input_image)
    image_arrays = tf.image.decode_image(image_file)
    image_arrays.set_shape((None, None, 3))
    image_arrays = tf.expand_dims(image_arrays, axis=0)
    driver.benchmark(image_arrays, FLAGS.bm_runs, FLAGS.trace_filename)
  elif FLAGS.mode == 'dry':
    # transfer to tf2 format ckpt
    ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    driver.model.save_weights(ckpt_path)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  app.run(main)
