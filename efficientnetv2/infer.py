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
"""A simple example on how to use keras model for inference."""
import time
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_datasets as tfds

import effnetv2_model
import preprocessing
import utils

FLAGS = flags.FLAGS


def define_flags():
  """Define all flags for binary run."""
  flags.DEFINE_string('mode', 'eval', 'Running mode.')
  flags.DEFINE_string('image_path', None, 'Location of test image.')
  flags.DEFINE_integer('image_size', None, 'Image size.')
  flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
  flags.DEFINE_string('model_name', 'efficientnetv2-s', 'Model name to use.')
  flags.DEFINE_string('dataset_cfg', 'Imagenet', 'dataset config name.')
  flags.DEFINE_string('hparam_str', '', 'k=v,x=y pairs or yaml file.')
  flags.DEFINE_bool('debug', False, 'If true, run in eager for debug.')
  flags.DEFINE_string('export_dir', None, 'Export or saved model directory')
  flags.DEFINE_string('trace_file', '/tmp/a.trace', 'If set, dump trace file.')
  flags.DEFINE_integer('batch_size', 16, 'Batch size.')
  flags.DEFINE_bool('mixed_precision', False, 'If True, use mixed precision.')


def build_tf2_model():
  """Build the tf2 model."""
  tf.config.run_functions_eagerly(FLAGS.debug)
  if FLAGS.mixed_precision:
    # Use 'mixed_float16' if running on GPUs.
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

  model = effnetv2_model.get_model(
      FLAGS.model_name,
      FLAGS.hparam_str,
      include_top=True,
      weights=FLAGS.model_dir or 'imagenet')
  model.summary()
  return model


def tf2_eval_dataset():
  """Run TF2 benchmark and inference."""
  model = build_tf2_model()
  isize = FLAGS.image_size or model.cfg.eval.isize

  def preprocess_fn(features):
    features['image'] = preprocessing.preprocess_image(
        features['image'], isize, is_training=False)
    return features

  @tf.function
  def f(x):
    return model(x)

  top1_acc = tf.keras.metrics.Accuracy()
  pbar = tf.keras.utils.Progbar(None)
  data = tfds.load('imagenet2012', split='validation')
  ds = data.map(preprocess_fn).batch(FLAGS.batch_size)
  for i, batch in enumerate(ds.prefetch(tf.data.experimental.AUTOTUNE)):
    logits = f(batch['image'])
    top1_acc.update_state(batch['label'], tf.argmax(logits, axis=-1))
    pbar.update(i, [('top1', top1_acc.result().numpy())])
  print('\n top1= {:.4f}'.format(top1_acc.result().numpy()))


def tf2_benchmark():
  """Run TF2 benchmark and inference."""
  model = build_tf2_model()
  isize = FLAGS.image_size or model.cfg.eval.isize
  if FLAGS.export_dir:
    tf.saved_model.save(model, FLAGS.export_dir)
    model = tf.saved_model.load(FLAGS.export_dir)

  batch_size = FLAGS.batch_size
  data_dtype = tf.float16 if FLAGS.mixed_precision else tf.float32
  imgs = tf.ones((batch_size, isize, isize, 3), dtype=data_dtype)

  @tf.function
  def f(x):
    return model(x, training=False)

  print('starting warmup.')
  for _ in range(10):  # warmup runs.
    f(imgs)

  print('start benchmark.')
  start = time.perf_counter()
  for _ in range(10):
    f(imgs)
  end = time.perf_counter()
  inference_time = (end - start) / 10

  print('Per batch inference time: ', inference_time)
  print('FPS: ', batch_size / inference_time)


def tf1_benchmark():
  """Run TF1 inference and benchmark."""
  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  from tensorflow.python.client import timeline
  with tf1.Session() as sess:
    model = effnetv2_model.EffNetV2Model(FLAGS.model_name, FLAGS.hparam_str)
    batch_size = FLAGS.batch_size
    run_options = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
    run_metadata = tf1.RunMetadata()
    isize = FLAGS.image_size or model.cfg.eval.isize
    data_dtype = tf.float16 if FLAGS.mixed_precision else tf.float32
    inputs = tf.ones((batch_size, isize, isize, 3), data_dtype)
    output = model(inputs, training=False)
    sess.run(tf1.global_variables_initializer())

    print('starting warmup.')
    for _ in range(5):
      sess.run(output)

    print('starting benchmark.')
    start = time.perf_counter()
    for _ in range(10):
      sess.run(output)
    end = time.perf_counter()
    inference_time = (end - start) / 10

    print('Per batch inference time: ', inference_time)
    print('FPS: ', batch_size / inference_time)

    if FLAGS.trace_file:
      sess.run(output, options=run_options, run_metadata=run_metadata)
      with tf.io.gfile.GFile(FLAGS.trace_file, 'w') as f:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        f.write(trace.generate_chrome_trace_format(show_memory=True))


def tf1_export_ema_ckpt():
  """Restore variables from a given checkpoint."""
  with tf1.Session() as sess:
    model = effnetv2_model.EffNetV2Model(FLAGS.model_name, FLAGS.hparam_str)
    batch_size = FLAGS.batch_size
    isize = FLAGS.image_size or model.cfg.eval.isize
    inputs = tf.ones((batch_size, isize, isize, 3), tf.float32)
    _ = model(inputs, training=False)
    sess.run(tf1.global_variables_initializer())
    if tf.io.gfile.isdir(FLAGS.model_dir):
      ckpt_path = tf1.train.latest_checkpoint(FLAGS.model_dir)
    else:
      ckpt_path = FLAGS.model_dir

    ema = tf1.train.ExponentialMovingAverage(decay=0.0)
    ema_vars = utils.get_ema_vars()
    var_dict = ema.variables_to_restore(ema_vars)
    ema_assign_op = ema.apply(ema_vars)

    tf1.train.get_or_create_global_step()
    sess.run(tf1.global_variables_initializer())
    saver = tf1.train.Saver(var_dict, max_to_keep=1)
    # Restore all variables from ckpt.
    saver.restore(sess, ckpt_path)

    print('export model to {}'.format(FLAGS.export_dir))
    sess.run(ema_assign_op)
    saver = tf1.train.Saver(max_to_keep=1, save_relative_paths=True)
    saver.save(sess, FLAGS.export_dir)


def main(_):
  if FLAGS.mode == 'tf1export':
    tf1_export_ema_ckpt()
  elif FLAGS.mode == 'tf1bm':
    tf1_benchmark()
  elif FLAGS.mode == 'tf2bm':
    tf2_benchmark()
  elif FLAGS.mode == 'tf2eval':
    tf2_eval_dataset()
  else:
    raise ValueError(f'Invalid mode {FLAGS.mode}')


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  define_flags()
  app.run(main)
