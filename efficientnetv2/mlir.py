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
"""A simple example on how to export MLIR."""
import os
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import effnetv2_model
import utils
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.lite.python.util import get_grappler_config
from tensorflow.lite.python.util import run_graph_optimizations

FLAGS = flags.FLAGS


def define_flags():
  """Define all flags for binary run."""
  flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
  flags.DEFINE_string('model_name', 'efficientnetv2-b0', 'Model name to use.')
  flags.DEFINE_string('dataset_cfg', 'Imagenet', 'dataset config name.')
  flags.DEFINE_string('hparam_str', '', 'k=v,x=y pairs or yaml file.')
  flags.DEFINE_string('export_dir', None, 'Export or saved model directory')


def main(_):
  """Export model to MLIR."""
  model = effnetv2_model.get_model(
      FLAGS.model_name,
      FLAGS.hparam_str,
      include_top=True,
      pretrained=FLAGS.model_dir or True)
  # Use call (not build) to match the namescope: tensorflow issues/29576
  model(tf.ones([1, 224, 224, 3]), False)
  if FLAGS.model_dir:
    ckpt = FLAGS.model_dir
    if tf.io.gfile.isdir(ckpt):
      ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
    utils.restore_tf2_ckpt(model, ckpt, exclude_layers=('_head', 'optimizer'))
  model.summary()

  fff = tf.function(model).get_concrete_function(
      tf.TensorSpec([1, 224, 224, 3], tf.float32))

  frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(fff)

  input_tensors = [
      tensor for tensor in frozen_func.inputs if tensor.dtype != tf.resource
  ]
  output_tensors = frozen_func.outputs

  graph_def = run_graph_optimizations(
      graph_def,
      input_tensors,
      output_tensors,
      config=get_grappler_config([
          'pruning', 'function', 'constfold', 'shape', 'remap', 'memory',
          'common_subgraph_elimination', 'arithmetic', 'loop', 'dependency',
          'debug_stripper'
      ]),
      graph=frozen_func.graph)

  tf_mlir_graph = tf.mlir.experimental.convert_graph_def(graph_def)

  print('export model to {}.mlir'.format(FLAGS.model_name))
  export_dir = FLAGS.export_dir
  if export_dir is None:
    export_dir = '.'
  os.makedirs(export_dir, exist_ok=True)
  outfile = open('{}/{}.mlir'.format(export_dir, FLAGS.model_name), 'wb')
  outfile.write(tf_mlir_graph.encode())
  outfile.close()


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  define_flags()
  app.run(main)
