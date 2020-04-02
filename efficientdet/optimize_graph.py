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
"""
Optimize saved model graph
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, logging, flags
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph
import tensorflow.compat.v1 as tf
tf.enable_v2_behavior()

FLAGS=flags.FLAGS
flags.DEFINE_string('saved_model_dir', '/tmp/saved_model',
                    "Path to the dir with TensorFlow 'SavedModel' file and variables")
flags.DEFINE_string('output_dir', '/tmp/optimized_saved_model',
                    "String where to saved optimized saved model")
flags.DEFINE_boolean('quantize', False, "Whether quantize graph")
flags.DEFINE_boolean('as_text', False, "Export graph as text")


def convert_graph_def_to_saved_model(export_dir, graph_filepath):
  if tf.io.gfile.exists(export_dir):
    tf.io.gfile.rmtree(export_dir)
  graph_def = get_graph_def_from_file(graph_filepath)
  with tf.Session(graph=tf.Graph()) as session:
    tf.import_graph_def(graph_def, name='')
    signature_def_map = {
        "serving_default":
            tf.saved_model.predict_signature_def(
                {'image_arrays': session.graph.get_tensor_by_name('image_arrays:0')},
                {'detections': session.graph.get_tensor_by_name('detections:0')}),
        "serving_base64":
            tf.saved_model.predict_signature_def(
                {'image_files': session.graph.get_tensor_by_name('image_files:0')},
                {'detections': session.graph.get_tensor_by_name('detections:0')}),
    }
    b = tf.saved_model.Builder(export_dir)
    b.add_meta_graph_and_variables(
        session,
        tags=['serve'],
        signature_def_map=signature_def_map,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        clear_devices=True)
    b.save(FLAGS.as_text)
    logging.info('Optimized graph converted to SavedModel!')


def describe_graph(graph_def, show_nodes=False):
  print('Input Feature Nodes: {}'.format(
      [node.name for node in graph_def.node if node.op=='Placeholder']))
  print('')
  print('Unused Nodes: {}'.format(
      [node.name for node in graph_def.node if 'unused'  in node.name]))
  print('')
  print('Output Nodes: {}'.format(
      [node.name for node in graph_def.node if (
          'detections' in node.name)]))
  print('')
  print('Quantization Nodes: {}'.format(
      [node.name for node in graph_def.node if 'quant' in node.name]))
  print('')
  print('Constant Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Const'])))
  print('')
  print('Variable Count: {}'.format(
      len([node for node in graph_def.node if 'Variable' in node.op])))
  print('')
  print('Identity Count: {}'.format(
      len([node for node in graph_def.node if node.op=='Identity'])))
  print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

  if show_nodes==True:
    for node in graph_def.node:
      print('Op:{} - Name: {}'.format(node.op, node.name))


def get_graph_def_from_file(graph_filepath):
  with tf.Graph().as_default():
    with tf.io.gfile.GFile(graph_filepath, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      return graph_def


def main(_):
    freeze_graph(input_saved_model_dir=FLAGS.saved_model_dir,
                 output_graph='/tmp/frozen_graph.pb',
                 output_node_names='detections',
                 input_graph=None,
                 input_saver=False,
                 input_binary=True,
                 initializer_nodes='',
                 input_checkpoint=None,
                 restore_op_name=None,
                 filename_tensor_name=None,
                 clear_devices=False,
                 input_meta_graph=False)
    logging.info('Freeze graph succeed')
    graph_def = get_graph_def_from_file('/tmp/frozen_graph.pb')
    describe_graph(graph_def)

    transforms = [
        'remove_nodes(op=Identity, op=CheckNumerics)',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms',
        'fold_old_batch_norms',
        'remove_device'
    ]
    if FLAGS.quantize:
        transforms += [
            'quantize_weights',
        ]
    input_names = []
    output_names = ['detections']
    optimized_graph_def = TransformGraph(
        graph_def,
        input_names,
        output_names,
        transforms)
    describe_graph(optimized_graph_def)
    tf.io.write_graph(optimized_graph_def,
                         logdir='/tmp',
                         as_text=False,
                         name='optimized_model.pb')
    convert_graph_def_to_saved_model(FLAGS.output_dir, '/tmp/optimized_model.pb')
    logging.info("Optimized model saved at /tmp/optimized_model.pb")
    tf.io.gfile.remove('/tmp/frozen_graph.pb')
    tf.io.gfile.remove('/tmp/optimized_model.pb')
    logging.info("Finished")


if __name__ == "__main__":
    app.run(main)