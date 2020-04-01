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
"""Saved model export to tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, logging, flags
import tensorflow.compat.v1 as tf

tf.enable_v2_behavior()
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'saved_model_dir', default='/tmp/saved_model',
    help='Saved model directory')
flags.DEFINE_string(
    'output_path', default='/tmp/model.tflite',
    help='Tflite model save path')
flags.DEFINE_string(
    'sample_image', default='/tmp/test.png',
    help='Sample image path')
flags.DEFINE_integer(
    'input_image_size', default=512,
    help='Input image shape')


def main(_):
    image_size = FLAGS.input_image_size
    converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.model_dir, ['image_arrays'],
                                                         {'image_arrays': [None, image_size, image_size, 3]},
                                                         ['detections'],signature_key='serving_default')
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    tf.io.write_file(FLAGS.output_path, tflite_model)
    logging.info('Model saved at ' + FLAGS.output_path)
    # Active development, solution https://github.com/tensorflow/tensorflow/issues/32004
    if False:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the TensorFlow Lite model on random input data.

        input_data = tf.io.gfile.GFile(FLAGS.sample_image, 'rb').read()
        image = tf.io.decode_image(input_data, channels=3, dtype=tf.uint8)
        image = tf.image.resize(image, [image_size, image_size])
        logging.info('Sample image loaded ' + FLAGS.sample_image)
        image = tf.expand_dims(tf.cast(image, tf.uint8), 0)
        interpreter.set_tensor(input_details[0]['index'], np.array(image))

        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        logging.info(tflite_results)


if __name__ == '__main__':
    app.run(main)
