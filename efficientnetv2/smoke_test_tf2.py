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
"""Tests for EfficientNetV2 train in tf2 smoke tests."""
import sys
import tempfile

from absl import flags
from absl.testing import flagsaver
import tensorflow as tf

import cflags
import main_tf2 as main_lib


FLAGS = flags.FLAGS
GPU_TEST = 'gpu_test' in sys.argv[0]
TPU_TEST = 'test_tpu' in sys.argv[0]


class EfficientNetV2TF2Test(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    FLAGS.tpu = ''
    FLAGS.model_dir = tempfile.mkdtemp()
    FLAGS.data_dir = 'null'
    cls.hparam_str = (
        'train.batch_size=2,eval.batch_size=2,train.epochs=1,'
        'train.stages=0,train.lr_base=0,data.splits.train.num_images=2,data.splits.eval.num_images=6,')

  def _run_single_step_train_and_eval(self, hparam_str=''):
    """Run single step train and eval."""
    FLAGS.hparam_str = self.hparam_str + hparam_str
    FLAGS.mode = 'train'
    main_lib.main(None)

    tf.compat.v1.reset_default_graph()
    FLAGS.mode = 'eval'
    main_lib.main(None)
  
  # TODO(someone): remove `runtime.strategy` and use `use_tpu`
  @flagsaver.flagsaver(
      model_name='efficientnetv2-s', dataset_cfg='ImageNet')
  def test_cpu_b0_model_single_step(self):
    self._run_single_step_train_and_eval('runtime.strategy=cpu')

  @flagsaver.flagsaver(use_tpu=True)
  def test_tpu_b0_model_bfloat_single_step(self):
    if TPU_TEST:
      self._run_single_step_train_and_eval('')
    else:
      self.skipTest('Skip because no TPU is available.')

  @flagsaver.flagsaver(use_tpu=False)
  def test_tpu_b0_model_single_step_gpu(self):
    if GPU_TEST:
      # Disables export as tflite does not support NCHW layout.
      self._run_single_step_train_and_eval('model.data_format=channels_first')
    else:
      self.skipTest('Skip because no GPU is available.')


if __name__ == '__main__':
  cflags.define_flags()
  tf.test.main()
