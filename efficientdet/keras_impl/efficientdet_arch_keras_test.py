import tensorflow.compat.v1 as tf
import efficientdet_arch
from keras_impl import efficientdet_arch_keras as efficientdet_arch_keras
from tensorflow.python.framework.test_util import deprecated_graph_mode_only

class KerasTest(tf.test.TestCase):
  def test_resample_feature_map(self):
    feat = tf.random.uniform([1, 16, 16, 320])
    for apply_fn in [True, False]:
      for is_training in [True, False]:
        for use_tpu in [True, False]:
          with self.subTest(apply_fn=apply_fn, is_training=is_training, use_tpu=use_tpu):
            tf.random.set_random_seed(111111)
            expect_result = efficientdet_arch.resample_feature_map(feat,
                                                                   name='resample_p0',
                                                                   target_height=8,
                                                                   target_width=8,
                                                                   target_num_channels=64,
                                                                   apply_bn=apply_fn,
                                                                   is_training=is_training,
                                                                   use_tpu=use_tpu)
            tf.random.set_random_seed(111111)
            actual_result = efficientdet_arch_keras.ResampleFeatureMap(name='resample_p0',
                                                                       target_height=8,
                                                                       target_width=8,
                                                                       target_num_channels=64,
                                                                       apply_bn=apply_fn,
                                                                       is_training=is_training,
                                                                       use_tpu=use_tpu)(feat)
            self.assertAllCloseAccordingToType(expect_result, actual_result)

  @deprecated_graph_mode_only
  def test_name(self):
    feat = tf.random.uniform([1, 16, 16, 320])
    actual_result = efficientdet_arch_keras.ResampleFeatureMap(name='p0',
                                                               target_height=8,
                                                               target_width=8,
                                                               target_num_channels=64)(feat)
    self.assertEqual("resample_p0/max_pooling2d/MaxPool:0", actual_result.name)


if __name__ == '__main__':
  tf.test.main()