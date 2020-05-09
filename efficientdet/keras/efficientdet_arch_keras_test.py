import tensorflow.compat.v1 as tf
from tensorflow.python.framework.test_util import deprecated_graph_mode_only
import efficientdet_arch
from keras import efficientdet_arch_keras
import hparams_config

class KerasBiFPNTest(tf.test.TestCase):

  def test_BiFPNLayer_get_config(self):
    config = hparams_config.get_efficientdet_config()
    keras_bifpn = efficientdet_arch_keras.BiFPNLayer(
      fpn_name=config.fpn_name,
      min_level=config.min_level,
      max_level=config.max_level,
      fpn_weight_method=config.fpn_weight_method,
      apply_bn_for_resampling=config.apply_bn_for_resampling,
      is_training_bn=config.is_training_bn,
      conv_after_downsample=config.conv_after_downsample,
      use_native_resize_op=config.use_native_resize_op,
      data_format=config.data_format,
      image_size=config.image_size,
      fpn_num_filters=config.fpn_num_filters,
      conv_bn_act_pattern=config.conv_bn_act_pattern,
      act_type=config.act_type,
      pooling_type=config.pooling_type,
      separable_conv=config.separable_conv,
      use_tpu=config.use_tpu
    )

    layer_config = keras_bifpn.get_config()
    new_layer = efficientdet_arch_keras.BiFPNLayer(**layer_config)
    self.assertDictEqual(new_layer.get_config(), layer_config)

class KerasTest(tf.test.TestCase):
  def test_resample_feature_map(self):
    feat = tf.random.uniform([1, 16, 16, 320])
    for apply_fn in [True, False]:
      for is_training in [True, False]:
        for use_tpu in [True, False]:
          with self.subTest(apply_fn=apply_fn,
                            is_training=is_training,
                            use_tpu=use_tpu):
            tf.random.set_random_seed(111111)
            expect_result = efficientdet_arch.resample_feature_map(
                feat,
                name='resample_p0',
                target_height=8,
                target_width=8,
                target_num_channels=64,
                apply_bn=apply_fn,
                is_training=is_training,
                use_tpu=use_tpu)
            tf.random.set_random_seed(111111)
            actual_result = efficientdet_arch_keras.ResampleFeatureMap(
                name='resample_p0',
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
    actual_result = efficientdet_arch_keras.ResampleFeatureMap(
        name='p0',
        target_height=8,
        target_width=8,
        target_num_channels=64)(feat)
    self.assertEqual("resample_p0/max_pooling2d/MaxPool:0", actual_result.name)


if __name__ == '__main__':
  tf.test.main()
