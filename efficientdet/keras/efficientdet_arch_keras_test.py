import tensorflow.compat.v1 as tf

import keras.efficientdet_arch_keras as arch_keras
import hparams_config


class KerasBiFPNTest(tf.test.TestCase):

    def test_BiFPNLayer_get_config(self):
        config = hparams_config.get_efficientdet_config()
        keras_bifpn = arch_keras.BiFPNLayer(
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
        new_layer = arch_keras.BiFPNLayer(**layer_config)
        self.assertDictEqual(new_layer.get_config(), layer_config)


if __name__ == '__main__':
    tf.test.main()
