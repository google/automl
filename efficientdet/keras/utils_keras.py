from typing import Text
import tensorflow as tf


def batch_normalization(is_training_bn: bool,
                        init_zero: bool = False,
                        data_format: Text = 'channels_last',
                        momentum: float = 0.99,
                        epsilon: float = 1e-3,
                        strategy: Text = None,
                        name: Text = 'tpu_batch_normalization'):
  """Performs a batch normalization followed by a non-linear activation.

  Args:
    is_training_bn: `bool` for whether the model is training.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    strategy: `bool`, whether to use tpu version of batch norm.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == 'channels_first':
    axis = 1
  else:
    axis = -1
  if strategy in ['tpu', 'gpu'] and is_training_bn:
    batch_norm_class = tf.keras.layers.experimental.SyncBatchNormalization
  else:
    batch_norm_class = tf.keras.layers.BatchNormalization
  bn = batch_norm_class(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      gamma_initializer=gamma_initializer,
      name=name)

  return bn
