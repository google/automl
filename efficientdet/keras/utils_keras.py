import tensorflow as tf
from utils import batch_norm_class
from typing import Text


def batch_normalization(is_training_bn: bool,
                        init_zero: bool = False,
                        data_format: Text = 'channels_last',
                        momentum: float = 0.99,
                        epsilon: float = 1e-3,
                        use_tpu: bool = False,
                        name: Text = None):
  """Performs a batch normalization followed by a non-linear activation.

  Args:
    is_training_bn: `bool` for whether the model is training.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    use_tpu: `bool`, whether to use tpu version of batch norm.
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
    axis = 3

  batch_normalization = batch_norm_class(is_training_bn, use_tpu)(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      gamma_initializer=gamma_initializer,
      name=name)

  return batch_normalization
