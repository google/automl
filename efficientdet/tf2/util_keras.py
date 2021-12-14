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
"""Common keras utils."""
import collections

from typing import Optional, Text
from absl import logging
import tensorflow as tf
import utils

# Prefix variable name mapping from tf2 model to the hub module checkpoint.
HUB_CPT_NAME = collections.OrderedDict([('class_net/class-predict/', 'classes'),
                                        ('box_net/box-predict/', 'boxes'),
                                        ('', 'base_model')])


def build_batch_norm(is_training_bn: bool,
                     beta_initializer: Text = 'zeros',
                     gamma_initializer: Text = 'ones',
                     data_format: Text = 'channels_last',
                     momentum: float = 0.99,
                     epsilon: float = 1e-3,
                     strategy: Optional[Text] = None,
                     name: Text = 'tpu_batch_normalization'):
  """Build a batch normalization layer.

  Args:
    is_training_bn: `bool` for whether the model is training.
    beta_initializer: `str`, beta initializer.
    gamma_initializer: `str`, gamma initializer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    momentum: `float`, momentume of batch norm.
    epsilon: `float`, small value for numerical stability.
    strategy: `str`, whether to use tpu, gpus or other version of batch norm.
    name: the name of the batch normalization layer

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  axis = 1 if data_format == 'channels_first' else -1
  batch_norm_class = utils.batch_norm_class(is_training_bn, strategy)

  bn_layer = batch_norm_class(
      axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=True,
      scale=True,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      name=name)

  return bn_layer


def get_ema_vars(model):
  """Get all exponential moving average (ema) variables."""
  ema_vars = model.trainable_weights
  for v in model.weights:
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  ema_vars_dict = dict()
  # Remove duplicate vars
  for var in ema_vars:
    ema_vars_dict[var.ref()] = var
  return ema_vars_dict


def load_from_hub_checkpoint(model, ckpt_path_or_file):
  """Loads EfficientDetNet weights from EfficientDetNetTrainHub checkpoint."""

  def _get_cpt_var_name(var_name):
    for name_prefix, hub_name_prefix in HUB_CPT_NAME.items():
      if var_name.startswith(name_prefix):
        cpt_var_name = var_name[len(name_prefix):]  # remove the name_prefix
        cpt_var_name = cpt_var_name.replace('/', '.S')
        cpt_var_name = hub_name_prefix + '/' + cpt_var_name
        if name_prefix:
          cpt_var_name = cpt_var_name.replace(':0', '')
        break

    return cpt_var_name + '/.ATTRIBUTES/VARIABLE_VALUE'

  for var in model.weights:
    cpt_var_name = _get_cpt_var_name(var.name)
    var.assign(tf.train.load_variable(ckpt_path_or_file, cpt_var_name))

    logging.log_first_n(
        logging.INFO,
        'Init %s from %s (%s)' % (var.name, cpt_var_name, ckpt_path_or_file),
        10)


def restore_ckpt(model,
                 ckpt_path_or_file,
                 ema_decay=0.9998,
                 skip_mismatch=True,
                 exclude_layers=None):
  """Restore variables from a given checkpoint.

  Args:
    model: the keras model to be restored.
    ckpt_path_or_file: the path or file for checkpoint.
    ema_decay: ema decay rate. If None or zero or negative value, disable ema.
    skip_mismatch: whether to skip variables if shape mismatch,
      only works with tf1 checkpoint.
    exclude_layers: string list exclude layer's variables,
      only works with tf2 checkpoint.

  Raises:
    KeyError: if access unexpected variables.
  """
  if ckpt_path_or_file == '_':
    logging.info('Running test: do not load any ckpt.')
    return
  if tf.io.gfile.isdir(ckpt_path_or_file):
    ckpt_path_or_file = tf.train.latest_checkpoint(ckpt_path_or_file)

  var_list = tf.train.list_variables(ckpt_path_or_file)
  if (var_list[0][0] ==
      '_CHECKPOINTABLE_OBJECT_GRAPH'):
    try:
      # Use custom checkpoint solves mismatch shape issue.
      keys = {var[0].split('/')[0] for var in var_list}
      keys.add('optimizer')
      keys.discard('_CHECKPOINTABLE_OBJECT_GRAPH')
      if exclude_layers:
        exclude_layers = set(exclude_layers)
        keys = keys.difference(exclude_layers)
      ckpt = tf.train.Checkpoint(**{key: getattr(model, key, None)
                                    for key in keys
                                    if getattr(model, key, None)})
      status = ckpt.restore(ckpt_path_or_file)
      status.assert_nontrivial_match()
    except AssertionError:
      # The checkpoint for  EfficientDetNetTrainHub and EfficientDetNet are not
      # the same. If we trained from EfficientDetNetTrainHub using hub module
      # and then want to use the weight in EfficientDetNet, it needed to
      # manually load the model checkpoint.
      load_from_hub_checkpoint(model, ckpt_path_or_file)
  else:
    ema_vars = get_ema_vars(model)
    var_dict = {
        var.name[:-len(":0")]: var for var in ema_vars.values()
    }
    
    if ema_decay > 0:
      ema = tf.train.ExponentialMovingAverage(decay=0.0)
      optimizer = model.optimizer
      if optimizer:
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
          optimizer = optimizer.inner_optimizer
        optimizer.shadow_copy(ema_vars.values())
        opt_ema_fn = lambda var: optimizer.get_slot(var, 'average')
      else:
        opt_ema_fn = lambda var: var
      ema_var_dict = {
          ema.average_name(var): opt_ema_fn(var) for var in ema_vars.values()
      }
      var_dict.update(ema_var_dict)
    
    # add variables that not in var_dict
    for v in model.weights:
      if v.ref() not in ema_vars:
        var_dict[v.name[:-len(":0")]] = v
    # try to load graph-based checkpoint with ema support,
    # else load checkpoint via keras.load_weights which doesn't support ema.
    reader = tf.train.load_checkpoint(ckpt_path_or_file)
    var_shape_map = reader.get_variable_to_shape_map()
    for key, var in var_dict.items():
      if key in var_shape_map:
        if var_shape_map[key] != var.shape:
          msg = 'Shape mismatch: %s, expected %s, but got %s' % (
              key, str(var.shape), str(var_shape_map[key]))
          if skip_mismatch:
            logging.warning(msg)
          else:
            raise ValueError(msg)
        else:
          var.assign(reader.get_tensor(key), read_value=False)
          logging.log_first_n(
              logging.INFO, f'Init {var.name} from {key} ({ckpt_path_or_file})',
              10)
      else:
        msg = 'Not found %s in %s' % (key, ckpt_path_or_file)
        if skip_mismatch:
          logging.warning(msg)
        else:
          raise KeyError(msg)


def fp16_to_fp32_nested(input_nested):
  """Convert fp16 tensors in a nested structure to fp32.

  Args:
    input_nested: A Python dict, values being Tensor or Python list/tuple of
      Tensor or Non-Tensor.

  Returns:
    A Python dict with the same structure as `tensor_dict`,
    with all bfloat16 tensors converted to float32.
  """
  if isinstance(input_nested, tf.Tensor):
    if input_nested.dtype in (tf.bfloat16, tf.float16):
      return tf.cast(input_nested, dtype=tf.float32)
    else:
      return input_nested
  elif isinstance(input_nested, (list, tuple)):
    out_tensor_dict = [fp16_to_fp32_nested(t) for t in input_nested]
  elif isinstance(input_nested, dict):
    out_tensor_dict = {
        k: fp16_to_fp32_nested(v) for k, v in input_nested.items()
    }
  else:
    return input_nested
  return out_tensor_dict


### The following code breaks COCO training.
# def get_batch_norm(bn_class):
#   def _wrapper(*args, **kwargs):
#     if not kwargs.get('name', None):
#       kwargs['name'] = 'tpu_batch_normalization'
#     return bn_class(*args, **kwargs)
#   return _wrapper

# if tf.compat.v1.executing_eagerly_outside_functions():
#   utils.BatchNormalization = get_batch_norm(
#       tf.keras.layers.BatchNormalization)
#   utils.SyncBatchNormalization = get_batch_norm(
#       tf.keras.layers.experimental.SyncBatchNormalization)
#   utils.TpuBatchNormalization = get_batch_norm(
#       tf.keras.layers.experimental.SyncBatchNormalization)
