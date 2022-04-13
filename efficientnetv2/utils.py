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
"""Model utilities."""
import contextlib
import functools
import os
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons.layers as tfa_layers

from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import


def activation_fn(features: tf.Tensor, act_fn: str):
  """Customized non-linear activation type."""
  if act_fn in ('silu', 'swish'):
    return tf.nn.swish(features)
  elif act_fn == 'silu_native':
    return features * tf.sigmoid(features)
  elif act_fn == 'hswish':
    return features * tf.nn.relu6(features + 3) / 6
  elif act_fn == 'relu':
    return tf.nn.relu(features)
  elif act_fn == 'relu6':
    return tf.nn.relu6(features)
  elif act_fn == 'elu':
    return tf.nn.elu(features)
  elif act_fn == 'leaky_relu':
    return tf.nn.leaky_relu(features)
  elif act_fn == 'selu':
    return tf.nn.selu(features)
  elif act_fn == 'mish':
    return features * tf.math.tanh(tf.math.softplus(features))
  else:
    raise ValueError('Unsupported act_fn {}'.format(act_fn))


def get_act_fn(act_fn):
  if not act_fn:
    return tf.nn.silu
  if isinstance(act_fn, str):
    return functools.partial(activation_fn, act_fn=act_fn)
  return act_fn


def cross_replica_mean(t, num_shards_per_group=None):
  """Calculates the average value of input tensor across TPU replicas."""
  num_shards = tpu_function.get_tpu_context().number_of_shards
  if not num_shards_per_group:
    return tf.compat.v1.tpu.cross_replica_sum(t) / tf.cast(num_shards, t.dtype)

  group_assignment = None
  if num_shards_per_group > 1:
    if num_shards % num_shards_per_group != 0:
      raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0' %
                       (num_shards, num_shards_per_group))
    num_groups = num_shards // num_shards_per_group
    group_assignment = [[
        x for x in range(num_shards) if x // num_shards_per_group == y
    ] for y in range(num_groups)]
  return tf.compat.v1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
      num_shards_per_group, t.dtype)


class WarmupLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Provides a variety of learning rate decay schedules with warm up."""

  def __init__(self,
               initial_lr,
               steps_per_epoch=None,
               lr_decay_type='exponential',
               decay_factor=0.97,
               decay_epochs=2.4,
               total_steps=None,
               warmup_epochs=5,
               minimal_lr=0):
    super(WarmupLearningRateSchedule, self).__init__()
    self.initial_lr = initial_lr
    self.steps_per_epoch = steps_per_epoch
    self.lr_decay_type = lr_decay_type
    self.decay_factor = decay_factor
    self.decay_epochs = decay_epochs
    self.total_steps = total_steps
    self.warmup_epochs = warmup_epochs
    self.minimal_lr = minimal_lr

  def __call__(self, step):
    if self.lr_decay_type == 'exponential':
      assert self.steps_per_epoch is not None
      decay_steps = self.steps_per_epoch * self.decay_epochs
      lr = tf.keras.optimizers.schedules.ExponentialDecay(
          self.initial_lr, decay_steps, self.decay_factor, staircase=True)(
              step)
    elif self.lr_decay_type == 'cosine':
      assert self.total_steps is not None
      lr = 0.5 * self.initial_lr * (
          1 + tf.cos(np.pi * tf.cast(step, tf.float32) / self.total_steps))
    elif self.lr_decay_type == 'linear':
      assert self.total_steps is not None
      lr = (1.0 -
            tf.cast(step, tf.float32) / self.total_steps) * self.initial_lr
    elif self.lr_decay_type == 'constant':
      lr = self.initial_lr
    else:
      assert False, 'Unknown lr_decay_type : %s' % self.lr_decay_type

    if self.minimal_lr:
      lr = tf.math.maximum(lr, self.minimal_lr)

    if self.warmup_epochs:
      warmup_steps = int(self.warmup_epochs * self.steps_per_epoch)
      warmup_lr = (
          self.initial_lr * tf.cast(step, tf.float32) /
          tf.cast(warmup_steps, tf.float32))
      lr = tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr

  def get_config(self):
    return {
        'initial_lr': self.initial_lr,
        'steps_per_epoch': self.steps_per_epoch,
        'lr_decay_type': self.lr_decay_type,
        'decay_factor': self.decay_factor,
        'decay_epochs': self.decay_epochs,
        'total_steps': self.total_steps,
        'warmup_epochs': self.warmup_epochs,
        'minimal_lr': self.minimal_lr,
    }


def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
  """Build optimizer."""
  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    logging.info('Using Momentum optimizer')
    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp optimizer')
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate, decay,
                                                    momentum, epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam optimizer')
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
  else:
    logging.fatal('Unknown optimizer: %s', optimizer_name)

  return optimizer


class TpuBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super().__init__(fused=fused, **kwargs)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super()._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    num_shards_per_group = min(8, num_shards)  # aggregate up to 32 cores.
    if num_shards_per_group > 1:
      logging.info('TpuBatchNormalization with num_shards_per_group %d',
                   num_shards_per_group)
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = cross_replica_mean(shard_mean, num_shards_per_group)
      group_mean_of_square = cross_replica_mean(shard_mean_of_square,
                                                num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)

  def call(self, inputs, training=None):
    outputs = super().call(inputs, training)
    return outputs


class BatchNormalization(tf.keras.layers.BatchNormalization):
  """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super().__init__(**kwargs)


def normalization(norm_type: str,
                  axis=-1,
                  epsilon=0.001,
                  momentum=0.99,
                  groups=8,
                  name=None):
  """Normalization after conv layers."""
  if norm_type == 'gn':
    return tfa_layers.GroupNormalization(groups, axis, epsilon, name=name)

  if norm_type == 'tpu_bn':
    return TpuBatchNormalization(
        axis=axis, momentum=momentum, epsilon=epsilon, name=name)

  return BatchNormalization(
      axis=axis, momentum=momentum, epsilon=epsilon, name=name)


def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
  """Archive a checkpoint if the metric is better."""
  ckpt_dir, ckpt_name = os.path.split(ckpt_path)

  saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')
  saved_objective = float('-inf')
  if tf.io.gfile.exists(saved_objective_path):
    with tf.io.gfile.GFile(saved_objective_path, 'r') as f:
      saved_objective = float(f.read())
  if saved_objective > ckpt_objective:
    logging.info('Ckpt %s is worse than %s', ckpt_objective, saved_objective)
    return False

  filenames = tf.io.gfile.glob(ckpt_path + '.*')
  if filenames is None:
    logging.info('No files to copy for checkpoint %s', ckpt_path)
    return False

  # Clear the old folder.
  dst_dir = os.path.join(ckpt_dir, 'archive')
  if tf.io.gfile.exists(dst_dir):
    tf.io.gfile.rmtree(dst_dir)
  tf.io.gfile.makedirs(dst_dir)

  # Write checkpoints.
  for f in filenames:
    dest = os.path.join(dst_dir, os.path.basename(f))
    tf.io.gfile.copy(f, dest, overwrite=True)
  ckpt_state = tf.compat.v1.train.generate_checkpoint_state_proto(
      dst_dir,
      model_checkpoint_path=ckpt_name,
      all_model_checkpoint_paths=[ckpt_name])
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
    f.write(str(ckpt_state))
  with tf.io.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
    f.write('%s' % ckpt_eval)

  # Update the best objective.
  with tf.io.gfile.GFile(saved_objective_path, 'w') as f:
    f.write('%f' % ckpt_objective)

  logging.info('Copying checkpoint %s to %s', ckpt_path, dst_dir)
  return True


def get_ema_vars():
  """Get all exponential moving average (ema) variables."""
  ema_vars = tf.compat.v1.trainable_variables() + tf.compat.v1.get_collection(
      'moving_vars')
  for v in tf.compat.v1.global_variables():
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  return list(set(ema_vars))


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


def num_params_flops(readable_format=True):
  """Return number of parameters and flops."""
  nparams = np.sum([
      np.prod(v.get_shape().as_list())
      for v in tf.compat.v1.trainable_variables()
  ])
  options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
  options['output'] = 'none'
  flops = tf.compat.v1.profiler.profile(
      tf.compat.v1.get_default_graph(), options=options).total_float_ops
  # We use flops to denote multiply-adds, which is counted as 2 ops in tfprof.
  flops = flops // 2
  if readable_format:
    nparams = float(nparams) * 1e-6
    flops = float(flops) * 1e-9
  return nparams, flops


class Pair(tuple):

  def __new__(cls, name, value):
    return super().__new__(cls, (name, value))

  def __init__(self, name, _):  # pylint: disable=super-init-not-called
    self.name = name


def scalar(name, tensor, is_tpu=True):
  """Stores a (name, Tensor) tuple in a custom collection."""
  logging.info('Adding scalar summary %s', Pair(name, tensor))
  if is_tpu:
    tf.compat.v1.add_to_collection('scalar_summaries',
                                   Pair(name, tf.reduce_mean(tensor)))
  else:
    tf.summary.scalar(name, tf.reduce_mean(tensor))


def image(name, tensor, is_tpu=True):
  logging.info('Adding image summary %s', Pair(name, tensor))
  if is_tpu:
    tf.compat.v1.add_to_collection('image_summaries', Pair(name, tensor))
  else:
    tf.summary.image(name, tensor)


def get_tpu_host_call(global_step, model_dir, iterations_per_loop):
  """Get TPU host call for summaries."""
  scalar_summaries = tf.compat.v1.get_collection('scalar_summaries')
  if not scalar_summaries:
    return None  # No summaries to write.

  def host_call_fn(global_step, *args):
    """Training host call. Creates summaries for training metrics."""
    gs = global_step[0]
    with tf.summary.create_file_writer(
        model_dir, max_queue=iterations_per_loop).as_default():
      with tf.summary.record_if(True):
        for i, _ in enumerate(scalar_summaries):
          name = scalar_summaries[i][0]
          tensor = args[i][0]
          tf.summary.scalar(name, tensor, step=gs)
      return tf.compat.v1.summary.all_v2_summary_ops()

  reshaped_tensors = [tf.reshape(t, [1]) for _, t in scalar_summaries]
  global_step_t = tf.reshape(global_step, [1])
  return host_call_fn, [global_step_t] + reshaped_tensors


@contextlib.contextmanager
def float16_scope():
  """Scope class for float16."""

  def _custom_getter(getter, *args, **kwargs):
    """Returns a custom getter that methods must be called under."""
    cast_to_float16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == tf.float16:
      kwargs['dtype'] = tf.float32
      cast_to_float16 = True
    var = getter(*args, **kwargs)
    if cast_to_float16:
      var = tf.cast(var, tf.float16)
    return var

  with tf.compat.v1.variable_scope(
      '', custom_getter=_custom_getter) as varscope:
    yield varscope


def set_precision_policy(policy_name=None):
  """Set precision policy according to the name.

  Args:
    policy_name: precision policy name, one of 'float32', 'mixed_float16',
      'mixed_bfloat16', or None.
    loss_scale: whether to use loss scale (only for training).
  """
  if not policy_name:
    return

  assert policy_name in ('mixed_float16', 'mixed_bfloat16', 'float32')
  logging.info('use mixed precision policy name %s', policy_name)
  tf.compat.v1.keras.layers.enable_v2_dtype_behavior()
  policy = tf.keras.mixed_precision.Policy(policy_name)
  tf.keras.mixed_precision.set_policy(policy)


def build_model_with_precision(pp, mm, ii, tt, *args, **kwargs):
  """Build model with its inputs/params for a specified precision context.

  This is highly specific to this codebase, and not intended to be general API.
  Advanced users only. DO NOT use it if you don't know what it does.
  NOTE: short argument names are intended to avoid conficts with kwargs.

  Args:
    pp: A string, precision policy name, such as "mixed_float16".
    mm: A function, for rmodel builder.
    ii: A tensor, for model inputs.
    tt: A bool, If true, it is for training; otherwise, it is for eval.
    *args: A list of model arguments.
    **kwargs: A dict, extra model parameters.

  Returns:
    the output of mm model.
  """
  del tt
  if pp == 'mixed_bfloat16':
    set_precision_policy(pp)
    inputs = tf.cast(ii, tf.bfloat16)
    with tf.compat.v1.tpu.bfloat16_scope():
      outputs = mm(inputs, *args, **kwargs)
    set_precision_policy('float32')
  elif pp == 'mixed_float16':
    set_precision_policy(pp)
    inputs = tf.cast(ii, tf.float16)
    with float16_scope():
      outputs = mm(inputs, *args, **kwargs)
    set_precision_policy('float32')
  elif not pp or pp == 'float32':
    outputs = mm(ii, *args, **kwargs)
  else:
    raise ValueError('Unknow precision name {}'.format(pp))

  # Users are responsible to convert the dtype of all outputs.
  return outputs


def get_ckpt_var_map(ckpt_path,
                     ckpt_scope='',
                     var_scope='',
                     skip_mismatch=None,
                     init_ema=True):
  """Get a var map for restoring from pretrained checkpoints.

  Args:
    ckpt_path: string. A pretrained checkpoint path.
    ckpt_scope: string. Scope name for checkpoint variables.
    var_scope: string. Scope name for model variables.
    skip_mismatch: skip variables if shape mismatch.
    init_ema: If true, try to init from ema variables.

  Returns:
    var_map: a dictionary from checkpoint name to model variables.
  """
  logging.info('Init model from checkpoint %s', ckpt_path)
  var_map = {}
  # Get the list of vars to restore.
  model_vars = tf.compat.v1.get_collection(
      tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=var_scope)
  reader = tf.train.load_checkpoint(ckpt_path)
  ckpt_var_name_to_shape = reader.get_variable_to_shape_map()
  ckpt_var_names = set(reader.get_variable_to_shape_map().keys())
  for v in model_vars:
    v_name = v.op.name

    # filter special variables.
    flist = ['global_step', 'ExponentialMovingAverage', 'Momentum', 'RMSProp']
    if list(filter(lambda x, s=v_name: x in s, flist)):
      continue

    if not v.op.name.startswith(var_scope):
      logging.info('skip %s -- does not match scope %s', v_name, var_scope)
    cv_name = ckpt_scope + v.op.name[len(var_scope):]
    if init_ema and cv_name + '/ExponentialMovingAverage' in ckpt_var_names:
      cv_name = cv_name + '/ExponentialMovingAverage'  # prefer ema vars.

    if cv_name not in ckpt_var_names:
      if skip_mismatch:
        logging.info('skip %s (%s) -- not in ckpt', v_name, cv_name)
        continue
      raise ValueError(f'{v.op} is not in ckpt {ckpt_path}')

    cv_shape = ckpt_var_name_to_shape[cv_name]
    if v.shape != cv_shape:
      if skip_mismatch:
        logging.info('skip %s (%s vs %s) -- shape mismatch', v_name, v.shape,
                     cv_shape)
        continue
      raise ValueError(f'shape mismatch {v_name} ({v.shape} vs {cv_shape})')

    var_map[cv_name] = v

  if not var_map or len(var_map) < 5:
    raise ValueError(f'var_map={var_map} is almost empty, please check logs.')

  for (k, v) in var_map.items():
    logging.log_first_n(logging.INFO, f'Init {v.op.name} from ckpt var {k}', 10)

  return var_map


def restore_tf2_ckpt(model,
                     ckpt_path_or_file,
                     skip_mismatch=True,
                     exclude_layers=None):
  """Restore variables from a given checkpoint.

  Args:
    model: the keras model to be restored.
    ckpt_path_or_file: the path or file for checkpoint.
    skip_mismatch: whether to skip variables if shape mismatch,
      only works with tf1 checkpoint.
    exclude_layers: string list exclude layer's variables,
      only works with tf2 checkpoint.

  Raises:
    KeyError: if access unexpected variables.
  """
  ckpt_file = ckpt_path_or_file
  if tf.io.gfile.isdir(ckpt_file):
    ckpt_file = tf.train.latest_checkpoint(ckpt_file)

  # Try to load object-based checkpoint (by model.save_weights).
  var_list = tf.train.list_variables(ckpt_file)
  if var_list[0][0] == '_CHECKPOINTABLE_OBJECT_GRAPH':
    print(f'Load checkpointable from {ckpt_file}, excluding {exclude_layers}')
    keys = {var[0].split('/')[0] for var in var_list}
    keys.discard('_CHECKPOINTABLE_OBJECT_GRAPH')
    if exclude_layers:
      exclude_layers = set(exclude_layers)
      keys = keys.difference(exclude_layers)
    ckpt = tf.train.Checkpoint(**{key: getattr(model, key, None)
                                  for key in keys
                                  if getattr(model, key, None)})
    status = ckpt.restore(ckpt_file)
    status.assert_nontrivial_match()
    return

  print(f'Load TF1 graph based checkpoint from {ckpt_file}.')
  var_dict = {v.name.split(':')[0]: v for v in model.weights}
  reader = tf.train.load_checkpoint(ckpt_file)
  var_shape_map = reader.get_variable_to_shape_map()
  for key, var in var_dict.items():
    if key in var_shape_map:
      if var_shape_map[key] != var.shape:
        msg = 'Shape mismatch: %s' % key
        if skip_mismatch:
          logging.warning(msg)
        else:
          raise ValueError(msg)
      else:
        var.assign(reader.get_tensor(key), read_value=False)
        logging.log_first_n(logging.INFO,
                            f'Init {var.name} from {key} ({ckpt_file})', 10)
    else:
      msg = 'Not found %s in %s' % (key, ckpt_file)
      if skip_mismatch:
        logging.warning(msg)
      else:
        raise KeyError(msg)


class ReuableBackupAndRestore(tf.keras.callbacks.experimental.BackupAndRestore):
  """A BackupAndRestore callback that can be used across multiple model.fit()s."""

  def on_train_end(self, logs=None):
    # don't delete the backup, so it can be used for future model.fit()s
    pass
