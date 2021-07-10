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
"""Hparams for model architecture and trainer."""
import ast
import collections
import copy
from typing import Any, Dict, Text
import tensorflow as tf
import yaml


def eval_str_fn(val):
  if '|' in val:
    return [eval_str_fn(v) for v in val.split('|')]
  if val in {'true', 'false'}:
    return val == 'true'
  try:
    return ast.literal_eval(val)
  except (ValueError, SyntaxError):
    return val


# pylint: disable=protected-access
class Config(dict):
  """A config utility class."""

  def __init__(self, *args, **kwargs):
    super().__init__()
    input_config_dict = dict(*args, **kwargs)
    self.update(input_config_dict)

  def __len__(self):
    return len(self.__dict__)

  def __setattr__(self, k, v):
    if isinstance(v, dict) and not isinstance(v, Config):
      self.__dict__[k] = Config(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __getattr__(self, k):
    return self.__dict__[k]

  def __setitem__(self, k, v):
    self.__setattr__(k, v)

  def __getitem__(self, k):
    return self.__dict__[k]

  def __iter__(self):
    for key in self.__dict__:
      yield key

  def items(self):
    for key, value in self.__dict__.items():
      yield key, value

  def __repr__(self):
    return repr(self.as_dict())

  def __getstate__(self):
    return self.__dict__

  def __copy__(self):
    cls = self.__class__
    result = cls.__new__(cls)
    result.__dict__.update(self.__dict__)
    return result

  def __deepcopy__(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    for k, v in self.__dict__.items():
      result[k] = v
    return result

  def __str__(self):
    try:
      return yaml.dump(self.as_dict(), indent=4)
    except TypeError:
      return str(self.as_dict())

  def _update(self, config_dict, allow_new_keys=True):
    """Recursively update internal members."""
    if not config_dict:
      return

    for k, v in config_dict.items():
      if k not in self.__dict__:
        if allow_new_keys:
          self.__setattr__(k, v)
        else:
          raise KeyError('Key `{}` does not exist for overriding. '.format(k))
      else:
        if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
          self.__dict__[k]._update(v, allow_new_keys)
        elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
          self.__dict__[k]._update(v.as_dict(), allow_new_keys)
        else:
          self.__setattr__(k, v)

  def get(self, k, default_value=None):
    return self.__dict__.get(k, default_value)

  def update(self, config_dict):
    """Update members while allowing new keys."""
    self._update(config_dict, allow_new_keys=True)

  def keys(self):
    return self.__dict__.keys()

  def override(self, config_dict_or_str, allow_new_keys=False):
    """Update members while disallowing new keys."""
    if not config_dict_or_str:
      return
    if isinstance(config_dict_or_str, str):
      if '=' in config_dict_or_str:
        config_dict = self.parse_from_str(config_dict_or_str)
      elif config_dict_or_str.endswith('.yaml'):
        config_dict = self.parse_from_yaml(config_dict_or_str)
      else:
        raise ValueError(
            'Invalid string {}, must end with .yaml or contains "=".'.format(
                config_dict_or_str))
    elif isinstance(config_dict_or_str, dict):
      config_dict = config_dict_or_str
    else:
      raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

    self._update(config_dict, allow_new_keys)

  def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
      config_dict = yaml.load(f, Loader=yaml.FullLoader)
      return config_dict

  def save_to_yaml(self, yaml_file_path):
    """Write a dictionary into a yaml file."""
    with tf.io.gfile.GFile(yaml_file_path, 'w') as f:
      yaml.dump(self.as_dict(), f, default_flow_style=False)

  def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
    """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
    if not config_str:
      return {}
    config_dict = {}
    try:
      for kv_pair in config_str.split(','):
        if not kv_pair:  # skip empty string
          continue
        key_str, value_str = kv_pair.split('=')
        key_str = key_str.strip()

        def add_kv_recursive(k, v):
          """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
          if '.' not in k:
            return {k: eval_str_fn(v)}
          pos = k.index('.')
          return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}

        def merge_dict_recursive(target, src):
          """Recursively merge two nested dictionary."""
          for k in src.keys():
            if ((k in target and isinstance(target[k], dict) and
                 isinstance(src[k], collections.abc.Mapping))):
              merge_dict_recursive(target[k], src[k])
            else:
              target[k] = src[k]

        merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
      return config_dict
    except ValueError:
      raise ValueError('Invalid config_str: {}'.format(config_str))

  def as_dict(self):
    """Returns a dict representation."""
    config_dict = {}
    for k, v in self.__dict__.items():
      if isinstance(v, Config):
        config_dict[k] = v.as_dict()
      elif isinstance(v, (list, tuple)):
        config_dict[k] = [
            i.as_dict() if isinstance(i, Config) else copy.deepcopy(i)
            for i in v
        ]
      else:
        config_dict[k] = copy.deepcopy(v)
    return config_dict
    # pylint: enable=protected-access


registry_map = {}


def register(cls, prefix='effnet:'):
  """Register a function, mainly for config here."""
  registry_map[prefix + cls.__name__.lower()] = cls
  return cls


def lookup(name, prefix='effnet:') -> Any:
  name = prefix + name.lower()
  if name not in registry_map:
    raise ValueError(f'{name} not registered: {registry_map.keys()}')
  return registry_map[name]


base_config = Config(
    # model related params.
    model=dict(
        model_name='efficientnet-b0',
        data_format='channels_last',
        feature_size=1280,
        bn_type=None,   # 'tpu_bn',
        bn_momentum=0.9,
        bn_epsilon=1e-3,
        gn_groups=8,
        depth_divisor=8,
        min_depth=8,
        act_fn='silu',
        survival_prob=0.8,
        local_pooling=False,
        headbias=None,
        conv_dropout=None,
        dropout_rate=None,
        depth_coefficient=None,
        width_coefficient=None,
        blocks_args=None,
        num_classes=1000,  # must be the same as data.num_classes
    ),
    # train related params.
    train=dict(
        stages=0,
        epochs=350,
        min_steps=0,
        optimizer='rmsprop',
        lr_sched='exponential',
        lr_base=0.016,
        lr_decay_epoch=2.4,
        lr_decay_factor=0.97,
        lr_warmup_epoch=5,
        lr_min=0,
        ema_decay=0.9999,
        weight_decay=1e-5,
        weight_decay_inc=0.0,
        weight_decay_exclude='.*(bias|gamma|beta).*',
        label_smoothing=0.1,
        gclip=0,
        batch_size=4096,
        isize=None,
        split=None,  # dataset split, default to 'train'
        loss_type=None,  # loss type: sigmoid or softmax
        ft_init_ckpt=None,
        ft_init_ema=True,
        varsexp=None,  # trainable variables.
        sched=None,  # schedule
    ),
    eval=dict(
        batch_size=8,
        isize=None,  # image size
        split=None,  # dataset split, default to 'eval'
    ),
    # data related params.
    data=dict(
        ds_name='imagenet',
        augname='randaug',  # or 'autoaug'
        ra_num_layers=2,
        ram=15,
        mixup_alpha=0.,
        cutmix_alpha=0.,
        ibase=128,
        cache=True,
        resize=None,
        data_dir=None,
        multiclass=None,
        num_classes=1000,
        tfds_name=None,
        try_gcs=False,
        tfds_split=None,
        splits=dict(
            train=dict(
                num_images=None, files=None, tfds_split=None, slice=None),
            eval=dict(num_images=None, files=None, tfds_split=None, slice=None),
            minival=dict(
                num_images=None, files=None, tfds_split=None, slice=None),
            trainval=dict(
                num_images=None, files=None, tfds_split=None, slice=None),
        ),
    ),
    runtime=dict(
        iterations_per_loop=1000,  # larger value has better utilization.
        skip_host_call=False,
        mixed_precision=True,
        use_async_checkpointing=False,
        log_step_count_steps=64,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=5,
        strategy='tpu',  # None, gpu, tpu
    ))
