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
"""Hparams for model architecture and trainer."""

from __future__ import absolute_import
from __future__ import division
# gtype import
from __future__ import print_function

import ast
import copy
import json
import six


def eval_str_fn(val):
  if val in {'true', 'false'}:
    return val == 'true'
  try:
    return ast.literal_eval(val)
  except ValueError:
    return val


# pylint: disable=protected-access
class Config(object):
  """A config utility class."""

  def __init__(self, config_dict=None):
    self.update(config_dict)

  def __setattr__(self, k, v):
    self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)

  def __getattr__(self, k):
    return self.__dict__[k]

  def __repr__(self):
    return repr(self.as_dict())

  def __str__(self):
    try:
      return json.dumps(self.as_dict(), indent=4)
    except TypeError:
      return str(self.as_dict())

  def _update(self, config_dict, allow_new_keys=True):
    """Recursively update internal members."""
    if not config_dict:
      return

    for k, v in six.iteritems(config_dict):
      if k not in self.__dict__.keys():
        if allow_new_keys:
          self.__setattr__(k, v)
        else:
          raise KeyError('Key `{}` does not exist for overriding. '.format(k))
      else:
        if isinstance(v, dict):
          self.__dict__[k]._update(v, allow_new_keys)
        else:
          self.__dict__[k] = copy.deepcopy(v)

  def get(self, k, default_value=None):
    return self.__dict__.get(k, default_value)

  def update(self, config_dict):
    """Update members while allowing new keys."""
    self._update(config_dict, allow_new_keys=True)

  def override(self, config_dict_or_str):
    """Update members while disallowing new keys."""
    if isinstance(config_dict_or_str, str):
      config_dict = self.parse_from_str(config_dict_or_str)
    elif isinstance(config_dict_or_str, dict):
      config_dict = config_dict_or_str
    else:
      raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

    self._update(config_dict, allow_new_keys=False)

  def parse_from_str(self, config_str):
    """parse from a string in format 'x=a,y=2' and return the dict."""
    if not config_str:
      return {}
    config_dict = {}
    try:
      for kv_pair in config_str.split(','):
        if not kv_pair:  # skip empty string
          continue
        k, v = kv_pair.split('=')
        config_dict[k.strip()] = eval_str_fn(v.strip())
      return config_dict
    except ValueError:
      raise ValueError('Invalid config_str: {}'.format(config_str))

  def as_dict(self):
    """Returns a dict representation."""
    config_dict = {}
    for k, v in six.iteritems(self.__dict__):
      if isinstance(v, Config):
        config_dict[k] = v.as_dict()
      else:
        config_dict[k] = copy.deepcopy(v)
    return config_dict


# pylint: enable=protected-access


def default_detection_configs():
  """Returns a default detection configs."""
  h = Config()

  # model name.
  h.name = 'efficientdet-d1'

  # input preprocessing parameters
  h.image_size = 640
  h.input_rand_hflip = True
  h.train_scale_min = 0.1
  h.train_scale_max = 2.0
  h.autoaugment_policy = None

  # dataset specific parameters
  h.num_classes = 90
  h.skip_crowd_during_training = True

  # model architecture
  h.min_level = 3
  h.max_level = 7
  h.num_scales = 3
  h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
  h.anchor_scale = 4.0
  # is batchnorm training mode
  h.is_training_bn = True
  # optimization
  h.momentum = 0.9
  h.learning_rate = 0.08
  h.lr_warmup_init = 0.008
  h.lr_warmup_epoch = 1.0
  h.first_lr_drop_epoch = 200.0
  h.second_lr_drop_epoch = 250.0
  h.clip_gradients_norm = 10.0
  h.num_epochs = 300

  # classification loss
  h.alpha = 0.25
  h.gamma = 1.5
  # localization loss
  h.delta = 0.1
  h.box_loss_weight = 50.0
  # regularization l2 loss.
  h.weight_decay = 4e-5
  # enable bfloat
  h.use_bfloat16 = True

  # For detection.
  h.box_class_repeats = 3
  h.fpn_cell_repeats = 3
  h.fpn_num_filters = 88
  h.separable_conv = True
  h.apply_bn_for_resampling = True
  h.conv_after_downsample = False
  h.conv_bn_relu_pattern = False
  h.use_native_resize_op = False
  h.pooling_type = None

  # version.
  h.fpn_name = None
  h.fpn_config = None

  # No stochastic depth in default.
  h.survival_prob = None

  h.lr_decay_method = 'cosine'
  h.moving_average_decay = 0.9998
  h.ckpt_var_scope = None  # ckpt variable scope.
  # exclude vars when loading pretrained ckpts.
  h.var_exclude_expr = '.*/class-predict/.*'  # exclude class weights in default

  h.backbone_name = 'efficientnet-b1'
  h.backbone_config = None

  # RetinaNet.
  h.resnet_depth = 50
  return h


efficientdet_model_param_dict = {
    'efficientdet-d0':
        dict(
            name='efficientdet-d0',
            backbone_name='efficientnet-b0',
            image_size=512,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    'efficientdet-d1':
        dict(
            name='efficientdet-d1',
            backbone_name='efficientnet-b1',
            image_size=640,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
        ),
    'efficientdet-d2':
        dict(
            name='efficientdet-d2',
            backbone_name='efficientnet-b2',
            image_size=768,
            fpn_num_filters=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
        ),
    'efficientdet-d3':
        dict(
            name='efficientdet-d3',
            backbone_name='efficientnet-b3',
            image_size=896,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
        ),
    'efficientdet-d4':
        dict(
            name='efficientdet-d4',
            backbone_name='efficientnet-b4',
            image_size=1024,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d5':
        dict(
            name='efficientdet-d5',
            backbone_name='efficientnet-b5',
            image_size=1280,
            fpn_num_filters=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d6':
        dict(
            name='efficientdet-d6',
            backbone_name='efficientnet-b6',
            image_size=1280,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        ),
    'efficientdet-d7':
        dict(
            name='efficientdet-d7',
            backbone_name='efficientnet-b6',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fpn_name='bifpn_sum',  # Use unweighted sum for training stability.
        ),
}


def get_efficientdet_config(model_name='efficientdet-d1'):
  """Get the default config for EfficientDet based on model name."""
  h = default_detection_configs()
  h.override(efficientdet_model_param_dict[model_name])
  return h


retinanet_model_param_dict = {
    'retinanet-50':
        dict(name='retinanet-50', backbone_name='resnet50', resnet_depth=50),
    'retinanet-101':
        dict(name='retinanet-101', backbone_name='resnet101', resnet_depth=101),
}


def get_retinanet_config(model_name='retinanet-50'):
  """Get the default config for EfficientDet based on model name."""
  h = default_detection_configs()
  h.override(
      dict(
          retinanet_model_param_dict[model_name],
          ckpt_var_scope='',
      ))
  # cosine + ema often cause NaN for RetinaNet, so we use the default
  # stepwise without ema used in the original RetinaNet implementation.
  h.lr_decay_method = 'stepwise'
  h.moving_average_decay = 0

  return h


def get_detection_config(model_name):
  if model_name.startswith('efficientdet'):
    return get_efficientdet_config(model_name)
  elif model_name.startswith('retinanet'):
    return get_retinanet_config(model_name)
  else:
    raise ValueError('model name must start with efficientdet or retinanet.')
