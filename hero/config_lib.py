# Copyright 2024 Chen Liang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Experiments and sharding configs."""

import dataclasses
import json
from typing import Any, Tuple


################################################################################
# Config registries.


class ConfigRegistry():
  """Base class for config registries."""
  registry = {}
  OVERWRITE_DUPLICATE = False

  @classmethod
  def register(cls, config_cls):
    config_name = config_cls.__name__
    if config_name in cls.registry and not cls.OVERWRITE_DUPLICATE:
      raise ValueError(f'Duplicate config name: {config_name}')
    cls.registry[config_name] = config_cls
    return config_cls

  @classmethod
  def unregister(cls, config_name):
    if config_name in cls.registry:
      del cls.registry[config_name]

  @classmethod
  def get_config(cls, config_name):
    return cls.registry[config_name]()


class ExperimentConfigRegistry(ConfigRegistry):
  registry = {}


class ShardingConfigRegistry(ConfigRegistry):
  registry = {}


def serialize_config(config):
  data = dataclasses.asdict(config)
  type_name = type(config).__name__.lower()
  return json.dumps({'type': type_name, 'data': data})


################################################################################
# Sharding Configs.


@ShardingConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class GSPMDSharding():
  """Sharding config for GSPMD."""
  # Shape (model_dim, model_dim * expansion_factor)
  ffn0_partition: Any = ('data', 'model')

  # Shape (model_dim, model_dim * expansion_factor)
  ffn1_partition: Any = ('model', 'data')

  # Shape (model_dim, num_heads, per_head_size)
  attn_qkv_partition: Any = ('data', 'model', None)

  # Shape (model_dim, num_heads, per_head_size)
  attn_o_partition: Any = ('data', 'model', None)

  # Shape (vocab_size, model_dim)
  embed_partition: Any = ('model', 'data')

  # Shape (batch_size, seq_len, num_heads, per_head_size)
  attn_activation_partition: Any = (('replica', 'data'), None, 'model', None)

  # Shape (batch_size, seq_len, model_dim)
  activation_partition: Any = (('replica', 'data'), None, 'model')

  # Shape (batch_size, seq_len, model_dim * expansion_factor)
  ffn0_activation_partition: Any = (('replica', 'data'), None, 'model')

  # Shape (batch_size, seq_len, vocab_size)
  logits_partition: Any = (('replica', 'data'), None, 'model')

  # Shape (batch_size, seq_len)
  data_partition: Any = (('replica', 'data'), None)


@ShardingConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class DataParallelSharding():
  """Sharding config for data parallel."""
  # Shape (model_dim, model_dim * expansion_factor)
  ffn0_partition: Any = (None, None)

  # Shape (model_dim, model_dim * expansion_factor)
  ffn1_partition: Any = (None, None)

  # Shape (model_dim, num_heads, per_head_size)
  attn_qkv_partition: Any = (None, None, None)

  # Shape (model_dim, num_heads, per_head_size)
  attn_o_partition: Any = (None, None, None)

  # Shape (vocab_size, model_dim)
  embed_partition: Any = (None, None)

  # Shape (b, l, num_heads, per_head_size)
  attn_activation_partition: Any = (
      ('replica', 'data', 'model'), None, None, None)

  # Shape (batch_size, seq_len, model_dim)
  activation_partition: Any = (('replica', 'data', 'model'), None, None)

  # Shape (batch_size, seq_len, model_dim * expansion_factor)
  ffn0_activation_partition: Any = (('replica', 'data', 'model'), None, None)

  # Shape (batch_size, seq_len, vocab_size)
  logits_partition: Any = (('replica', 'data', 'model'), None, None)

  # Shape (batch_size, seq_len)
  data_partition: Any = (('replica', 'data', 'model'), None)


################################################################################
# Experiment Configs.


################################################################################
## Base experiment for others to inherit.


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
  """Base experiment config for others to inherit."""


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class BaseExperimentConfig(ExperimentConfig):
  """Base experiment config for training a 1.7B Transformer model LM1B."""
  # Example experiment config.
  seq_len: int = 1024
  vocab_size: int = 32_000
  model_dim: int = 2048
  per_head_dim: int = 128
  n_heads: int = 16
  n_layers: int = 14
  expand_factor: int = 8
  use_scan: bool = True
  model_seed: int = 42
  use_rmsnorm: bool = True
  use_pre_ln: bool = True
  use_post_ln: bool = True
  use_post_skip_ln: bool = False
  use_per_dim_scale: bool = True
  use_gated_activation_in_ffn: bool = True
  activation_dtype_name: str = 'bfloat16'
  use_flash_attention: bool = False
  window_size: int = 0
  use_window_chunk: bool = False
  n_kv_heads: int = 0

  # Data config
  batch_size: int = 64 * 16
  dataset_name: str = 'lm1b'
  dataset_seed: int = 42
  use_packing: bool = True
  use_validation_set: bool = False
  # How many steps / validation examples to evaluate on,
  # set to -1 to use whole set
  validation_num_eval_steps: int = -1
  # How often to run evaluation on validation set.
  validation_eval_interval: int = 1000
  # Batch size for evaluation on validation set,
  # set to -1 to use the same as `batch_size`.
  validation_eval_batch_size: int = -1
  feature_converter_name: str = 'LMFeatureConverter'

  # Training config
  optimizer_name: str = 'Adam'
  optimizer_config: Tuple[Tuple[str, Any], ...] = (
      ('beta1', 0.9), ('beta2', 0.95), ('epsilon', 1e-8))
  weight_decay: float = 1e-3

  num_train_steps: int = 100_000
  lr_schedule_name: str = 'cosine_decay'
  lr_schedule_config: Tuple[Tuple[str, Any], ...] = (
      ('lr', 1e-3),
      ('warmup_steps', 1_000), ('steps_after_decay', 0), ('end_decay', 0.1))
  clip_grad_norm: float = 1.0
  clip_update_norm: float = -1.0
  clip_local_update_rms: float = 1.0

  # Checkpoint and tensorboard config
  ckpt_interval: int = 1000
  ckpt_max_to_keep: int = 3
  tb_log_interval: int = 100
  log_additional_info: bool = True

  # Config for init from existing checkpoint.
  init_ckpt_dir: str = ''
  init_ckpt_step: int = -1
  init_ckpt_opt_state: bool = False
  reset_steps: bool = False

  # Add masks to only calculate loss on assistant responses.
  add_chat_loss_mask: bool = False
  mask_start_token: str = ''
  mask_end_token: str = ''
  vocab_path: str = ''

  # Name for the model, i.e., the main module.
  model_name: str = 'TransformerLM'


################################################################################
# Small C4 experiments with settings similar to the Chinchilla paper:
# https://arxiv.org/abs/2203.15556.


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops2e17TFM41MC4L2048BS128(BaseExperimentConfig):
  """C4 experiment with 2e17 flops."""
  # num_params: 40.645632 M
  # num_non_embedding_params: 14.824448 M
  # num_embedding_params: 25.821184 M
  # embedding_params_ratio: 0.6352757413145895
  # num_tokens: 678.428672 M
  # num_tokens / num_params: 16.691305771798554
  # num_tokens / num_non_embedding_params: 45.764177661117635
  # num_flops: 1.6545097284216422e+17
  # Fitted optimal ratio for 1.7e17: 16.69
  model_dim: int = 256  # 2048 // 8
  per_head_dim: int = 32  # 256 // 8
  n_heads: int = 8
  n_layers: int = 8  # 18 // 2 - 1
  expand_factor: int = 8
  vocab_size: int = 100_864
  seq_len: int = 2048

  # 40645632 * 16.69 / 2048 / 128 = 2588 steps
  dataset_name: str = 'c4.vb100864_openmix_v1'
  batch_size: int = 128
  num_train_steps: int = 2588
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 1e-1
  lr_schedule_config: Tuple[Tuple[str, Any], ...] = (
      ('lr', 3e-3), ('steps_after_decay', 0),
      ('end_decay', 0.1))

  ckpt_max_to_keep: int = 1

  use_validation_set: bool = True
  validation_num_eval_steps: int = 16
  validation_eval_interval: int = 500
  validation_eval_batch_size: int = 128


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops1e18TFM111MC4L2048BS256(Flops2e17TFM41MC4L2048BS128):
  """C4 experiment with 1e18 flops."""
  # num_params: 110.550528 M
  # num_non_embedding_params: 58.90816 M
  # num_embedding_params: 51.642368 M
  # embedding_params_ratio: 0.467138139765375
  # num_tokens: 1901.068288 M
  # num_tokens / num_params: 17.19637456638832
  # num_tokens / num_non_embedding_params: 32.27173091130329
  # num_flops: 1.2609846180147364e+18
  # Predicted optimal ratio for 1.3e18: 17.2
  model_dim: int = 512
  per_head_dim: int = 64
  n_heads: int = 8
  n_layers: int = 8
  expand_factor: int = 8
  vocab_size: int = 100_864

  # 110550528 * 17.2 / 2048 / 256 = 3626 steps
  batch_size: int = 256
  num_train_steps: int = 3626
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 1e-1
  lr_schedule_config: Tuple[Tuple[str, Any], ...] = (
      ('lr', 3e-3), ('steps_after_decay', 0),
      ('end_decay', 0.1))

  use_validation_set: bool = True
  validation_num_eval_steps: int = 8
  validation_eval_interval: int = 500
  validation_eval_batch_size: int = 256


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class Flops1e19TFM338MC4L2048BS512(Flops2e17TFM41MC4L2048BS128):
  """C4 experiment with 1e19 flops."""
  # num_params: 338.440192 M
  # num_non_embedding_params: 235.155456 M
  # num_embedding_params: 103.284736 M
  # embedding_params_ratio: 0.3051786946155615
  # num_tokens: 6149.89824 M
  # num_tokens / num_params: 18.171299938276835
  # num_tokens / num_non_embedding_params: 26.15247948999321
  # num_flops: 1.2488236446756372e+19
  # Fitted optimal ratio for 1.2e19: 17.97
  model_dim: int = 1024
  per_head_dim: int = 128
  n_heads: int = 8
  n_layers: int = 8
  expand_factor: int = 8
  vocab_size: int = 100_864

  # 338440192 * 17.97 / 2048 / 512 = 5800 steps
  batch_size: int = 512
  num_train_steps: int = 5800
  lr_schedule_name: str = 'cosine_decay'
  weight_decay: float = 1e-1
  lr_schedule_config: Tuple[Tuple[str, Any], ...] = (
      ('lr', 3e-3),
      ('steps_after_decay', 0),
      ('warmup_steps', 1_000),
      ('end_decay', 0.1))

  use_validation_set: bool = True
  validation_num_eval_steps: int = 4
  validation_eval_interval: int = 500
  validation_eval_batch_size: int = 512


################################################################################
## Tiny experiments for tests.


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TransformerLMTest(BaseExperimentConfig):
  """Tiny experiment for local tests."""
  # Model config
  model_dim: int = 8
  per_head_dim: int = 4
  n_heads: int = 2
  n_layers: int = 2
  expand_factor: int = 2
  use_scan: bool = True
  use_flash_attention: bool = False
  activation_dtype_name: str = 'bfloat16'

  # Data config
  num_train_steps: int = 2000
  batch_size: int = 4

  vocab_size: int = 32_000
  seq_len: int = 64
  dataset_name: str = 'imdb_reviews.vb32000_t5_cc'
  lr_schedule_name: str = 'cosine_decay'
  lr_schedule_config: Tuple[Tuple[str, Any], ...] = (
      ('lr', 1e-3),
      ('warmup_steps', 100), ('steps_after_decay', 10), ('end_decay', 0.1))
  clip_grad_norm: float = -1.0
  clip_update_norm: float = -1.0
  use_validation_set: bool = True
  validation_num_eval_steps: int = 2
  validation_eval_interval: int = 5
  validation_eval_batch_size: int = -1

  # Checkpoint and tensorboard config
  ckpt_interval: int = 10
  ckpt_max_to_keep: int = 3
  tb_log_interval: int = 2


@ExperimentConfigRegistry.register
@dataclasses.dataclass(frozen=True)
class TransformerLMTestNoScan(TransformerLMTest):
  """Tiny experiment for local tests without scan."""
  use_scan: bool = False
