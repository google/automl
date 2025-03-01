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
"""All modeling components including architecture, training and inference."""

import abc
import collections
import copy
import dataclasses
import functools
import json
import os
import time
from typing import Any, Callable, final, Mapping, MutableMapping, Optional, Protocol, Tuple, Union

from absl import logging
from clu import metric_writers
import einops
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
import orbax.checkpoint as ocp
import config_lib
import yaml




################################################################################
# Global variables.

################################################################################
## Global constants.
MESH_SHAPE = None
DCN_MESH_SHAPE = None

################################################################################
## Type aliases.
DTypeLike = jax.typing.DTypeLike
PartitionAnnotation = Optional[Tuple[Optional[str], ...]]
PRNGKey = jax.Array
PyTree = Any
SimplyConfig = Any
Batch = MutableMapping[str, Union[np.ndarray, jnp.ndarray]]


################################################################################
# Module registries.


class NameRegistry():
  """Base class for registries."""
  registry = {}
  OVERWRITE_DUPLICATE = False

  @classmethod
  def register(cls, fn, name=None):
    if name is None:
      name = fn.__name__
    if name in cls.registry and not cls.OVERWRITE_DUPLICATE:
      raise ValueError(f'Duplicate name: {name}')
    cls.registry[name] = fn
    return fn

  @classmethod
  def unregister(cls, name):
    if name in cls.registry:
      del cls.registry[name]

  @classmethod
  def get(cls, name):
    return cls.registry[name]


class OptimizerRegistry(NameRegistry):
  registry = {}


class ModelRegistry(NameRegistry):
  registry = {}


################################################################################
# Mesh.


def get_mesh_shape(num_devices):
  log2_num_devices = np.log(num_devices) / np.log(2)
  assert (log2_num_devices).is_integer()
  half = log2_num_devices // 2
  if log2_num_devices % 2 == 0:
    return (int(2 ** half), int(2 ** half))
  else:
    return (int(2 ** half), int(2 ** (half + 1)))


def get_default_mesh(mesh_shape=None, dcn_mesh_shape=None,
                     print_debug_info=False):
  """Creates a mesh for the given mesh_shape and dcn_mesh_shape."""
  num_devices = len(jax.devices())
  if mesh_shape is None:
    mesh_shape = get_mesh_shape(num_devices)
  if len(mesh_shape) == 2:
    mesh_shape = (1,) + mesh_shape
  if dcn_mesh_shape is not None and dcn_mesh_shape[0] > 1:
    if print_debug_info:
      print(f'hybrid, ici mesh_shape: {mesh_shape}')
      print(f'hybrid, dcn_mesh_shape: {dcn_mesh_shape}')
    devices = mesh_utils.create_hybrid_device_mesh(mesh_shape, dcn_mesh_shape)
  else:
    devices = mesh_utils.create_device_mesh(mesh_shape)
  return js.Mesh(devices, axis_names=('replica', 'data', 'model'))


def mesh_sharding(
    pspec: Optional[Tuple[Union[Optional[str],
                                Tuple[Optional[str], ...]], ...]] = None,
    mesh: Optional[js.Mesh] = None,
    ) -> js.Sharding:
  if mesh is None:
    mesh = get_default_mesh(MESH_SHAPE, DCN_MESH_SHAPE)
  if pspec is None:
    return js.NamedSharding(mesh, js.PartitionSpec())
  else:
    return js.NamedSharding(mesh, js.PartitionSpec(*pspec))


################################################################################
# Initialization.


def xavier_init(prng_key, shape, dtype, in_dim, out_dim):
  scale = jnp.sqrt(6 / (in_dim + out_dim))
  return jax.random.uniform(
      prng_key, shape, dtype=dtype, minval=-1.0, maxval=1.0) * jnp.array(
          scale, dtype=dtype)


################################################################################
# Architecture.


def gelu(x: jax.Array):
  return 0.5 * x * (1.0 + jnp.tanh(
      jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))


def squared_relu(x: jax.Array):
  return jnp.square(jax.nn.relu(x))


def soft_cap(x: jax.Array, cap: float):
  cap = jnp.asarray(cap, x.dtype)
  return jnp.asarray(cap * jnp.tanh(x / cap), x.dtype)


class SimplyModule:
  """An untra-simplified version of `flax.nn.Module`."""

  @final
  def __post_init__(self):
    """Call setup() to initialize the module and setup any attributes."""
    # `__post_init__` is set to be `final` to avoid accidental overriding, if
    # you need to add additional setup logic, please use `setup` instead.
    self.setup()

  # Added as a workaround to calm down pytypecheck.
  def __getattr__(self, name: str) -> Any:
    """Call setup() before getting any setup-defined attributes."""
    raise AttributeError(f'Unknown attribute: {name}')

  # We assume multiple calls to `setup()` should have the same effect as a
  # single call.
  def setup(self) -> None:
    """Setup any attributes. Typically used for instantiating sub-modules."""

  def init(self, *args, **kwargs) -> PyTree:
    """initialize the parameters associated with the module."""

  def apply(self, *args, **kwargs) -> Any:
    """Run forward pass of the module with parameters and inputs."""


@dataclasses.dataclass
class Embedding(SimplyModule):
  """Embedding layer."""
  vocab_size: int
  dim: int
  var_scale: float = 1.0
  lookup_scale: float = 1.0
  use_lookup: bool = True
  # Mixed precision related.
  weight_dtype: DTypeLike = jnp.float32
  activation_dtype: DTypeLike = jnp.bfloat16
  # Sharding related.
  partition: PartitionAnnotation = None

  def init(self, prng_key: PRNGKey) -> PyTree:
    scaling_factor = (self.var_scale / jnp.sqrt(self.dim)).astype(
        self.weight_dtype)
    result = jax.random.normal(
        prng_key, shape=[self.vocab_size, self.dim],
        dtype=self.weight_dtype) * scaling_factor
    result = jax.lax.with_sharding_constraint(
        result, mesh_sharding(self.partition))
    return result

  def apply(self, params: PyTree, x: jax.Array) -> jax.Array:
    # Make the variance of the lookup value to be lookup_scale.
    # This is added so that the value has different scale when used as inputs
    # versus softmax weights.
    scaling_factor = (
        self.lookup_scale / self.var_scale * jnp.sqrt(self.dim)
        ).astype(params.dtype)
    if self.use_lookup:
      output = jnp.take(params, x, axis=0)
    else:
      onehot_x = jax.nn.one_hot(x, self.vocab_size, dtype=x.dtype)
      output = jnp.einsum('ij,...i->...j', params, onehot_x)
    return output * scaling_factor


@dataclasses.dataclass
class Linear(SimplyModule):
  """Linear layer."""
  input_dim: int
  output_dim: int
  use_bias: bool = True
  # Mixed precision related.
  weight_dtype: DTypeLike = jnp.float32
  activation_dtype: DTypeLike = jnp.bfloat16
  # Sharding related.
  weight_partition: PartitionAnnotation = None
  output_partition: PartitionAnnotation = None
  # Others.
  weight_name: str = 'w'
  bias_name: str = 'b'
  use_external_weights: bool = False

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    if not self.use_external_weights:
      params[self.weight_name] = xavier_init(
          prng_key, shape=[self.input_dim, self.output_dim],
          in_dim=self.input_dim, out_dim=self.output_dim,
          dtype=self.weight_dtype)
      params[self.weight_name] = jax.lax.with_sharding_constraint(
          params[self.weight_name], mesh_sharding(self.weight_partition))

    if self.use_bias:
      params[self.bias_name] = jnp.zeros(
          shape=[self.output_dim], dtype=self.weight_dtype)
      params[self.bias_name] = jax.lax.with_sharding_constraint(
          params[self.bias_name],
          mesh_sharding(
              (self.weight_partition[-1],) if self.weight_partition else None))
    return params

  def apply(self, params: PyTree, x: jax.Array) -> jax.Array:
    w = maybe_dequantize_array(
        params[self.weight_name], dtype=self.activation_dtype)
    output = jnp.einsum('ij,...i->...j', w, x)
    if self.use_bias:
      b = maybe_dequantize_array(
          params[self.bias_name], dtype=self.activation_dtype)
      output += b
    output = jax.lax.with_sharding_constraint(
        output, mesh_sharding(self.output_partition))
    return output


@dataclasses.dataclass
class LayerNorm(SimplyModule):
  """Layer normalization layer (can be also configured as RMSNorm)."""
  dim: int
  axis: int = -1
  use_bias: bool = True  # Set to False if want to use RMSNorm.
  use_scale: bool = True
  # Mixed precision related.
  weight_dtype: DTypeLike = 'float32'
  activation_dtype: DTypeLike = 'bfloat16'
  # Sharding related.
  scale_partition: PartitionAnnotation = None
  bias_partition: PartitionAnnotation = None
  # Others.
  epsilon: float = 1e-6

  def init(self, prng_key: PRNGKey | None = None) -> PyTree:
    del prng_key
    assert self.use_bias or self.use_scale
    params = {}
    if self.use_bias:
      params['bias'] = jnp.zeros(self.dim, dtype=self.weight_dtype)
      params['bias'] = jax.lax.with_sharding_constraint(
          params['bias'], mesh_sharding(self.bias_partition))
    if self.use_scale:
      params['scale'] = jnp.zeros(self.dim, dtype=self.weight_dtype)
      params['scale'] = jax.lax.with_sharding_constraint(
          params['scale'], mesh_sharding(self.scale_partition))
    return params

  def apply(self, params: PyTree, x: jax.Array) -> jax.Array:
    inputs_dtype = x.dtype
    # Perform reduction in float32 for better stability.
    x = x.astype(jnp.float32)
    if self.use_bias:
      mean = jnp.mean(x, axis=self.axis, keepdims=True)
      x -= mean
    if self.use_scale:
      var = jnp.mean(jnp.square(x), axis=self.axis, keepdims=True)
      var = jnp.asarray(var, self.activation_dtype)
      x = jnp.asarray(x, self.activation_dtype)
      epsilon = jnp.asarray(self.epsilon, self.activation_dtype)
      # Temporarily convert to float32 to run rsqrt.
      if not jnp.issubdtype(var.dtype, jnp.floating):
        var = jnp.asarray(var, jnp.float32)
        x = jnp.asarray(x, jnp.float32)
        epsilon = jnp.asarray(self.epsilon, jnp.float32)
      x *= jax.lax.rsqrt(var + epsilon)
      x = jnp.asarray(x, self.activation_dtype)
      x *= jnp.asarray(1.0, self.activation_dtype) + params['scale']
    if self.use_bias:
      x += params['bias']
    x = x.astype(inputs_dtype)
    return x


@dataclasses.dataclass
class PerDimScale(SimplyModule):
  """Layer to scale individual dims of the input."""
  dim: int
  axis: int = -1
  # Mixed precision related.
  weight_dtype: DTypeLike = jnp.float32
  activation_dtype: DTypeLike = jnp.bfloat16

  def init(self) -> PyTree:
    params = {}
    params['scale'] = jnp.zeros(self.dim, dtype=self.weight_dtype)
    return params

  def apply(self, params: PyTree, x: jax.Array) -> jax.Array:
    r_softplus_0 = 1.442695041
    scaling_factor = jnp.array(
        r_softplus_0 / jnp.sqrt(self.dim), dtype=self.activation_dtype)
    scaling_factor *= jax.nn.softplus(params['scale'])
    x *= scaling_factor
    return x


def get_large_negative_value(dtype: DTypeLike):
  if jnp.issubdtype(dtype, jnp.inexact):
    dtype_max = jnp.finfo(dtype).max
  elif jnp.issubdtype(dtype, jnp.integer):
    dtype_max = jnp.iinfo(dtype).max
  else:
    raise ValueError(f'Unsupported dtype: {dtype}')
  return jnp.asarray(-0.7 * dtype_max, dtype)


def rotary_positional_embedding(
    embedding_mat, segment_positions=None,
    min_timescale=1, max_timescale=10_000):
  """Add rotary positional embedding (rope) to the given embedding matrix."""
  embedding_dims = embedding_mat.shape[-1]
  half_embedding_dim = embedding_dims // 2
  fraction = 2 * jnp.arange(0, half_embedding_dim) / embedding_dims
  timescale = min_timescale * (max_timescale / min_timescale)**fraction
  query_segment_pos = segment_positions
  if query_segment_pos is None:
    seq_length = embedding_mat.shape[1]
    query_segment_pos = jnp.arange(
        seq_length, dtype=jnp.float32)[jnp.newaxis, :]
  else:
    query_segment_pos = jnp.asarray(query_segment_pos, dtype=jnp.float32)
  query_segment_pos = query_segment_pos[:, :, jnp.newaxis, jnp.newaxis]
  timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
  sinusoid_inp = query_segment_pos / timescale
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)
  # Convert to float32.
  embedding_dtype = embedding_mat.dtype
  embedding_mat = jnp.asarray(embedding_mat, jnp.float32)
  first_half, second_half = jnp.split(embedding_mat, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  embedding_mat = jnp.concatenate([first_part, second_part], axis=-1)
  # Convert back to original dtype.
  embedding_mat = jnp.asarray(embedding_mat, embedding_dtype)
  return embedding_mat


def create_mask(
    *, seq_len=None, segment_ids=None, segment_positions=None,
    use_causal=True, window_size=0, dtype=jnp.float32):
  """Create a mask for attention.

  Args:
    seq_len: The sequence length.
    segment_ids: The segment ids.
    segment_positions: The segment positions.
    use_causal: Whether to use causal attention.
    window_size: Attends how many tokens ahead (excluding self). Used when
      greater than 0 and use_causal is True.
    dtype: The dtype of the mask.

  Returns:
    The mask in dtype of shape [batch_size, seq_len, seq_len], with 0 as
    attendable and 1 as unattendabe.
  """
  if seq_len is not None:
    l = seq_len
  elif segment_ids is not None:
    l = segment_ids.shape[-1]
  elif segment_positions is not None:
    l = segment_positions.shape[-1]
  else:
    raise ValueError(
        'You must provide one of seq_len, segment_ids or segment_positions to'
        ' compute the mask.'
    )

  # From here, until the return line, the masks are taking 1 as attendable and
  # 0 as unattendabe. We flip it at the return line.
  masks = []
  if segment_ids is not None:
    a = einops.rearrange(segment_ids, '... l -> ... l 1')
    b = einops.rearrange(segment_ids, '... l -> ... 1 l')
    seg_mask = (a == b).astype(dtype)
    masks.append(seg_mask)
  if use_causal:
    pos = segment_positions
    if pos is None:
      pos = jnp.arange(l)[jnp.newaxis, :]
    a = einops.rearrange(pos, 'b l -> b l 1')
    b = einops.rearrange(pos, 'b l -> b 1 l')
    causal_mask = (a >= b).astype(dtype)
    masks.append(causal_mask)
    if window_size > 0 and window_size + 1 < l:
      window_mask = (a - b <= window_size).astype(dtype)
      masks.append(window_mask)

  if masks:
    mask = masks[0]
    for m in masks[1:]:
      mask = mask * m
  else:
    mask = jnp.ones(shape=(1, l, l), dtype=dtype)
  return (1 - mask).astype(dtype)


def chunked_local_attn(q, k, v, mask, window_size, dtype=jnp.bfloat16):
  """Chunked local attention.

  It splits the sequence into chunks of size window_size and performs local
  attention within each chunk, i.e. query i-th chunk attends to key/value in
  (i-1)-th and i-th chunks, in order to reduce unnecessary computation.

  Args:
    q: The query in [batch_size, seq_len, num_heads, model_dim].
    k: The key in [batch_size, seq_len, num_heads, model_dim].
    v: The value in [batch_size, seq_len, num_heads, model_dim].
    mask: The mask in [batch_size, num_heads, seq_len, seq_len].
    window_size: The size of the sliding window.
    dtype: The dtype of the output.

  Returns:
    The output of the attention.
  """
  seq_len = k.shape[1]
  assert seq_len % window_size == 0
  chunked_q = einops.rearrange(q, 'b (c w) ... -> b c w ...', w=window_size)
  chunked_k = einops.rearrange(k, 'b (c w) ... -> b c w ...', w=window_size)
  chunked_v = einops.rearrange(v, 'b (c w) ... -> b c w ...', w=window_size)

  chunked_mask = einops.rearrange(
      mask,
      'b ... (c1 w1) (c2 w2) -> b c1 c2 ... w1 w2',
      w1=window_size,
      w2=window_size,
  )

  # output0: [batch_size, window_size, num_heads, model_dim]
  output0, _ = attn(
      chunked_q[:, 0],
      chunked_k[:, 0],
      chunked_v[:, 0],
      chunked_mask[:, 0, 0],
      dtype=dtype,
  )

  # Prepare k/v and mask for concantation of (i-1)-th and i-th chunks.
  # Chunked mask is implemented by taking the diagnal using einsum.
  # chunked_mask0 (current chunk) and chunked_mask1 (previous chunk):
  #   [batch_size, num_chunks-1, num_heads, window_size, window_size]
  chunked_mask0 = jnp.einsum('bcc...->bc...', chunked_mask[:, 1:, 1:])
  chunked_mask1 = jnp.einsum('bcc...->bc...', chunked_mask[:, 1:, :-1])
  # w2_chunked_mask:
  #   [batch_size, num_chunks-1, num_heads, window_size, 2*window_size]
  w2_chunked_mask = jnp.concat([chunked_mask1, chunked_mask0], axis=-1)
  # w2_chunked_k and w2_chunked_v:
  #   [batch_size, num_chunks-1, 2*window_size, num_heads, model_dim]
  w2_chunked_k = jnp.concat([chunked_k[:, :-1], chunked_k[:, 1:]], axis=2)
  w2_chunked_v = jnp.concat([chunked_v[:, :-1], chunked_v[:, 1:]], axis=2)

  # chunked_output1:
  #   [batch_size, num_chunks-1, window_size, num_heads, model_dim]
  chunked_output1, _ = attn(
      chunked_q[:, 1:],
      w2_chunked_k,
      w2_chunked_v,
      w2_chunked_mask,
      dtype=dtype,
  )
  # output1: [batch_size, (num_chunks-1)*window_size, num_heads, model_dim]
  output1 = einops.rearrange(
      chunked_output1, 'b c w ... -> b (c w) ...'
  )

  # output: [batch_size, seq_len, num_heads, model_dim]
  output = jnp.concat([output0, output1], axis=1)
  return output


def attn(q, k, v, mask, dtype='bfloat16'):
  """Apply multi-head attention."""
  group_axis = 'g' if len(q.shape) > len(k.shape) else ''
  attn_logit_mat = jnp.einsum(
      f'...t{group_axis}hi,...qhi->...{group_axis}htq', q, k
  ).astype(jnp.float32)
  attn_logit_mat = soft_cap(attn_logit_mat, 50.0)
  attn_logit_mat += mask * get_large_negative_value(attn_logit_mat.dtype)
  attn_mat = jax.nn.softmax(attn_logit_mat, axis=-1)
  attn_mat = attn_mat.astype(dtype)
  output = jnp.einsum(
      f'...{group_axis}htq,...qhi->...t{group_axis}hi', attn_mat, v
  )
  return output, attn_mat


@dataclasses.dataclass
class Attention(SimplyModule):
  """Standard Multi-head Attention layer."""
  model_dim: int
  n_heads: int
  per_head_dim: int
  use_causal: bool = True
  add_extra_output: bool = False
  use_per_dim_scale: bool = False
  use_combined_qkv: bool = True
  # Mixed precision related.
  activation_dtype: DTypeLike = jnp.bfloat16
  weight_dtype: DTypeLike = jnp.float32
  # Sharding related.
  qkv_partition: PartitionAnnotation = None
  o_partition: PartitionAnnotation = None
  attn_activation_partition: PartitionAnnotation = None
  output_partition: PartitionAnnotation = None
  # Decoding related.
  update_kv_cache_in_place: bool = True
  # Experimental flags.
  use_flash_attention: bool = False
  window_size: int = 0
  use_window_chunk: bool = False
  n_kv_heads: int = 0

  def setup(self) -> None:
    assert self.model_dim % self.n_heads == 0

    if self.use_per_dim_scale:
      self.per_dim_scale = PerDimScale(
          self.per_head_dim,
          weight_dtype=self.weight_dtype,
          activation_dtype=self.activation_dtype)

    if self.n_kv_heads <= 0:
      self.n_kv_heads = self.n_heads
    if self.n_heads % self.n_kv_heads != 0:
      raise ValueError(
          f'n_heads ({self.n_heads}) must be a multiple of n_kv_heads'
          f'({self.n_kv_heads}).'
      )

  def init(self, prng_key: PRNGKey) -> PyTree:
    qkey, kkey, vkey, okey = jax.random.split(prng_key, num=4)
    params = {}
    q_shape = [self.model_dim, self.n_heads, self.per_head_dim]
    kv_shape = [self.model_dim, self.n_kv_heads, self.per_head_dim]
    if self.use_combined_qkv:
      if self.n_heads == self.n_kv_heads:
        params['qkv_proj'] = xavier_init(
            qkey, shape=[3, *q_shape], dtype=self.weight_dtype,
            in_dim=self.model_dim, out_dim=self.n_heads * self.per_head_dim)
        params['qkv_proj'] = jax.lax.with_sharding_constraint(
            params['qkv_proj'], mesh_sharding((None,) + self.qkv_partition))
      else:
        params['q_proj'] = xavier_init(
            qkey, shape=q_shape, dtype=self.weight_dtype,
            in_dim=self.model_dim, out_dim=self.n_heads * self.per_head_dim)
        params['q_proj'] = jax.lax.with_sharding_constraint(
            params['q_proj'], mesh_sharding(self.qkv_partition))
        params['kv_proj'] = xavier_init(
            kkey, shape=[2, *kv_shape], dtype=self.weight_dtype,
            in_dim=self.model_dim, out_dim=self.n_kv_heads * self.per_head_dim)
        params['kv_proj'] = jax.lax.with_sharding_constraint(
            params['kv_proj'], mesh_sharding((None,) + self.qkv_partition))
    else:
      params['q_proj'] = xavier_init(
          qkey, shape=q_shape, dtype=self.weight_dtype,
          in_dim=self.model_dim, out_dim=self.n_heads * self.per_head_dim)
      params['k_proj'] = xavier_init(
          kkey, shape=kv_shape, dtype=self.weight_dtype,
          in_dim=self.model_dim, out_dim=self.n_kv_heads * self.per_head_dim)
      params['v_proj'] = xavier_init(
          vkey, shape=kv_shape, dtype=self.weight_dtype,
          in_dim=self.model_dim, out_dim=self.n_kv_heads * self.per_head_dim)

      for k in ['q_proj', 'k_proj', 'v_proj']:
        params[k] = jax.lax.with_sharding_constraint(
            params[k], mesh_sharding(self.qkv_partition))

    params['o_proj'] = xavier_init(
        okey, shape=q_shape, dtype=self.weight_dtype,
        in_dim=self.n_heads * self.per_head_dim, out_dim=self.model_dim)

    if self.use_per_dim_scale:
      params['per_dim_scale'] = self.per_dim_scale.init()

    params['o_proj'] = jax.lax.with_sharding_constraint(
        params['o_proj'], mesh_sharding(self.o_partition))

    return params

  def apply(
      self,
      params: PyTree, x: jax.Array,
      segment_ids: jax.Array | None = None,
      segment_positions: jax.Array | None = None,
      decode_state: PyTree | None = None,
  ) -> tuple[jax.Array, PyTree]:
    # x: [batch_size, seq_len, model_dim]
    assert len(x.shape) == 3
    assert x.shape[-1] == self.model_dim
    seq_len = x.shape[1]
    extra_output = {}
    if self.use_combined_qkv:
      if self.n_heads == self.n_kv_heads:
        qkv = jnp.einsum(
            'cijk,bsi->cbsjk',
            maybe_dequantize_array(
                params['qkv_proj'], dtype=self.activation_dtype),
            x).astype(self.activation_dtype)
        qkv = jax.lax.with_sharding_constraint(
            qkv, mesh_sharding((None,) + self.attn_activation_partition))
        q, k, v = qkv
      else:
        q = jnp.einsum(
            'ijk,...i->...jk',
            maybe_dequantize_array(
                params['q_proj'], dtype=self.activation_dtype),
            x).astype(self.activation_dtype)
        kv = jnp.einsum(
            'cijk,...i->c...jk',
            maybe_dequantize_array(
                params['kv_proj'], dtype=self.activation_dtype),
            x).astype(self.activation_dtype)
        kv = jax.lax.with_sharding_constraint(
            kv, mesh_sharding((None,) + self.attn_activation_partition))
        k, v = kv
    else:
      keys = ['q_proj', 'k_proj', 'v_proj']
      q, k, v = [
          jnp.einsum(
              'ijk,...i->...jk',
              maybe_dequantize_array(
                  params[key], dtype=self.activation_dtype),
              x).astype(self.activation_dtype)
          for key in keys]

    q = jax.lax.with_sharding_constraint(
        q, mesh_sharding(self.attn_activation_partition))
    k = jax.lax.with_sharding_constraint(
        k, mesh_sharding(self.attn_activation_partition))
    v = jax.lax.with_sharding_constraint(
        v, mesh_sharding(self.attn_activation_partition))

    q = rotary_positional_embedding(
        q, segment_positions=segment_positions)
    k = rotary_positional_embedding(
        k, segment_positions=segment_positions)

    if self.use_per_dim_scale:
      q = self.per_dim_scale.apply(params['per_dim_scale'], q)
    else:
      q = q / jnp.sqrt(self.per_head_dim)

    # n_groups = n_heads // n_kv_heads
    # q in [batch_size, seq_len, n_groups, n_kv_heads, per_head_dim]
    # k in [batch_size, seq_len, n_kv_heads, per_head_dim]
    # v in [batch_size, seq_len, n_kv_heads, per_head_dim]
    q = einops.rearrange(
        q,
        '... (g n_kv_heads) h -> ... g n_kv_heads h',
        n_kv_heads=self.n_kv_heads,
    )
    group_sharding = (
        *self.attn_activation_partition[:2],
        None,
        *self.attn_activation_partition[2:],
    )

    mask = create_mask(
        seq_len=seq_len,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        use_causal=self.use_causal,
        window_size=self.window_size,
    )
    # Add the group and head dimension.
    mask = einops.rearrange(mask, 'b l1 l2 -> b 1 1 l1 l2')

    cache_mask = None
    if decode_state is not None:
      assert self.use_causal
      if ('k' in decode_state and 'v' in decode_state and
          'cache_mask' in decode_state):
        k_cache = decode_state['k']
        v_cache = decode_state['v']
        cache_mask = decode_state['cache_mask']
        if self.update_kv_cache_in_place and segment_positions is not None:
          # Assume that we are dealing with one decode step.
          assert segment_positions.shape[1] == 1
          # Assume that all the tokens in the batch share the same position.
          cache_position = segment_positions[0][0]
          # Insert the new key and value at the cache_position.
          k = jax.lax.dynamic_update_slice_in_dim(
              k_cache, k, cache_position, axis=1)
          v = jax.lax.dynamic_update_slice_in_dim(
              v_cache, v, cache_position, axis=1)
          # Remove the mask (changing from 1 to 0) at cache_position.
          cache_mask = jax.lax.dynamic_update_slice_in_dim(
              cache_mask, jnp.zeros((k.shape[0], 1)),
              cache_position, axis=1)
          mask = einops.rearrange(cache_mask, 'b l -> b 1 1 1 l')
        else:
          k = jnp.concatenate([k_cache, k], axis=1)
          v = jnp.concatenate([v_cache, v], axis=1)
          mask = jnp.zeros_like(mask)
      else:
        if segment_ids is None:
          cache_mask = jnp.zeros(shape=(k.shape[0], k.shape[1]))
        else:
          cache_mask = jnp.asarray(segment_ids == 0, dtype=jnp.float32)

    # q: [batch_size, seq_len, n_groups, self.n_kv_heads, self.per_head_dim]
    # k, v: [batch_size, seq_len, self.n_kv_heads, self.per_head_dim]
    if (
        self.use_window_chunk
        and self.window_size > 0
        and self.window_size + 1 < seq_len
        and decode_state is None
    ):
      # We don't do this trick at decoding time, as we have better way there.
      output = chunked_local_attn(
          q, k, v, mask, self.window_size, dtype=self.activation_dtype
      )
    else:
      output, attn_mat = attn(q, k, v, mask, dtype=self.activation_dtype)
      if self.add_extra_output:
        extra_output['attn_mat'] = attn_mat

    output = jax.lax.with_sharding_constraint(
        output, mesh_sharding(group_sharding))

    output = jax.lax.with_sharding_constraint(
        output, mesh_sharding(group_sharding))
    output = einops.rearrange(
        output, '... n_groups n_kv_heads i -> ... (n_groups n_kv_heads) i'
    )
    output = jax.lax.with_sharding_constraint(
        output, mesh_sharding(self.attn_activation_partition))
    output = jnp.einsum(
        'jhi,bthi->btj',
        maybe_dequantize_array(
            params['o_proj'], dtype=self.activation_dtype), output)
    output = jax.lax.with_sharding_constraint(
        output, mesh_sharding(self.output_partition))

    if decode_state is not None:
      extra_output['decode_state'] = {'k': k, 'v': v}
      if self.update_kv_cache_in_place:
        extra_output['decode_state']['cache_mask'] = cache_mask
    return output, extra_output


@dataclasses.dataclass
class TransformerBlock(SimplyModule):
  """A single transformer block."""
  model_dim: int
  n_heads: int
  per_head_dim: int
  expand_factor: int
  use_rmsnorm: bool = False
  use_pre_ln: bool = True
  use_post_ln: bool = False
  use_post_skip_ln: bool = False
  use_gated_activation_in_ffn: bool = False
  use_per_dim_scale: bool = False
  # Mixed precision related.
  activation_dtype: DTypeLike = jnp.bfloat16
  # Sharding related.
  attn_qkv_partition: PartitionAnnotation = None
  attn_o_partition: PartitionAnnotation = None
  attn_activation_partition: PartitionAnnotation = None
  ffn0_partition: PartitionAnnotation = None
  ffn0_activation_partition: PartitionAnnotation = None
  ffn1_partition: PartitionAnnotation = None
  activation_partition: PartitionAnnotation = None
  # Below are for experimental usage.
  use_flash_attention: bool = False
  window_size: int = 0
  use_window_chunk: bool = False
  n_kv_heads: int = 0

  def setup(self) -> None:
    self.expand_dim = self.expand_factor * self.model_dim
    if self.use_pre_ln:
      self.pre_ln_0 = LayerNorm(
          dim=self.model_dim, use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype)
      self.pre_ln_1 = LayerNorm(
          dim=self.model_dim, use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype)
    if self.use_post_ln:
      self.post_ln_0 = LayerNorm(
          dim=self.model_dim, use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype)
      self.post_ln_1 = LayerNorm(
          dim=self.model_dim, use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype)
    if self.use_post_skip_ln:
      self.post_skip_ln_0 = LayerNorm(
          dim=self.model_dim, use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype)
      self.post_skip_ln_1 = LayerNorm(
          dim=self.model_dim, use_bias=not self.use_rmsnorm,
          activation_dtype=self.activation_dtype)

    self.attn = Attention(
        self.model_dim, self.n_heads, self.per_head_dim,
        use_per_dim_scale=self.use_per_dim_scale,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        qkv_partition=self.attn_qkv_partition,
        o_partition=self.attn_o_partition,
        attn_activation_partition=self.attn_activation_partition,
        output_partition=self.activation_partition,
        # Others.
        use_flash_attention=self.use_flash_attention,
        window_size=self.window_size,
        use_window_chunk=self.use_window_chunk,
        n_kv_heads=self.n_kv_heads)
    self.ffn_0 = Linear(
        self.model_dim, self.expand_dim, use_bias=True,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        weight_partition=self.ffn0_partition,
        output_partition=self.ffn0_activation_partition)
    self.ffn_1 = Linear(
        self.expand_dim, self.model_dim, use_bias=True,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        weight_partition=self.ffn1_partition,
        output_partition=self.activation_partition)
    if self.use_gated_activation_in_ffn:
      self.ffn_0_gate = Linear(
          self.model_dim, self.expand_dim, use_bias=True,
          # Mixed precision related.
          activation_dtype=self.activation_dtype,
          # Sharding related.
          weight_partition=self.ffn0_partition,
          output_partition=self.ffn0_activation_partition)

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    ffn0_key, ffn0gate_key, ffn1_key, attn_key = jax.random.split(
        prng_key, num=4)
    params['ffn_0'] = self.ffn_0.init(ffn0_key)
    if self.use_gated_activation_in_ffn:
      params['ffn_0_gate'] = self.ffn_0_gate.init(ffn0gate_key)
    params['ffn_1'] = self.ffn_1.init(ffn1_key)
    params['attn'] = self.attn.init(attn_key)
    if self.use_pre_ln:
      params['pre_ln_0'] = self.pre_ln_0.init()
      params['pre_ln_1'] = self.pre_ln_1.init()
    if self.use_post_ln:
      params['post_ln_0'] = self.post_ln_0.init()
      params['post_ln_1'] = self.post_ln_1.init()
    if self.use_post_skip_ln:
      params['post_skip_ln_0'] = self.post_skip_ln_0.init()
      params['post_skip_ln_1'] = self.post_skip_ln_1.init()

    return params

  def apply(
      self,
      params: PyTree, x: jax.Array,
      segment_ids: jax.Array | None = None,
      segment_positions: jax.Array | None = None,
      decode_state: PyTree | None = None,
  ) -> tuple[jax.Array, PyTree]:
    extra_output = {}
    x_res = x
    if self.use_pre_ln:
      x = self.pre_ln_0.apply(params['pre_ln_0'], x)
    x, attn_extra_output = self.attn.apply(
        params['attn'], x,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        decode_state=decode_state)
    if self.use_post_ln:
      x = self.post_ln_0.apply(params['post_ln_0'], x)
    x += x_res
    if self.use_post_skip_ln:
      x = self.post_skip_ln_0.apply(params['post_skip_ln_0'], x)
    x = jax.lax.with_sharding_constraint(
        x, mesh_sharding(self.activation_partition))

    x_res = x
    if self.use_pre_ln:
      x = self.pre_ln_1.apply(params['pre_ln_1'], x)
    projected_x = self.ffn_0.apply(params['ffn_0'], x)
    if self.use_gated_activation_in_ffn:
      gate = self.ffn_0_gate.apply(params['ffn_0_gate'], x)
      x = jnp.asarray(gelu(gate), self.activation_dtype) * projected_x
    else:
      x = jnp.asarray(gelu(projected_x), self.activation_dtype)
    x = self.ffn_1.apply(params['ffn_1'], x)
    if self.use_post_ln:
      x = self.post_ln_1.apply(params['post_ln_1'], x)
    x += x_res
    if self.use_post_skip_ln:
      x = self.post_skip_ln_1.apply(params['post_skip_ln_1'], x)
    x = jax.lax.with_sharding_constraint(
        x, mesh_sharding(self.activation_partition))

    if decode_state is not None:
      extra_output['decode_state'] = attn_extra_output['decode_state']
    return x, extra_output


@ModelRegistry.register
@dataclasses.dataclass
class TransformerLM(SimplyModule):
  """A decoder-only Transformer."""
  config: SimplyConfig
  sharding_config: SimplyConfig | None = None

  def setup(self) -> None:
    config = self.config
    sharding_config = self.sharding_config
    if sharding_config is None:
      sharding_config = config_lib.GSPMDSharding()

    self.activation_dtype = get_dtype(config.activation_dtype_name)
    self.embed = Embedding(
        vocab_size=config.vocab_size, dim=config.model_dim,
        partition=sharding_config.embed_partition,
        activation_dtype=self.activation_dtype)
    self.blocks = []
    for _ in range(config.n_layers):
      self.blocks.append(TransformerBlock(
          config.model_dim, config.n_heads, config.per_head_dim,
          config.expand_factor,
          use_rmsnorm=config.use_rmsnorm,
          use_pre_ln=config.use_pre_ln,
          use_post_ln=config.use_post_ln,
          use_post_skip_ln=config.use_post_skip_ln,
          use_per_dim_scale=config.use_per_dim_scale,
          use_gated_activation_in_ffn=config.use_gated_activation_in_ffn,
          # Mixed precision related.
          activation_dtype=self.activation_dtype,
          # Sharding related.
          attn_qkv_partition=sharding_config.attn_qkv_partition,
          attn_o_partition=sharding_config.attn_o_partition,
          attn_activation_partition=sharding_config.attn_activation_partition,
          ffn0_partition=sharding_config.ffn0_partition,
          ffn0_activation_partition=sharding_config.ffn0_activation_partition,
          ffn1_partition=sharding_config.ffn1_partition,
          activation_partition=sharding_config.activation_partition,
          # Others.
          use_flash_attention=config.use_flash_attention,
          window_size=config.window_size,
          use_window_chunk=config.use_window_chunk,
          n_kv_heads=config.n_kv_heads,
          ))
    self.final_ln = LayerNorm(
        dim=config.model_dim, use_bias=not config.use_rmsnorm,
        activation_dtype=self.activation_dtype)
    self.output_layer = Linear(
        config.model_dim, config.vocab_size, use_bias=True,
        use_external_weights=True,
        # Mixed precision related.
        activation_dtype=self.activation_dtype,
        # Sharding related.
        weight_partition=sharding_config.embed_partition[::-1],
        output_partition=sharding_config.logits_partition)

  def init(self, prng_key: PRNGKey) -> PyTree:
    params = {}
    prng_key, embed_key, output_layer_key = jax.random.split(prng_key, num=3)
    params['embed'] = self.embed.init(embed_key)
    for i in range(self.config.n_layers):
      prng_key, block_key = jax.random.split(prng_key, num=2)
      params[f'block_{i}'] = self.blocks[i].init(block_key)
    params['final_ln'] = self.final_ln.init()
    params['output_layer'] = self.output_layer.init(output_layer_key)
    return params

  def apply(
      self,
      params: PyTree, x: jax.Array,
      segment_ids: jax.Array | None = None,
      segment_positions: jax.Array | None = None,
      decode_state: PyTree | None = None,
  ) -> tuple[jax.Array, PyTree]:
    def convert_to_lower_bits(x, activation_dtype):
      # Only convert if the activation_dtype is lower bits than the params.
      if x.dtype.itemsize > activation_dtype.dtype.itemsize:
        return jnp.asarray(x, dtype=activation_dtype)
      else:
        return x
    params = jax.tree_util.tree_map(
        functools.partial(
            convert_to_lower_bits, activation_dtype=self.activation_dtype),
        params)
    x = self.embed.apply(params['embed'], x)

    extra_output = {}
    extra_output['decode_state'] = {}

    if self.config.use_scan:
      # Prepare parameters for scan.
      ps = jax.tree_util.tree_map(
          lambda *x: jnp.stack(x),
          *[params[f'block_{i}'] for i in range(self.config.n_layers)])
      if decode_state is None:
        blocks_decode_states = None
      else:
        blocks_decode_states = decode_state.get('blocks', {})
      # Assumes each layer is using the same hyperparameters.
      def process_blocks(inputs, p):
        block_params, decode_state = p
        result = self.blocks[0].apply(
            block_params, inputs, segment_ids=segment_ids,
            segment_positions=segment_positions,
            decode_state=decode_state)
        return result
      x, block_extra_output = jax.lax.scan(
          jax.remat(process_blocks),
          init=x, xs=(ps, blocks_decode_states))
      if decode_state is not None:
        extra_output['decode_state']['blocks'] = (
            block_extra_output['decode_state'])
    else:
      for i in range(self.config.n_layers):
        block_name = f'block_{i}'
        if decode_state is None:
          block_decode_state = None
        else:
          block_decode_state = decode_state.get(block_name, {})
        x, block_extra_output = self.blocks[i].apply(
            params[block_name], x,
            segment_ids=segment_ids,
            segment_positions=segment_positions,
            decode_state=block_decode_state)
        if decode_state is not None:
          extra_output['decode_state'][block_name] = (
              block_extra_output['decode_state'])

    x = self.final_ln.apply(params['final_ln'], x)
    output_layer_params = {
        self.output_layer.weight_name: params['embed'].T,
        self.output_layer.bias_name:
            params['output_layer'][self.output_layer.bias_name]}
    logits = self.output_layer.apply(output_layer_params, x)
    logits = soft_cap(logits, 30.0)
    return logits, extra_output

  def predict_probs(
      self,
      params: PyTree,
      x: jax.Array,
      temperature: float = 1.0
  ) -> jax.Array:
    logits, _ = self.apply(params, x)
    logits = logits.astype(jnp.float32)
    logits /= temperature
    return jax.nn.softmax(logits, axis=-1)


################################################################################
# Training.

################################################################################
## Optimizers.



def get_init_steps():
  return jax.lax.with_sharding_constraint(
      jnp.array(0, dtype=jnp.int32), mesh_sharding(None))


class Optimizer(abc.ABC):
  """An untra-simplified version of `flax.nn.Module`."""

  def init(self, params):
    """Initializes the state associated with the optimizer."""

  @abc.abstractmethod
  def apply(self, state, grad):
    """Applies the update rule to the optimizer state and the gradient."""


@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class SGD(Optimizer):
  """Stochastic Gradient Descent Optimizer."""

  def init(self, params):
    state = {}
    state['params'] = params
    state['steps'] = get_init_steps()
    return state

  def apply(self, state, grad):
    return grad, state


@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class Adam(Optimizer):
  """Adam Optimizer."""

  beta1: float = 0.9
  beta2: float = 0.95
  epsilon: float = 1e-8

  def init(self, params):
    state = {}
    state['params'] = params
    state['m'] = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(
            jnp.zeros_like(x), x.sharding),
        params)
    state['v'] = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(
            jnp.zeros_like(x), x.sharding),
        params)
    state['steps'] = get_init_steps()
    return state

  def apply(self, state, grad):
    state['m'] = jax.tree_util.tree_map(
        lambda m, g: m * self.beta1 + g * (1 - self.beta1), state['m'], grad)
    state['v'] = jax.tree_util.tree_map(
        lambda v, g: v * self.beta2 + jnp.square(g) * (1 - self.beta2),
        state['v'], grad)
    update = jax.tree_util.tree_map(
        lambda x, y: (x / (1 - self.beta1 ** (state['steps'] + 1))) /
        (jnp.sqrt(y / (1 - self.beta2 ** (state['steps'] + 1))) + self.epsilon),
        state['m'], state['v'])
    return update, state


@OptimizerRegistry.register
@dataclasses.dataclass(frozen=True)
class Lion(Optimizer):
  """Lion Optimizer."""

  beta1: float = 0.95
  beta2: float = 0.98
  momentum_dtype: jax.typing.DTypeLike = 'bfloat16'

  def init(self, params):
    state = {}
    state['params'] = params
    state['m'] = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(
            jnp.zeros_like(x, dtype=self.momentum_dtype), x.sharding),
        params)
    state['steps'] = get_init_steps()
    return state

  def apply(self, state, grad):
    grad = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=self.momentum_dtype), grad)
    update = jax.tree_util.tree_map(
        lambda m, g: jnp.sign(m * self.beta1 + g * (1 - self.beta1)),
        state['m'], grad)
    state['m'] = jax.tree_util.tree_map(
        lambda v, g: v * self.beta2 + g * (1 - self.beta2),
        state['m'], grad)
    return update, state


################################################################################
## Learning rate schedules.


def create_lr_schedule(config):
  """Creates a learning rate schedule from given config."""
  lr_schedule_config = dict(config.lr_schedule_config)
  if ('decay_start' in lr_schedule_config and
      isinstance(lr_schedule_config['decay_start'], float) and
      lr_schedule_config['decay_start'] > 0 and
      lr_schedule_config['decay_start'] < 1.0):
    lr_schedule_config['decay_start'] = int(
        config.num_train_steps * lr_schedule_config['decay_start'])

  if (('decay_steps' in lr_schedule_config) and
      ('steps_after_decay' in lr_schedule_config)):
    raise ValueError('Cannot specify both decay_steps and steps_after_decay.')
  elif 'steps_after_decay' in lr_schedule_config:
    lr_schedule_config['decay_steps'] = (
        config.num_train_steps - lr_schedule_config['steps_after_decay'])
    if 'decay_start' in lr_schedule_config:
      lr_schedule_config['decay_steps'] -= lr_schedule_config['decay_start']
    elif 'warmup_steps' in lr_schedule_config:
      lr_schedule_config['decay_steps'] -= lr_schedule_config['warmup_steps']
    del lr_schedule_config['steps_after_decay']
  if config.lr_schedule_name == 'cosine_decay':
    return functools.partial(cosine_decay_lr, **lr_schedule_config)
  elif config.lr_schedule_name == 'constant':
    return functools.partial(constant_lr, **lr_schedule_config)
  else:
    raise ValueError(
        f'Unknown lr_schedule: {config.lr_schedule_name}')


def cosine_decay_lr(steps, lr, decay_steps, warmup_steps=1, end_decay=0.1,
                    decay_start=None):
  """Cosine decay learning rate schedule."""
  # Linear warmup.
  steps += 1
  warmup_factor = jnp.minimum(steps, warmup_steps) / warmup_steps
  if decay_start is None:
    decay_start = warmup_steps
  decay_progress = jnp.maximum(0.0, steps - decay_start) / decay_steps
  decay_factor = (
      1 + jnp.cos(jnp.minimum(decay_progress, 1.0) * jnp.pi)) / 2
  actual_lr = lr * warmup_factor * ((1 - end_decay) * decay_factor + end_decay)
  actual_lr = jax.lax.with_sharding_constraint(actual_lr, mesh_sharding(None))
  return actual_lr


def constant_lr(steps, lr, warmup_steps):
  steps += 1
  warmup_factor = jnp.minimum(steps, warmup_steps) / warmup_steps
  actual_lr = lr * warmup_factor
  actual_lr = jax.lax.with_sharding_constraint(actual_lr, mesh_sharding(None))
  return actual_lr


################################################################################
## Loss and backprop.


def compute_loss(model, params, batch):
  """Computes the cross entropy loss."""
  inputs = batch['decoder_input_tokens']
  targets = batch['decoder_target_tokens']
  loss_weights = batch['decoder_loss_weights']
  segment_ids = batch.get('decoder_segment_ids', None)
  segment_positions = batch.get('decoder_positions', None)
  logits, _ = model.apply(
      params, inputs,
      segment_ids=segment_ids, segment_positions=segment_positions)
  # Always use float32 in softmax.
  logits = logits.astype(jnp.float32)
  targets = jax.nn.one_hot(targets, model.config.vocab_size, axis=-1)
  token_loss = jnp.einsum(
      'blv,blv->bl', targets, jax.nn.log_softmax(logits))
  total_loss = - jnp.sum(token_loss * loss_weights)
  loss = total_loss / jnp.sum(loss_weights)
  loss = jax.lax.with_sharding_constraint(loss, mesh_sharding(None))
  return loss


def compute_distill_loss(model, params, teacher_model, teacher_params, batch):
  """Computes the distillation loss."""
  inputs = batch['decoder_input_tokens']
  loss_weights = batch['decoder_loss_weights']
  segment_ids = batch.get('decoder_segment_ids', None)
  segment_positions = batch.get('decoder_positions', None)
  logits, _ = model.apply(
      params, inputs,
      segment_ids=segment_ids, segment_positions=segment_positions)
  teacher_logits, _ = teacher_model.apply(
      teacher_params, inputs,
      segment_ids=segment_ids, segment_positions=segment_positions)
  # Always use float32 in softmax.
  logits = logits.astype(jnp.float32)
  teacher_logits = teacher_logits.astype(jnp.float32)
  teacher_logits = jax.lax.stop_gradient(teacher_logits)
  token_loss = jnp.einsum(
      'blv,blv->bl',
      jax.nn.softmax(teacher_logits),
      jax.nn.log_softmax(teacher_logits) - jax.nn.log_softmax(logits))
  total_loss = jnp.sum(token_loss * loss_weights)
  loss = total_loss / jnp.sum(loss_weights)
  loss = jax.lax.with_sharding_constraint(loss, mesh_sharding(None))
  return loss


def train_one_step(state, batch, model, opt,
                   teacher_model=None,
                   lr=1e-4,
                   clip_grad_norm=-1, clip_update_norm=-1,
                   clip_update_rms=-1,
                   clip_local_update_rms=-1,
                   weight_decay=-1,
                   custom_loss_fn=None,
                   add_log_info=False):
  """Trains one step."""
  clip_norm_fn = functools.partial(
      clip_tree_fn, fn=tree_norm, fn_name='norm')
  clip_rms_fn = functools.partial(
      clip_tree_fn, fn=tree_rms, fn_name='rms')

  norm_info_fn = functools.partial(
      compute_tree_info_fn, fn=tree_norm, fn_name='norm')
  rms_info_fn = functools.partial(
      compute_tree_info_fn, fn=tree_rms, fn_name='rms')

  log_dict = {}
  if add_log_info:
    log_dict.update(norm_info_fn(state['params'], name='weights'))
    log_dict.update(rms_info_fn(state['params'], name='weights'))

  if teacher_model is None:
    loss_fn = compute_loss if custom_loss_fn is None else custom_loss_fn
    loss, grad = jax.value_and_grad(
        loss_fn, argnums=1)(model, state['params'], batch)
  else:
    loss_fn = compute_distill_loss if custom_loss_fn is None else custom_loss_fn
    loss, grad = jax.value_and_grad(
        loss_fn, argnums=1)(
            model, state['params'],
            teacher_model, state['teacher_params'], batch)

  if clip_grad_norm > 0:
    grad, clip_log_dict = clip_norm_fn(
        grad, name='grad', threshold=clip_grad_norm,
        add_log_info=add_log_info)
    log_dict.update(clip_log_dict)
  else:
    log_dict.update(norm_info_fn(grad, name='grad'))

  update, new_state = opt.apply(state, grad)
  if clip_update_norm > 0:
    update, clip_log_dict = clip_norm_fn(
        update, name='update', threshold=clip_update_norm,
        add_log_info=add_log_info)
    log_dict.update(clip_log_dict)
  else:
    log_dict.update(norm_info_fn(update, name='update'))

  if clip_update_rms > 0 or clip_local_update_rms > 0:
    update, clip_log_dict = clip_rms_fn(
        update, name='update',
        clip_local=clip_local_update_rms > 0,
        threshold=(clip_local_update_rms
                   if clip_local_update_rms > 0 else clip_update_rms),
        add_log_info=add_log_info)
    log_dict.update(clip_log_dict)
  else:
    log_dict.update(rms_info_fn(update, name='update'))

  if weight_decay > 0:
    update = jax.tree_util.tree_map(
        lambda x, y: x + y * weight_decay, update, new_state['params'])
  new_state['params'] = jax.tree_util.tree_map(
      lambda x, y: x - y * lr, new_state['params'], update)
  new_state['steps'] += 1
  return loss, new_state, log_dict


def tree_norm(tree):
  flat, _ = jax.tree_util.tree_flatten(tree)
  norm = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in flat]))
  return norm


def tree_rms(tree):
  flat, _ = jax.tree_util.tree_flatten(tree)
  # Cast to float32 to avoid overflow.
  total_size = sum([jnp.asarray(jnp.size(x), jnp.float32) for x in flat])
  rms = jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in flat]) / total_size)
  return rms


def clip_tree_fn(
    tree, name, threshold, fn, fn_name,
    clip_local=False, add_log_info=False):
  """Clips the pytree with the given threshold."""
  val = local_val = clipped_tree = None
  if add_log_info or not clip_local:
    val = fn(tree)
    factor = jnp.minimum(threshold, val) / val
    clipped_tree = jax.tree_util.tree_map(lambda x: x * factor, tree)

  if add_log_info or clip_local:
    local_val = jax.tree_util.tree_map(fn, tree)
    clipped_tree = jax.tree_util.tree_map(
        lambda x, y: x * jnp.minimum(threshold, y) / y, tree, local_val)

  log_dict = {}
  if add_log_info:
    log_dict[f'global_{name}_{fn_name}'] = val
    log_dict[f'local_{name}_{fn_name}'] = local_val
    log_dict[f'global_clipped_{name}_{fn_name}'] = fn(clipped_tree)
    log_dict[f'local_clipped_{name}_{fn_name}'] = jax.tree_util.tree_map(
        fn, clipped_tree)
  return clipped_tree, log_dict


def compute_tree_info_fn(tree, name, fn, fn_name):
  log_dict = {}
  log_dict[f'global_{name}_{fn_name}'] = fn(tree)
  log_dict[f'local_{name}_{fn_name}'] = jax.tree_util.tree_map(fn, tree)
  return log_dict


################################################################################
# Evaluation.


def evaluate(loss_fn, params, dataset, print_debug_info=False):
  """Evaluates the loss on the given dataset."""
  loss_sum = 0.0
  for batch in dataset.as_numpy_iterator():
    batch_loss = loss_fn(params=params, batch=batch)
    if print_debug_info:
      print(f'batch_loss.sharding: {batch_loss.sharding}')
      print(f'batch_loss: {batch_loss.addressable_data(0)}')
      print(f'batch_loss.is_fully_addressable '
            f'{batch_loss.is_fully_addressable}')
      print(f'batch_loss.is_fully_replicated {batch_loss.is_fully_replicated}')
      print(f'batch_loss.sharding.device_set {batch_loss.sharding.device_set}')
    loss_sum += batch_loss.addressable_data(0)
  return loss_sum


################################################################################
# Experiment.


def run_experiment(
    config, sharding_config, mesh_shape, create_dataset, experiment_dir,
    dcn_mesh_shape=None, model=None):
  """Main experiment loop."""
  global MESH_SHAPE
  global DCN_MESH_SHAPE
  MESH_SHAPE = mesh_shape
  DCN_MESH_SHAPE = dcn_mesh_shape
  prng_key = jax.random.PRNGKey(config.model_seed)
  is_primary = jax.process_index() == 0

  # Create experiment folder with tensorboard log and checkpoint.
  if not os.path.exists(experiment_dir) and is_primary:
    os.mkdir(experiment_dir)

  config_path = os.path.join(experiment_dir, 'experiment_config.json')
  if is_primary:
    with open(config_path, 'w') as f:
      json.dump(
          dataclasses.asdict(config), f, default=str, indent=2
      )

  config_path = os.path.join(experiment_dir, 'sharding_config.json')
  if is_primary:
    with open(config_path, 'w') as f:
      json.dump(dataclasses.asdict(sharding_config), f, indent=2)

  logdir = os.path.join(experiment_dir, 'tb_log')
  if not os.path.exists(logdir) and is_primary:
    os.mkdir(logdir)

  writer = metric_writers.create_default_writer(
      logdir=logdir,
      just_logging=jax.process_index() != 0,
      asynchronous=True,
  )

  ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
  options = ocp.CheckpointManagerOptions(
      save_interval_steps=config.ckpt_interval,
      max_to_keep=config.ckpt_max_to_keep,
      async_options=ocp.AsyncOptions(
          timeout_secs=360000,
      ),
  )
  # Create model.
  if model is None:
    model_cls = ModelRegistry.get(config.model_name)
    model = model_cls(config, sharding_config=sharding_config)
  opt = OptimizerRegistry.get(config.optimizer_name)(
      **dict(config.optimizer_config))

  def get_abstract_params(model):
    return eval_shape_with_sharding(model.init, jax.random.PRNGKey(0))

  def get_abstract_state(model, opt):
    abstract_params = get_abstract_params(model)
    abstract_state = opt.init(abstract_params)
    return abstract_state

  def load_checkpoint(config, model, opt=None):
    init_mngr = ocp.CheckpointManager(
        config.init_ckpt_dir, checkpointers=ocp.PyTreeCheckpointer())
    t1 = time.time()
    if config.init_ckpt_step == -1:
      ckpt_step = init_mngr.latest_step()
    else:
      ckpt_step = config.init_ckpt_step
    if config.init_ckpt_opt_state:
      if opt is None:
        raise ValueError(
            'You must provide the optimizer if you want to load the '
            'optimizer state from the checkpoint.')
      abstract_state = get_abstract_state(model, opt)
    else:
      abstract_state = {'params': get_abstract_params(model)}
    state = init_mngr.restore(
        ckpt_step,
        args=ocp.args.PyTreeRestore(
            abstract_state,
            restore_args=ocp.checkpoint_utils.construct_restore_args(
                abstract_state),
            transforms={}))
    if config.reset_steps:
      state['steps'] = get_init_steps()
    dt = time.time() - t1
    logging.info(
        '%s secs used for loading checkpoint %s at step %s.',
        dt, config.init_ckpt_dir, ckpt_step)
    init_mngr.close()
    return state

  mngr = ocp.CheckpointManager(ckpt_dir, options=options)
  if mngr.latest_step() is not None:  # continue training from lastest ckpt.
    abstract_state = get_abstract_state(model, opt)
    t1 = time.time()
    logging.info('restoring the checkpoint at step %s', mngr.latest_step())
    state = mngr.restore(
        mngr.latest_step(), args=ocp.args.StandardRestore(abstract_state))
    ckpt_restore_time = time.time() - t1
    logging.info('%s secs used in restoring checkpoint.', ckpt_restore_time)
  elif config.init_ckpt_dir:  # initialize from a given external ckpt.
    state = load_checkpoint(config, model, opt)
  else:  # initialize from scratch.
    state = opt.init(jax.jit(model.init)(prng_key))

  # Add the teacher configuration if specified.
  teacher_model = None
  if hasattr(config, 'teacher'):
    teacher_model_cls = ModelRegistry.get(config.teacher.model_name)
    teacher_model = teacher_model_cls(
        config.teacher, sharding_config=sharding_config)
    teacher_params = load_checkpoint(config.teacher, teacher_model)['params']
    state['teacher_params'] = teacher_params
    del teacher_params

  params_shape = jax.tree_util.tree_map(lambda x: str(x.shape), state['params'])
  logging.info('params shape: %s', params_shape)
  params_sharding = jax.tree_util.tree_map(
      lambda x: str(x.sharding), state['params'])
  logging.info('params sharding: %s', params_sharding)
  num_params = sum(jax.tree.leaves(
      jax.tree_util.tree_map(lambda x: np.prod(x.shape), state['params'])))
  logging.info('num_params: %s M', num_params/1e6)

  param_info_map = jax.tree_util.tree_map(
      lambda x, y: f'{x} :: {y}', params_shape, params_sharding
  )
  param_info_text = yaml.dump(
      param_info_map, default_flow_style=False, sort_keys=False
  )
  experiment_config_text = json.dumps(
      dataclasses.asdict(config), default=str, indent=2
  )
  sharding_config_text = json.dumps(
      dataclasses.asdict(sharding_config), indent=2
  )
  model_text = json.dumps(
      model, default=lambda o: getattr(o, '__dict__', str(o)), indent=2
  )
  writer.write_texts(
      step=0,
      texts={
          'num_params': f'`{num_params}`',
          'param_info_text': f'```\n{param_info_text}\n```',
          'experiment_config_text': f'```\n{experiment_config_text}\n```',
          'sharding_config_text': f'```\n{sharding_config_text}\n```',
          'model': f'```\n{model_text}\n```'
      },
  )
  writer.flush()

  if is_primary:
    with open(os.path.join(experiment_dir, 'params_info.json'), 'w') as f:
      json.dump(
          {
              'params_shape': params_shape,
              'params_sharding': params_sharding,
              'num_params': str(num_params),
          },
          f,
          indent=2,
      )
    with open(os.path.join(experiment_dir, 'model_info.json'), 'w') as f:
      f.write(model_text)

  # Compile loss, train and learning rate functions.
  t1 = time.time()

  @functools.partial(jax.jit, static_argnames=['add_log_info'])
  def train_one_step_fn(state, batch, lr, add_log_info=False):
    return train_one_step(
        state=state,
        batch=batch,
        lr=lr,
        model=model,
        opt=opt,
        teacher_model=teacher_model,
        clip_grad_norm=config.clip_grad_norm,
        clip_update_norm=config.clip_update_norm,
        clip_local_update_rms=config.clip_local_update_rms,
        weight_decay=config.weight_decay,
        add_log_info=add_log_info,
    )

  if config.use_validation_set:
    loss_fn = jax.jit(functools.partial(compute_loss, model=model))
  else:
    loss_fn = None

  lr_fn = jax.jit(create_lr_schedule(config))
  dt = time.time() - t1
  logging.info('%s secs used for compiling train, loss and lr functions.', dt)

  metrics_aggregator = MetricsAggregator()

  # Start training.
  start_steps = int(state['steps'].addressable_data(0))
  steps = start_steps

  train_set, validation_set = create_dataset(config, start_steps)

  # Initialize train and validation datasets.
  logging.info('Initializing dataset.')
  logging.info('jax.process_index(): %s', jax.process_index())
  logging.info('sharding_config.data_partition: %s',
               sharding_config.data_partition)
  # Data is initially fully sharded across all devices.
  init_data_sharding = mesh_sharding(
      (('replica', 'data', 'model'), None))
  if sharding_config is not None:
    data_sharding = mesh_sharding(sharding_config.data_partition)
  else:
    data_sharding = mesh_sharding(config.data_partition)
  build_global_array_fn = functools.partial(
      build_global_array,
      global_shape=(config.batch_size, config.seq_len),
      init_sharding=init_data_sharding,
      final_sharding=data_sharding)

  train_set_iter = iter(train_set)
  prev_step_timestamp = time.time()
  while steps < config.num_train_steps:
    with jax.profiler.StepTraceAnnotation('train', step_num=steps):
      logging.info('steps: %s', steps)
      print(f'steps: {steps}')
      t1 = time.time()
      batch = next(train_set_iter)
      batch = jax.tree_util.tree_map(build_global_array_fn, batch)
      batch_generation_time = time.time() - t1
      assert state['steps'].is_fully_replicated
      lr = lr_fn(state['steps'])
      assert lr.is_fully_replicated

      if mngr.should_save(steps) or steps == config.num_train_steps - 1:
        t1 = time.time()
        mngr.save(steps, args=ocp.args.StandardSave(state))
        ckpt_save_time = time.time() - t1
        logging.info('%s secs used in saving checkpoint.', ckpt_save_time)
      else:
        ckpt_save_time = 0.0

      # Log every tb_log_interval steps or at the very end.
      should_tb_log = (steps % config.tb_log_interval == 0 or
                       steps == (config.num_train_steps - 1))
      add_log_info = config.log_additional_info and should_tb_log
      loss, state, log_dict = train_one_step_fn(
          state=state,
          batch=batch,
          lr=lr,
          add_log_info=add_log_info,
      )

      if add_log_info:
        batch_stats_info = compute_batch_stats_info(batch)
        logging.info('========== batch_stats_info ==========')
        for k, v in batch_stats_info.items():
          logging.info('%s: %s', k, v)
        log_dict.update(batch_stats_info)

      train_loss = loss.addressable_data(0)
      train_loss = np.array(train_loss)
      logging.info('train_loss: %s', train_loss)
      print(f'train_loss: {train_loss}')
      step_time = time.time() - prev_step_timestamp
      logging.info('%s secs per step.', step_time)
      print(f'{step_time} secs per step')
      prev_step_timestamp = time.time()
      metrics_aggregator.add('train_loss', train_loss)

      if should_tb_log:
        t1 = time.time()
        aggregated_metrics = metrics_aggregator.aggregate()
        metrics_aggregator.reset()
        metrics_dict = dict(
            loss=aggregated_metrics['train_loss'],
            # accuracy=accuracy,
            lr=lr,
            secs_per_step=step_time,
            steps_per_sec=1 / step_time,
            secs_per_batch_generation=batch_generation_time,
            secs_per_ckpt_save=ckpt_save_time)
        metrics_dict.update(flatten_dict(log_dict))
        writer.write_scalars(steps, metrics_dict)
        writer.flush()
        event_write_time = time.time() - t1
        logging.info('%s secs per writing tensorboard event.', event_write_time)

      if config.use_validation_set and validation_set is not None:
        # Run eval every validation_eval_interval steps or at the very end.
        should_validation_eval = (
            steps % config.validation_eval_interval == 0 or
            steps == (config.num_train_steps - 1))
        if should_validation_eval:
          mean_eval_loss = 0.0
          # The `loss_weights` is normally the same as `num_tokens`.
          total_weights = 0.0
          total_num_tokens = 0
          validation_eval_start_time = time.time()
          for eval_steps, eval_batch in enumerate(validation_set.repeat(1)):
            eval_batch = jax.tree_util.tree_map(
                build_global_array_fn, eval_batch)
            eval_batch_stats_info = compute_batch_stats_info(eval_batch)
            eval_loss = loss_fn(params=state['params'], batch=eval_batch)
            eval_loss = np.array(eval_loss.addressable_data(0))
            num_tokens = np.array(
                eval_batch_stats_info['num_tokens'].addressable_data(0))
            batch_weights = np.array(
                eval_batch_stats_info['total_weights'].addressable_data(0))
            if total_weights <= 1e-6:
              mean_eval_loss = eval_loss
            else:
              weights_ratio = batch_weights / total_weights
              # Iteratively update mean_eval_loss to avoid numerical overflow.
              mean_eval_loss = (
                  mean_eval_loss + (eval_loss - mean_eval_loss) * weights_ratio)
            total_weights += batch_weights
            total_num_tokens += num_tokens
            if (config.validation_num_eval_steps > 0 and
                (eval_steps >= config.validation_num_eval_steps)):
              break
          validation_eval_time = time.time() - validation_eval_start_time
          logging.info(
              '%s secs in validation eval, %s steps, %s secs per step.',
              validation_eval_time, config.validation_num_eval_steps,
              validation_eval_time / config.validation_num_eval_steps)
          writer.write_scalars(
              steps,
              dict(validation_loss=mean_eval_loss,
                   validation_weights=total_weights,
                   validation_tokens=total_num_tokens,
                   validation_eval_time=validation_eval_time))
          writer.flush()
          print(f'validation_loss: {mean_eval_loss}')
          print(f'validation_eval_time: {validation_eval_time}')
      steps += 1
  # Ensure all the checkpoints are saved.
  mngr.close()
  writer.close()


class MetricsAggregator(object):
  """Metrics aggregator."""

  _metrics: Mapping[str, list[np.ndarray]]

  def __init__(self):
    self._metrics = collections.defaultdict(list)

  def add(self, name: str, value: np.ndarray) -> None:
    self._metrics[name].append(value)

  def reset(self) -> None:
    self._metrics = collections.defaultdict(list)

  def aggregate(self) -> Mapping[str, np.ndarray]:
    return {k: np.mean(v) for k, v in self._metrics.items()}


def flatten_dict(d: dict[str, Any]):
  """Flattens a nested dictionary."""
  result_dict = {}
  for k, v in d.items():
    if isinstance(v, dict):
      vd = flatten_dict(v)
      for vk, vv in vd.items():
        new_key = k + '/' + vk
        if new_key in result_dict:
          raise ValueError(f'Duplicate key: {vk}')
        else:
          result_dict[new_key] = vv
    else:
      result_dict[k] = v
  return result_dict


@jax.jit
def compute_batch_stats_info(
    batch: Batch,
    pad_id: int = 0) -> Mapping[str, Any]:
  """Computes statistics of the given batch."""
  result = {}
  batch_size = batch['decoder_target_tokens'].shape[0]
  result['num_seq'] = batch_size
  seq_len = batch['decoder_target_tokens'].shape[1]
  result['seq_len'] = seq_len

  tokens_per_seq = np.sum(
      batch['decoder_target_tokens'] != pad_id, axis=-1).astype(np.int32)
  result['num_tokens'] = tokens_per_seq.sum()
  result['avg_num_tokens_per_seq'] = tokens_per_seq.mean()
  result['std_num_tokens_per_seq'] = tokens_per_seq.std()

  ratio_of_nonpad_tokens = tokens_per_seq / seq_len
  result['avg_ratio_nonpad_tokens_per_seq'] = ratio_of_nonpad_tokens.mean()
  result['std_ratio_nonpad_tokens_per_seq'] = ratio_of_nonpad_tokens.std()

  loss_weights_per_seq = np.sum(batch['decoder_loss_weights'], axis=-1)
  result['total_weights'] = loss_weights_per_seq.sum()
  result['avg_weights_per_seq'] = loss_weights_per_seq.mean()
  result['std_weights_per_seq'] = loss_weights_per_seq.std()

  if 'decoder_segment_ids' in batch:
    num_segments = np.max(batch['decoder_segment_ids'], axis=-1)
    result['num_segments'] = num_segments.sum()
    result['avg_num_segments_per_seq'] = num_segments.mean()
    result['std_num_segments_per_seq'] = num_segments.std()
    result['avg_segment_length'] = tokens_per_seq.sum() / num_segments.sum()
  return result


################################################################################
# Decoding


@dataclasses.dataclass
class SamplingParams():
  top_k: int = -1
  top_p: float = 1.0
  temperature: float = 1.0
  max_decode_steps: int = 256
  num_samples: int = 4
  intermediate_decode_steps: Optional[int] = None
  sort_by: Optional[str] = 'avg_output_score'


@dataclasses.dataclass
class SamplingOutput():
  """Output of a sampling."""
  input_text: str
  input_token_ids: list[int]
  output_text: str
  output_token_ids: list[int]
  # Sum log prob.
  sum_output_score: float
  # Average log prob.
  avg_output_score: float
  # log prob of each token.
  input_token_scores: list[float]
  # Sum log prob.
  sum_input_score: float
  # Average log prob.
  avg_input_score: float
  # log prob of each token.
  output_token_scores: list[float]
  # Sampling params that generated this output.
  params: SamplingParams


class SimplyVocab(Protocol):
  pad_id: Optional[int]
  bos_id: Optional[int]
  eos_id: Optional[int]

  def encode(self, text: str) -> list[int]: ...

  def decode(self, token_ids: list[int]) -> str: ...


class TestVocab(SimplyVocab):
  """A toy vocabulary for testing."""

  def __init__(self, vocab_list, bos_id=2, eos_id=-1, pad_id=0, unk_id=3):
    self.bos_id = bos_id
    self.eos_id = eos_id
    self.pad_id = pad_id
    self.unk_id = unk_id
    start_id = max(unk_id, pad_id, eos_id, bos_id) + 1
    self._vocab_dict = dict(
        [(w, (i + start_id)) for i, w in enumerate(vocab_list)])
    self._rev_vocab_dict = {v: k for k, v in self._vocab_dict.items()}

  def encode(self, text: str) -> list[int]:
    return [self._vocab_dict.get(w, self.unk_id) for w in text.split()]

  def decode(self, token_ids: list[int]) -> str:
    return ' '.join([self._rev_vocab_dict.get(i, '<unk>') for i in token_ids])


def get_prefill_size(n, min_prefill_size=256):
  return max(int(np.exp2(np.ceil(np.log2(n)))), min_prefill_size)


class LMInterface():
  """An interface to interact with a language model."""

  def __init__(self, model: SimplyModule, params: PyTree,
               vocab: SimplyVocab,
               default_sampling_params: Optional[SamplingParams] = None,
               bos_id: Optional[int] = None, pad_id: Optional[int] = None,
               min_prefill_size: int = 256, max_input_len: int = -1,
               extra_eos_ids: Optional[list[int]] = None,
               extra_eos_tokens: Optional[list[str]] = None,
               disable_jit=False):
    """An interface to interact with a language model.

    Args:
      model: The model to use, for example, a TransformerLM instance.
      params: The `params` to use in `model.apply`.
      vocab: The vocabulary instance to use.
      default_sampling_params: Default sampling params for `generate`.
      bos_id: The bos id to use, if not given then it will use the 
        `bos_id` field of the `vocab`.
      pad_id: The pad id to use, if not given then it will use the 
        `pad_id` field of the `vocab`.
      min_prefill_size: The minimum prefill size to use.
      max_input_len: The max input length to use, longer inputs will be 
        truncated to keep the last `max_input_len` tokens, if set to -1 
        then it will use the full input length.
      extra_eos_ids: Extra eos ids to include.
      extra_eos_tokens: Extra eos tokens to include.
      disable_jit: Whether to disable jit for debugging purposes.
    Returns:
      A LMInterface instance.
    """
    self.max_input_len = max_input_len
    # Minimum prefill size so that we only run jit compilation once for smaller
    # sequence lengths.
    self.min_prefill_size = min_prefill_size
    self.model = model
    self.vocab = vocab
    self.eos_ids = [vocab.eos_id]
    if extra_eos_ids is not None:
      self.eos_ids.extend(extra_eos_ids)
    if extra_eos_tokens is not None:
      for token in extra_eos_tokens:
        encoded_token_ids = vocab.encode(token)
        assert len(encoded_token_ids) == 1, (
            f'Invalid eos token {token} , '
            f'valid eos token must be a single token in vocab.')
        self.eos_ids.append(encoded_token_ids[0])
    self.eos_ids = list(set(self.eos_ids))
    # The token id to append at the beginning of the input.
    if pad_id is None:
      self.pad_id = vocab.pad_id
    else:
      self.pad_id = pad_id
    if bos_id is None:
      self.bos_id = vocab.bos_id
    else:
      self.bos_id = bos_id

    if default_sampling_params is None:
      default_sampling_params = SamplingParams()
    self.default_sampling_params = default_sampling_params
    self.decode_fn = functools.partial(
        sample_decode, apply_fn=model.apply,
        # Disable scan if jit is disabled.
        use_scan=False if disable_jit else self.model.config.use_scan,
        eos_ids=self.eos_ids,
        pad_id=vocab.pad_id)
    if not disable_jit:
      self.decode_fn = jax.jit(
          self.decode_fn,
          static_argnames=['intermediate_decode_steps', 'max_decode_steps'])
    self.model_params = params

  def generate(self, input_text: str, seed: Optional[int] = None,
               prefill_size: int = -1,
               sampling_params: Optional[SamplingParams] = None,
               include_eos_in_output_text: bool = False,
               include_bos_in_input_text: bool = False,
               print_debug_info: bool = False) -> list[SamplingOutput]:
    """Generate samples from a given input text.

    Args:
      input_text: Input text to generate samples for.
      seed: Seed for controlling the randomness.
      prefill_size: Prefill size to use for the generation, if set to -1 then
        it will be inferred from the input length and self.min_prefill_size.
      sampling_params: Sampling params to use for the generation.
      include_eos_in_output_text: Whether to include the eos token when
        generating the `output_text` field of the sampling outputs.
        Note that even if this is set to `True`, the `vocab.decode` can
        still skip the eos token.
      include_bos_in_input_text: Whether to include the bos token in the
        `input_text` field of the sampling outputs.
      print_debug_info: Whether to print debug info.

    Returns:
      A list of `SamplingOutput`, ranked by the `sort_by` field of the
      `sampling_params`. Note that the eos token and bos token are included in 
      the `output_token_ids` and `input_token_ids` field of the 
      `SamplingOutput`, but the `input_token_scores` will not include the bos 
      token so its length is one less than `input_token_ids`.
    """
    if sampling_params is None:
      sampling_params = self.default_sampling_params
    if seed is None:
      seed = int(time.time() * 1000)
    prng_key = jax.random.PRNGKey(seed)
    # Encode the prompt.
    inputs = np.array(self.vocab.encode(input_text)).reshape([1, -1])

    # If max_input_len is given then truncate the input.
    if self.max_input_len > 0:
      inputs = inputs[:, -self.max_input_len:]

    # Save the actual input length.
    input_len = inputs.shape[1]

    # If prefill is not given then infer it from the input length plus bos_id.
    if prefill_size <= 0:
      prefill_size = get_prefill_size(input_len + 1, self.min_prefill_size)
    assert input_len < prefill_size

    # Add bos id and pad inputs to prefill_size.
    prefill_pad_size = prefill_size - input_len - 1
    inputs = np.pad(
        inputs, ((0, 0), (1, 0),),
        constant_values=self.bos_id)
    inputs = np.pad(
        inputs, ((0, 0), (0, prefill_pad_size),),
        constant_values=self.pad_id)

    # Create segment positions and segment ids.
    segment_positions = np.arange(prefill_size).reshape([1, prefill_size])
    segment_ids = np.concatenate(
        [np.ones([1, input_len+1], dtype=jnp.int32),
         np.zeros([1, prefill_pad_size], dtype=jnp.int32)],
        axis=1)

    # Repeat inputs to generate multiple samples.
    if sampling_params.num_samples > 1:
      inputs = einops.repeat(
          inputs, '1 t -> b t', b=sampling_params.num_samples)
      segment_positions = einops.repeat(
          segment_positions, '1 t -> b t', b=sampling_params.num_samples)
      segment_ids = einops.repeat(
          segment_ids, '1 t -> b t', b=sampling_params.num_samples)

    # Prepare the max_decode_steps and intermediate_decode_steps to
    # pass to decode_fn.
    max_decode_steps = sampling_params.max_decode_steps
    intermediate_decode_steps = sampling_params.intermediate_decode_steps
    if max_decode_steps < prefill_pad_size:
      max_decode_steps = -1
      intermediate_decode_steps = -1
    elif intermediate_decode_steps is None:
      intermediate_decode_steps = max_decode_steps
    else:
      intermediate_decode_steps = min(
          intermediate_decode_steps, max_decode_steps)

    # generate token_ids and token_scores.
    batch_token_ids, batch_token_scores = self.decode_fn(
        self.model_params, inputs, segment_positions=segment_positions,
        segment_ids=segment_ids,
        prng_key=prng_key,
        temperature=sampling_params.temperature,
        top_k=sampling_params.top_k,
        top_p=sampling_params.top_p,
        max_decode_steps=max_decode_steps,
        intermediate_decode_steps=intermediate_decode_steps)

    # Post process the outputs.
    sample_outputs = []
    batch_size = batch_token_ids.shape[0]
    full_len = input_len + sampling_params.max_decode_steps
    for i in range(batch_size):
      raw_token_ids = batch_token_ids[i].tolist()
      raw_token_scores = batch_token_scores[i].tolist()
      token_ids = []
      token_scores = []
      for t, token_id in enumerate(raw_token_ids):
        token_ids.append(token_id)
        token_scores.append(raw_token_scores[t])
        if token_id in self.eos_ids and t >= input_len:
          break
      token_ids = token_ids[:full_len]
      token_scores = token_scores[:full_len]
      if token_ids[-1] in self.eos_ids and not include_eos_in_output_text:
        text = self.vocab.decode(token_ids[:-1])
      else:
        text = self.vocab.decode(token_ids)
      if print_debug_info:
        print(f's: {token_ids}')
        print(f'l: {token_scores}')
        print(f'sample: {text}')
        print(input_len)
      if include_bos_in_input_text:
        reconstructed_input_text = self.vocab.decode(
            [self.bos_id] + token_ids[:input_len])
      else:
        reconstructed_input_text = text[:len(input_text)]
      sample_outputs.append(SamplingOutput(
          input_text=reconstructed_input_text,
          output_text=text[len(input_text):],
          input_token_ids=[self.bos_id] + token_ids[:input_len],
          output_token_ids=token_ids[input_len:],
          sum_input_score=float(np.sum(token_scores[:input_len])),
          avg_input_score=float(np.mean(token_scores[:input_len])),
          input_token_scores=token_scores[:input_len],
          sum_output_score=float(np.sum(token_scores[input_len:])),
          avg_output_score=float(np.mean(token_scores[input_len:])),
          output_token_scores=token_scores[input_len:],
          params=sampling_params))
    if sampling_params.sort_by == 'avg_output_score':
      key_fn = lambda x: x.avg_output_score
    elif sampling_params.sort_by == 'sum_output_score':
      key_fn = lambda x: x.sum_output_score
    elif sampling_params.sort_by is None:
      key_fn = None
    else:
      raise ValueError(f'Unknown sort_by: {sampling_params.sort_by}')
    if key_fn is not None:
      sample_outputs = sorted(sample_outputs, key=key_fn, reverse=True)
    return sample_outputs

  def count_num_tokens(self, text: str) -> int:
    return len(self.vocab.encode(text))


def create_top_p_mask(logits, top_p):
  sorted_logits = jnp.flip(jnp.sort(logits, axis=-1), axis=-1)
  probs = jax.nn.softmax(sorted_logits, axis=-1)
  d = (jnp.cumsum(probs, axis=-1) <= top_p).astype(jnp.float32)
  indices = jnp.sum(d, axis=-1, keepdims=True).astype(jnp.int32)
  threshold = jnp.take_along_axis(sorted_logits, indices, axis=-1)
  mask = (logits < threshold).astype(logits.dtype)
  return mask


def create_top_k_mask(logits, top_k):
  # logits: [batch_size, 1, vocab_size]
  sorted_logits = jnp.sort(logits, axis=-1)
  indices = jnp.full(logits.shape[:-1] + (1,), -top_k)
  threshold = jnp.take_along_axis(sorted_logits, indices, axis=-1)
  mask = (logits < threshold).astype(logits.dtype)
  return mask


def sample_from_logits(key, logits, temperature=1.0, top_k=-1, top_p=1.0):
  """Samples from the given logits."""
  logits = logits[:, -1, :].reshape(logits.shape[0], 1, logits.shape[-1])
  def true_fn(logits):
    new_ids = jnp.argmax(logits, axis=-1)
    return new_ids
  def false_fn(logits):
    logits /= temperature
    def top_p_mask_fn(logits):
      return create_top_p_mask(logits, top_p)
    def top_k_mask_fn(logits):
      return create_top_k_mask(logits, top_k)
    mask = jax.lax.cond(top_k > 0, top_k_mask_fn, top_p_mask_fn, logits)
    logits += mask * get_large_negative_value(logits.dtype)
    new_ids = jax.random.categorical(
        key, logits.astype(jnp.float32), axis=-1).reshape(
            [logits.shape[0], 1])
    new_ids = jax.lax.with_sharding_constraint(new_ids, mesh_sharding(None))
    return new_ids
  return jax.lax.cond(temperature == 0.0, true_fn, false_fn, logits)


def pad_kv_cache(d, pad_steps, use_scan=True):
  """Pads the KV cache."""
  for k, v in d.items():
    if k == 'k' or k == 'v':
      pad_widths = ((0, 0), (0, pad_steps), (0, 0), (0, 0))
      if use_scan:
        pad_widths = ((0, 0),) + pad_widths
      d[k] = jnp.pad(v, pad_widths, constant_values=0)
    elif k == 'cache_mask':
      pad_widths = ((0, 0), (0, pad_steps))
      if use_scan:
        # Add another dimension for scan layers.
        pad_widths = ((0, 0),) + pad_widths
      d[k] = jnp.pad(v, pad_widths, constant_values=1)
    else:
      d[k] = pad_kv_cache(v, pad_steps, use_scan=use_scan)
  return d


def compute_loglikelihood(input_tokens, logits, temperature=1.0,
                          use_lookup=False):
  """Computes the loglikelihood of the given tokens."""
  # Compute loss in float32.
  # Logits: [batch_size, seq_len, vocab_size]
  logits = jnp.asarray(logits, dtype=jnp.float32) / jnp.asarray(
      temperature, dtype=jnp.float32)
  lse = jax.nn.logsumexp(logits, axis=-1)
  if use_lookup:
    targets = jnp.asarray(input_tokens, dtype=jnp.int32)
    target_logit = jnp.take_along_axis(
        logits, targets[:, :, None], axis=-1).squeeze(axis=-1)
  else:
    targets = jax.nn.one_hot(input_tokens, logits.shape[-1], axis=-1)
    target_logit = jnp.einsum('blv,blv->bl', targets, logits)
  # Calculate the loglikelihood of the target token.
  target_ll = target_logit - lse
  return target_ll, (target_logit, lse)


def sample_decode(params, inputs, segment_positions, segment_ids,
                  prng_key, apply_fn,
                  eos_ids, pad_id=0,
                  use_scan=True, temperature=1.0, top_k=-1, top_p=1.0,
                  intermediate_decode_steps=256,
                  max_decode_steps=256):
  """Run decoding by sampling starting from given inputs."""
  # inputs: all tokens ids plus bos_id, padded to prefill size.
  batch_size = inputs.shape[0]

  # Initialize decode state that holds the KV cache.
  decode_state = {}
  logits, extra_output = apply_fn(
      params, inputs, decode_state=decode_state,
      segment_positions=segment_positions,
      segment_ids=segment_ids)
  decode_state = extra_output['decode_state']
  # All input tokens not counting the bos token, thus minus 1.
  input_len = jnp.asarray(segment_ids == 1, dtype=jnp.int32).sum(axis=1)[0] - 1
  input_token_scores, _ = compute_loglikelihood(
      inputs[:, 1:], logits[:, :-1], temperature=temperature)

  if max_decode_steps < 0:
    num_iters = 0
  else:
    num_iters = ((max_decode_steps + intermediate_decode_steps - 1) //
                 intermediate_decode_steps)

  # All tokens not counting the bos token.
  tokens = inputs[:, 1:]
  token_scores = input_token_scores
  start_tokens = jax.lax.dynamic_slice_in_dim(
      inputs, input_len, 1, axis=1)

  start_position = input_len
  for i in range(num_iters + 1):
    if i > 0:
      decode_state = pad_kv_cache(
          decode_state, intermediate_decode_steps, use_scan=use_scan)
      # Create the array to hold the generated token ids.
      new_tokens = jnp.full(
          (batch_size, intermediate_decode_steps),
          fill_value=pad_id, dtype=jnp.int32)
      # Compute loss on the last token.
      new_token_scores = jnp.full(
          (batch_size, intermediate_decode_steps),
          fill_value=0.0, dtype=jnp.float32)
      tokens = jnp.concatenate([tokens, new_tokens], axis=1)
      token_scores = jnp.concatenate([token_scores, new_token_scores], axis=1)
    sampling_state = continue_decode(
        apply_fn, params, start_tokens,
        tokens=tokens,
        token_scores=token_scores,
        decode_state=decode_state, prng_key=prng_key,
        start_position=start_position,
        input_len=input_len,
        batch_size=batch_size, pad_id=pad_id, eos_ids=eos_ids,
        temperature=temperature, top_k=top_k, top_p=top_p)
    (start_tokens, segment_positions, decode_state,
     prng_key, token_scores, tokens) = sampling_state
    start_position = segment_positions[0][0]

  return tokens, token_scores


def continue_decode(
    apply_fn, params, start_tokens, decode_state, prng_key,
    start_position, input_len,
    batch_size, pad_id, eos_ids,
    tokens=None, token_scores=None,
    temperature=1.0, top_k=-1, top_p=1.0):
  """Continue the decoding process."""

  init_segment_positions = jnp.full((batch_size, 1), fill_value=start_position)
  init_sampling_state = (
      start_tokens,
      init_segment_positions,
      decode_state, prng_key, token_scores, tokens)

  is_output_token = jnp.reshape(
      jnp.arange(start=1, stop=tokens.shape[1]+1) > input_len,
      [1, -1])

  def body_fn(sampling_state):
    (input_tokens, segment_positions, decode_state,
     prng_key, token_scores, tokens) = sampling_state
    prng_key, key = jax.random.split(prng_key, 2)
    logits, extra_output = apply_fn(
        params, input_tokens, decode_state=decode_state,
        # Add 1 to the segment_positions to account for the bos_id.
        segment_positions=segment_positions)
    input_tokens = sample_from_logits(
        key, logits, temperature=temperature, top_k=top_k, top_p=top_p)
    token_score, _ = compute_loglikelihood(
        input_tokens, logits, temperature=temperature)
    # Assume all tokens in the batch share the same position.
    tokens = jax.lax.dynamic_update_slice_in_dim(
        tokens, input_tokens,
        segment_positions[0][0], axis=1)
    token_scores = jax.lax.dynamic_update_slice_in_dim(
        token_scores, token_score,
        segment_positions[0][0], axis=1)

    decode_state = extra_output['decode_state']
    segment_positions += 1
    return (input_tokens, segment_positions, decode_state,
            prng_key, token_scores, tokens)

  def cond_fn(sampling_state):
    tokens = sampling_state[-1]
    eos_reached = jnp.logical_and(
        functools.reduce(
            jnp.logical_or,
            [tokens == eos_id for eos_id in eos_ids]),
        is_output_token).any(axis=1)
    # Check whether all the pad locations are filled.
    len_reached = (tokens[:, -1] != pad_id)
    return ~(eos_reached | len_reached).all()

  final_sampling_state = jax.lax.while_loop(
      cond_fn, body_fn, init_sampling_state)
  return final_sampling_state


################################################################################
# Utilities


def get_dtype(dtype_name: str) -> DTypeLike:
  return dict(
      bfloat16=jnp.bfloat16,
      float32=jnp.float32,
      float16=jnp.float16,
      float8_e4m3fn=jnp.float8_e4m3fn,
      float8_e4m3b11fnuz=jnp.float8_e4m3b11fnuz,
      float8_e4m3fnuz=jnp.float8_e4m3fnuz,
      float8_e5m2=jnp.float8_e5m2,
      float8_e5m2fnuz=jnp.float8_e5m2fnuz,
      int8=jnp.int8,
  )[dtype_name]


def build_global_array(inputs, global_shape, init_sharding, final_sharding):
  arrays = jax.device_put(
      jnp.split(inputs, len(jax.local_devices()), axis=0),
      jax.local_devices())
  arr = jax.make_array_from_single_device_arrays(
      global_shape, init_sharding, arrays)
  arr = jax.lax.with_sharding_constraint(arr, final_sharding)
  return arr


def make_global_array(arr, data_partition):
  return jax.make_array_from_callback(
      arr.shape, mesh_sharding(data_partition),
      lambda idx: arr[idx])


def make_shape_dtype_struct_with_sharding(
    x: jax.Array, sharding: jax.sharding.Sharding) -> jax.ShapeDtypeStruct:
  return jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=sharding)


def eval_shape_with_sharding(
    fn: Callable[..., Any], *args, **kwargs) -> jax.ShapeDtypeStruct:
  """Return output shape and sharding information for given function."""
  jitted_fn = jax.jit(fn)
  shapes = jax.eval_shape(jitted_fn, *args, **kwargs)
  shardings = jitted_fn.lower(*args, **kwargs).compile().output_shardings
  return jax.tree_util.tree_map(
      make_shape_dtype_struct_with_sharding, shapes, shardings)


def get_scaling_info(config, also_print=False):
  """Get scaling information for the given config."""
  model_cls = ModelRegistry.get(config.model_name)
  model = model_cls(config)
  info_dict = {}
  params = jax.eval_shape(model.init, jax.random.PRNGKey(0))
  num_params = np.sum(jax.tree_util.tree_leaves(
      jax.tree_util.tree_map(
          lambda x: jnp.prod(jnp.array(x.shape)), params)))
  num_tokens = (np.float32(config.batch_size) * config.seq_len *
                config.num_train_steps)
  num_embedding_params = config.vocab_size * config.model_dim
  num_non_embedding_params = num_params - num_embedding_params
  num_flops = num_params * num_tokens * 6
  info_dict['num_params'] = num_params
  info_dict['num_non_embedding_params'] = num_non_embedding_params
  info_dict['num_embedding_params'] = num_embedding_params
  info_dict['embedding_params_ratio'] = num_embedding_params / num_params
  info_dict['num_tokens'] = num_tokens
  info_dict['num_flops'] = num_params * num_tokens * 6
  if also_print:
    print(f'num_params: {num_params/1e6} M')
    print(f'num_non_embedding_params: {num_non_embedding_params/1e6} M')
    print(f'num_embedding_params: {num_embedding_params/1e6} M')
    print(f'embedding_params_ratio: {num_embedding_params/num_params}')
    print(f'num_tokens: {num_tokens/1e6} M')
    print(f'num_tokens / num_params: {num_tokens / num_params}')
    print(f'num_tokens / num_non_embedding_params: '
          f'{num_tokens / num_non_embedding_params}')
    print(f'num_flops: {num_flops}')
  return info_dict


def quantize_array(w, symmetric=False):
  if symmetric:
    scale = jnp.max(jnp.abs(w)) / 127
    quant_w = jnp.asarray(jnp.round(w / scale), dtype=jnp.int8)
    result = {'quant_array': quant_w, 'scale': scale}
  else:
    scale = (jnp.max(w) - jnp.min(w)) / 256
    zero_point = (jnp.max(w) + jnp.min(w)) / 2
    quant_w = jnp.asarray(jnp.round((w - zero_point) / scale), dtype=jnp.int8)
    result = {'quant_array': quant_w, 'scale': scale,
              'zero_point': zero_point}
  return result


def maybe_dequantize_array(a, dtype=jnp.bfloat16):
  if isinstance(a, jnp.ndarray):
    return jnp.asarray(a, dtype=dtype)
  quant_w = a['quant_array']
  dequant_w = jnp.asarray(quant_w, dtype=dtype) * a['scale']
  if 'zero_point' in a:
    dequant_w += a['zero_point']
  return dequant_w


def quantize_tfm_params(params, symmetric=False):
  """Quantize the Transformer parameters."""
  if isinstance(params, jnp.ndarray):
    return params
  quant_params = {}
  for key in params:
    if key == 'attn' or key.startswith('ffn_'):
      subparams = copy.copy(params[key])
      for subkey in [
          'w', 'b', 'o_proj', 'q_proj', 'k_proj', 'v_proj', 'qkv_proj']:
        if subkey in subparams:
          subparams[subkey] = quantize_array(
              subparams[subkey], symmetric=symmetric)
      quant_params[key] = subparams
    else:
      quant_params[key] = quantize_tfm_params(
          params[key], symmetric=symmetric)
  return quant_params
