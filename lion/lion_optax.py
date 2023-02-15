# Copyright 2023 Google Research. All Rights Reserved.
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
"""Optax implementation of the Lion optimizer."""

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
import optax


def _scale_by_learning_rate(
    learning_rate: optax.ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def lion(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
) -> optax.GradientTransformation:
  """Lion.

  Args:
    learning_rate: A fixed global scaling factor.
    b1: Exponential decay rate to combine the gradient and the moment.
    b2: Exponential decay rate to track the moment of past gradients.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip. Note
      that the Adam gradient transformations are applied to all parameters.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return optax.chain(
      scale_by_lion(
          b1=b1, b2=b2, mu_dtype=mu_dtype),
      optax.add_decayed_weights(weight_decay, mask),
      _scale_by_learning_rate(learning_rate),
  )


def update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


class ScaleByLionState(NamedTuple):
  """State for the Lion algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates


def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the Lion algorithm.

  Args:
    b1: rate for combining moment and the current grad.
    b2: decay rate for the exponentially weighted average of grads.
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
    return ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)

  def update_fn(updates, state, params=None):
    del params
    mu = update_moment(updates, state.mu, b2, 1)
    mu = jax.tree_map(lambda x: x.astype(mu_dtype), mu)
    count_inc = optax.safe_int32_increment(state.count)
    updates = jax.tree_util.tree_map(
        lambda g, m: jnp.sign((1. - b1) * g + b1 * m), updates, state.mu)
    return updates, ScaleByLionState(count=count_inc, mu=mu)

  return optax.GradientTransformation(init_fn, update_fn)
