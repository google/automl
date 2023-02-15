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
"""TF1 implementation of the Lion optimizer."""
from typing import Optional, Union, Callable

import tensorflow.compat.v1 as tf
from tensorflow.python.ops import resource_variable_ops

VType = Union[Callable, float, tf.Tensor]


class Lion(tf.compat.v1.train.Optimizer):
  """Optimizer that implements the discovered algorithm in automl-hero."""

  def __init__(self,
               learning_rate: VType = 0.0001,
               beta1: VType = 0.9,
               beta2: VType = 0.99,
               wd: Optional[VType] = 0.0,
               use_locking=False,
               name="Lion"):
    r"""Construct a new Lion optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor. The rate to combine
        the gradient and the moment estimate.
      beta2: A float value or a constant float tensor. The exponential decay
        rate for the moment estimate.
      wd: Optional[A float value or a constant float tensor].
        The decoupled weight decay.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
    """
    super(Lion, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._wd = None if isinstance(wd, float) and wd < 0 else wd

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._wd_t = None

  def _create_slots(self, var_list):
    # Create slots for the moment.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    wd = self._call_if_callable(self._wd)

    self._lr_t = tf.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = tf.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = tf.convert_to_tensor(beta2, name="beta2")
    if wd is not None:
      self._wd_t = tf.convert_to_tensor(wd, name="weight_decay")

  def _apply_dense_shared(self, grad, var):
    m = self.get_slot(var, "m")

    lr_t = tf.cast(self._lr_t, dtype=var.dtype)
    beta1_t = tf.cast(self._beta1_t, dtype=var.dtype)
    beta2_t = tf.cast(self._beta2_t, dtype=var.dtype)
    if self._wd_t is None:
      weight_decay_t = None
    else:
      weight_decay_t = tf.cast(self._wd_t, dtype=var.dtype)

    updates_grad = tf.sign(m * beta1_t + grad * (1. - beta1_t))
    if weight_decay_t is not None:
      updates_grad = updates_grad + var * weight_decay_t

    var_update = tf.assign_sub(
        var, lr_t * updates_grad, use_locking=self._use_locking)
    with tf.control_dependencies([var_update]):
      m_update = tf.assign(m, m * beta2_t + grad * (1. - beta2_t))
    return tf.group(*[var_update, m_update])

  def _apply_dense(self, grad, var):
    return self._apply_dense_shared(grad, var)

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense_shared(grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    m = self.get_slot(var, "m")
    lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
    wd_t = tf.cast(self._wd_t, var.dtype.base_dtype)

    m_update = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
    with tf.control_dependencies([m_update]):
      m_update = scatter_add(m, indices, grad * (1. - beta1_t))
      with tf.control_dependencies([m_update]):
        var_update = tf.assign_sub(
            var,
            lr_t * (tf.sign(m) + var * wd_t),
            use_locking=self._use_locking)
        with tf.control_dependencies([var_update]):
          m_update = scatter_add(m, indices, grad * (beta1_t - 1.))
          with tf.control_dependencies([m_update]):
            m_update = tf.assign(
                m, m * beta2_t / beta1_t, use_locking=self._use_locking)
            with tf.control_dependencies([m_update]):
              m_update = scatter_add(m, indices, grad * (1. - beta2_t))
    return tf.group(*[var_update, m_update])

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_shared(
        grad.values,
        var,
        grad.indices,
        lambda x, i, v: tf.scatter_add(
            x,
            i,
            v,
            use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
    with tf.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    return self._apply_sparse_shared(grad, var, indices,
                                     self._resource_scatter_add)
