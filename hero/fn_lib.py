"""The functions used in the program search.

This library implements the functions that will be used in the program search.
Having a shared library of functions will make it easier to share the discovered
algorithms between different tasks. For example, if we want to evaluate one
algorithm on both language and vision tasks.
"""

from typing import Sequence, Optional

import jax
import jax.nn

import jax.numpy as jnp

import core


@jax.jit
def interpolate(x, y, weight):
  return core.tree_add(
      core.tree_mult(1.0 - weight, x), core.tree_mult(weight, y))


@jax.jit
def global_norm(tree):
  leaves = jax.tree_leaves(tree)
  norm = jnp.sqrt(sum([jnp.vdot(x, x) for x in leaves]))
  return norm


@jax.jit
def tree_dot(tree1, tree2):
  tree_result = jax.tree_map(jnp.vdot, tree1, tree2)
  return sum(jax.tree_leaves(tree_result))


@jax.jit
def tree_cosine_sim(tree1, tree2):
  tree_result = jax.tree_map(jnp.vdot, tree1, tree2)
  dot_result = sum(jax.tree_leaves(tree_result))
  norm1 = global_norm(tree1)
  norm2 = global_norm(tree2)
  return dot_result / (norm1 * norm2)


@jax.jit
def clip_by_global_norm(tree, clip_norm):
  l2_g = global_norm(tree)
  g_factor = jnp.minimum(1.0, clip_norm / l2_g)
  return core.tree_mult(g_factor, tree)


def get_math_fns(allowed_fns: Optional[Sequence[str]] = None):
  """Get the dictionary containing the math functions."""

  fn_dict = {}

  noarg_fn_dict = dict(
      get_pi=lambda: jnp.pi, get_e=lambda: jnp.e, get_eps=lambda: 1e-8)

  for k, v in noarg_fn_dict.items():
    fn_dict[k] = core.Function(v, 0, [])

  def nonneg(f):
    def g(x):
      return f(jnp.fabs(x))
    return g

  def map_to_tree(f):
    def g(x):
      return jax.tree_map(f, x)
    return g

  unary_fn_dict = dict(
      abs=jnp.abs,
      cos=jnp.cos,
      sin=jnp.sin,
      tan=jnp.tan,
      arcsin=jnp.arcsin,
      arccos=jnp.arccos,
      arctan=jnp.arctan,
      sinh=jnp.sinh,
      cosh=jnp.cosh,
      tanh=jnp.tanh,
      arcsinh=jnp.arcsinh,
      arccosh=jnp.arccosh,
      arctanh=jnp.arctanh,
      exp=jnp.exp,
      exp2=jnp.exp2,
      exp10=lambda x: jnp.power(10, x),
      expm1=jnp.expm1,
      log=nonneg(jnp.log),
      log10=nonneg(jnp.log10),
      log2=nonneg(jnp.log2),
      log1p=lambda x: jnp.log(jnp.fabs(1 + x)),
      square=jnp.square,
      sqrt=nonneg(jnp.sqrt),
      cube=lambda x: jnp.power(x, 3),
      cbrt=lambda x: jnp.cbrt,
      sign=jnp.sign,
      reciprocal=jnp.reciprocal,
      norm=jnp.linalg.norm,
      invert=jnp.invert,
      negative=jnp.negative)

  for k, v in unary_fn_dict.items():
    fn_dict[k] = core.Function(map_to_tree(v), 1, [core.is_numeric])

  fn_dict['global_norm'] = core.Function(global_norm, 1, [core.is_numeric])

  fn_dict['interpolate'] = core.Function(
      interpolate, 3,
      [core.ExampleAnnotation(1.0).check, core.is_numeric, core.is_numeric])

  fn_dict['dot'] = core.Function(
      tree_dot, 2, [core.is_numeric, core.is_numeric])

  fn_dict['cosine_sim'] = core.Function(
      tree_cosine_sim, 2, [core.is_numeric, core.is_numeric])

  fn_dict['clip_by_global_norm'] = core.Function(
      clip_by_global_norm, 2,
      [core.is_numeric, core.ExampleAnnotation(1.0).check])

  fn_dict['power'] = core.Function(
      jnp.power, 2, [core.is_numeric, core.is_numeric])

  if allowed_fns is not None:
    new_fn_dict = {}
    allowed_fns_set = set(allowed_fns)
    for key in fn_dict:
      if key in allowed_fns_set:
        new_fn_dict[key] = fn_dict[key]
  else:
    new_fn_dict = fn_dict

  return new_fn_dict

