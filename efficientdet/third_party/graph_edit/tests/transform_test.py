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
"""Tests for tensorflow.contrib.graph_editor."""

import sys
import collections
import functools
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

import third_party.graph_edit as ge
from third_party.graph_edit.tests import match

tf.disable_eager_execution()

# Precision tolerance for floating-point value tests.

ERROR_TOLERANCE = 1e-3


class TransformTest(test.TestCase):
  """Test transform."""

  def setUp(self):
    """Set up."""
    self.graph = ops.Graph()
    with self.graph.as_default():
      c0 = constant_op.constant(1.0, shape=[10], name="Const")  # pylint: disable=W0212
      c0.op._set_attr("_foo", attr_value_pb2.AttrValue(s=b"foo"))  # pylint: disable=W0212
      c1 = constant_op.constant(1.0, shape=[10], name="Const")
      c2 = constant_op.constant(1.0, shape=[10], name="Const")
      i = constant_op.constant(1.0, shape=[10], name="Input")
      self.o = math_ops.add(c2, math_ops.add(c1, math_ops.add(c0, i)))

  def test_copy(self):
    """Test copy graph."""
    graph = ops.Graph()
    _, info = ge.copy(self.graph, graph)
    self.assertEqual(
        set(op.name for op in self.graph.get_operations()),
        set(op.name for op in graph.get_operations()))
    src_ops = self.graph.get_operations()
    dst_ops = graph.get_operations()
    for op in src_ops:
      op_ = info.transformed(op)
      self.assertTrue(op_ in dst_ops)
      self.assertEqual(op.name, op_.name)
      self.assertEqual(info.original(op_), op)
    src_ts = ge.util.get_tensors(self.graph)
    dst_ts = ge.util.get_tensors(graph)
    for t in src_ts:
      t_ = info.transformed(t)
      self.assertTrue(t_ in dst_ts)
      self.assertEqual(t.name, t_.name)
      self.assertEqual(info.original(t_), t)

  def test_copy_assert(self):
    """Test copy graph."""
    ops.reset_default_graph()
    a = constant_op.constant(1)
    b = constant_op.constant(1)
    eq = math_ops.equal(a, b)
    assert_op = control_flow_ops.Assert(eq, [a, b])
    with ops.control_dependencies([assert_op]):
      _ = math_ops.add(a, b)
    sgv = ge.make_view([assert_op, eq.op, a.op, b.op])
    copier = ge.Transformer()
    _, info = copier(sgv, sgv.graph, "", "")
    new_assert_op = info.transformed(assert_op)
    self.assertIsNotNone(new_assert_op)

  def test_transform(self):
    """Test transform graph."""
    transformer = ge.Transformer()

    def my_transform_op_handler(info, op, new_inputs):
      add_noise = op.name.startswith("Add")
      op_, op_outputs_ = ge.transform.copy_op_handler(info, op, new_inputs)
      if not add_noise:
        return op_, op_outputs_
      # add some noise to op
      with info.graph_.as_default():
        t_ = math_ops.add(
            constant_op.constant(1.0, shape=[10], name="Noise"),
            op_.outputs[0],
            name="AddNoise")
      # return the "noisy" op
      return op_, [t_]

    transformer.transform_op_handler = my_transform_op_handler

    graph = ops.Graph()
    transformer(self.graph, graph, "", "")
    matcher0 = match.OpMatcher("AddNoise").input_ops(
        "Noise",
        match.OpMatcher("Add").input_ops("Const", "Input"))
    matcher1 = match.OpMatcher("AddNoise_1").input_ops(
        "Noise_1",
        match.OpMatcher("Add_1").input_ops("Const_1", matcher0))
    matcher2 = match.OpMatcher("AddNoise_2").input_ops(
        "Noise_2",
        match.OpMatcher("Add_2").input_ops("Const_2", matcher1))
    top = ge.select_ops("^AddNoise_2$", graph=graph)[0]
    self.assertTrue(matcher2(top))

  def test_transform_nodedef_fn(self):
    """Test transform nodedef_fn."""
    transformer = ge.Transformer()

    def nodedef_fn(node_def):
      if "_foo" in node_def.attr:
        del node_def.attr["_foo"]
      node_def.attr["_bar"].s = b"bar"
      return node_def

    my_copy_op_handler = functools.partial(
        ge.transform.copy_op_handler, nodedef_fn=nodedef_fn)
    transformer.transform_op_handler = my_copy_op_handler

    graph = ops.Graph()
    transformer(self.graph, graph, "", "")

    c0_before = self.graph.get_operation_by_name("Const")
    c0_after = graph.get_operation_by_name("Const")
    self.assertEquals(c0_before.get_attr("_foo"), b"foo")
    with self.assertRaises(ValueError):
      c0_after.get_attr("_foo")

    all_ops = graph.get_operations()
    for op in all_ops:
      self.assertEquals(op.get_attr("_bar"), b"bar")

  def test_copy_with_input_replacements(self):
    """Test copy with input replacements."""
    with self.graph.as_default():
      ten = constant_op.constant(10.0, shape=[10], name="Input")
      sgv, _ = ge.copy_with_input_replacements(self.o.op,
                                               {self.o.op.inputs[1]: ten})
      with session.Session() as sess:
        val = sess.run(sgv.outputs[0])
      self.assertNear(
          np.linalg.norm(val - np.array([11])), 0.0, ERROR_TOLERANCE)

  def test_graph_replace(self):
    """Test replace graph."""
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    c_new = ge.graph_replace(c, {a: a_new})
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      c_val, c_new_val = sess.run([c, c_new])
    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val, 3.001, ERROR_TOLERANCE)

  def test_graph_replace_dict(self):
    """Test replace graph with dict."""
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    c_new = ge.graph_replace({"c": c}, {a: a_new})
    self.assertTrue(isinstance(c_new, dict))
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      c_val, c_new_val = sess.run([c, c_new])
    self.assertTrue(isinstance(c_new_val, dict))
    self.assertNear(c_val, 2.001, ERROR_TOLERANCE)
    self.assertNear(c_new_val["c"], 3.001, ERROR_TOLERANCE)

  def test_graph_replace_ordered_dict(self):
    """Test replace graph with ord dict."""
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    c_new = ge.graph_replace(collections.OrderedDict({"c": c}), {a: a_new})
    self.assertTrue(isinstance(c_new, collections.OrderedDict))

  def test_graph_replace_named_tuple(self):
    """Test replace graph with named tuple."""
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = variables.Variable(1.0, name="b")
    eps = constant_op.constant(0.001, name="eps")
    c = array_ops.identity(a + b + eps, name="c")
    a_new = constant_op.constant(2.0, name="a_new")
    one_tensor = collections.namedtuple("OneTensor", ["t"])
    c_new = ge.graph_replace(one_tensor(c), {a: a_new})
    self.assertTrue(isinstance(c_new, one_tensor))

  def test_graph_replace_missing(self):
    """Test replace missing."""
    ops.reset_default_graph()
    a = constant_op.constant(1.0, name="a")
    b = constant_op.constant(2.0, name="b")
    c = a + 2 * b
    d = constant_op.constant(2.0, name="d")
    res = ge.graph_replace([b, c], {a: d})
    self.assertEqual(res[0].name, "b:0")
    self.assertEqual(res[1].name, "add_1:0")

  def test_graph_replace_gradients(self):
    """Test replace gradients."""
    ops.reset_default_graph()
    w = variables.VariableV1(0.0, name="w")
    y = math_ops.multiply(math_ops.multiply(w, w, name="mul1"), w, name="mul2")
    g = tf.gradients(y, w, name="grad")[0]

    # Extract the operations.
    replacement_ts = {w.op: g}
    # replacement_ts = {w.value(): g}
    original_mul1_grad = (
        ops.get_default_graph().get_operation_by_name("grad/mul1_grad/Mul_1"))

    # Should not raise exception.
    res = ge.graph_replace(g, replacement_ts, dst_scope="res")

    # Extract the operations after graph_replace.
    result_mul1_grad = (
        ops.get_default_graph().get_operation_by_name(
            "res/grad/mul1_grad/Mul_1"))

    # Make sure _original_ops are as expected.
    self.assertEqual(original_mul1_grad._original_op.name, u"mul1")  # pylint: disable=W0212
    self.assertEqual(result_mul1_grad._original_op.name, u"res/mul1")  # pylint: disable=W0212
    self.assertNotEqual(res.name, g.name)
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())

      g_val, res_val = sess.run([g, res])
    self.assertNear(g_val, 0.0, ERROR_TOLERANCE)
    self.assertNear(res_val, 0.0, ERROR_TOLERANCE)

  # def test_graph_while_loop(self):
  #   """Test while loop in copied graph."""
  #   graph = ops.Graph()
  #   with graph.as_default():
  #     max_index = array_ops.placeholder(dtype=dtypes.int32, shape=tuple())
  #     index_start = constant_op.constant(1)
  #     sum_start = constant_op.constant(0)
  #     _, result = control_flow_ops.while_loop(
  #         cond=lambda i, unused_s: i <= max_index,
  #         body=lambda i, s: (i + 1, s + i),
  #         loop_vars=[index_start, sum_start])
  #   copied_graph = ops.Graph()
  #   _, copy_info = ge.copy(graph, dst_graph=copied_graph, dst_scope="imported")
  #   copied_result = copy_info.transformed(result)
  #   copied_max_index = copy_info.transformed(max_index)
  #   with copied_graph.as_default():
  #     with session.Session() as sess:
  #       n = 10
  #       sum_val = sess.run(copied_result, feed_dict={copied_max_index: n})
  #       self.assertEqual(sum_val, 55)

  # def test_graph_cond(self):
  #   """Test cond in copied graph."""
  #   graph = ops.Graph()
  #   with graph.as_default():
  #     choice = array_ops.placeholder(shape=(), dtype=dtypes.bool)
  #     result = tf.cond(choice, lambda: constant_op.constant(1),
  #                      lambda: constant_op.constant(2))
  #   copied_graph = ops.Graph()
  #   _, copy_info = ge.copy(graph, dst_graph=copied_graph, dst_scope="imported")
  #   copied_result = copy_info.transformed(result)
  #   copied_choice = copy_info.transformed(choice)
  #   with copied_graph.as_default():
  #     with session.Session() as sess:
  #       res = sess.run(copied_result, feed_dict={copied_choice: True})
  #       self.assertEqual(res, 1)
  #       res = sess.run(copied_result, feed_dict={copied_choice: False})
  #       self.assertEqual(res, 2)


if __name__ == "__main__":
  test.main()
