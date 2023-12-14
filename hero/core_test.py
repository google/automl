"""Tests for Hero core."""

import logging
import time
import unittest

import flax
import jax
import jax.numpy as jnp

import core
import fn_lib


_ADAM_STRING = """
def train(w, m, v, gradient):
  m = interpolate(m, gradient, 0.1)
  g2 = square(gradient)
  v = interpolate(v, g2, 0.001)
  epsilon = 1e-8
  sqrt_v = sqrt(v)
  sqrt_v = sqrt_v + epsilon
  update = m / sqrt_v
  return (update, m, v)
"""

_ADAM_STRING_ARG_ANNOTATION = """
def train(w: Params, m: Params, v: Params, gradient: Params):
  m = interpolate(m, gradient, 0.1)
  g2 = square(gradient)
  v = interpolate(v, g2, 0.001)
  epsilon = 1e-8
  sqrt_v = sqrt(v)
  sqrt_v = sqrt_v + epsilon
  update = m / sqrt_v
  return (update, m, v)
"""


def get_mock_regression_dataset(seed=1):
  num_features = 2
  num_train = 5
  num_valid = 3
  key = jax.random.PRNGKey(seed)
  key, subkey = jax.random.split(key)
  # The ground truth weights for each feature.
  weights = jnp.arange(num_features)
  key, subkey = jax.random.split(key)
  train_data = jax.random.normal(subkey, shape=[num_train, num_features])
  train_labels = jnp.dot(train_data, weights)
  key, subkey = jax.random.split(key)
  valid_data = jax.random.normal(subkey, shape=[num_valid, num_features])
  valid_labels = jnp.dot(valid_data, weights)
  return train_data, train_labels, valid_data, valid_labels


class SimpleNN(flax.linen.Module):

  @flax.linen.compact
  def __call__(self, x):
    x = flax.linen.Dense(features=1)(x)
    return x


@jax.jit
def mse_loss(preds, labels):
  preds = jnp.reshape(preds, labels.shape)
  return jnp.mean(jnp.square(preds - labels))


@jax.jit
def eval_model(params, data, labels):
  model = SimpleNN()
  preds = model.apply(params, data)
  return mse_loss(preds, labels)


@jax.jit
def compute_grad(params, data, labels):
  def loss_fn(params):
    model = SimpleNN()
    preds = model.apply(params, data)
    loss = mse_loss(preds, labels)
    return loss
  grad = jax.grad(loss_fn)(params)
  return grad


def get_mock_setup():
  t1 = time.time()
  (train_data, train_labels,
   valid_data, valid_labels) = get_mock_regression_dataset(1)
  t2 = time.time()
  logging.info('%d sec used creating dataset.', t2-t1)

  t1 = time.time()
  model = SimpleNN()
  params = model.init(
      jax.random.PRNGKey(0), train_data.shape)
  t2 = time.time()
  logging.info('%d sec used in initialization.', t2-t1)
  return (train_data, train_labels, valid_data, valid_labels), params


_CODE_1 = core.normalize_code("""
    def train1(params: Params, params_1: Params, steps: Int, data: Data, labels: Labels):
      v788 = (data + data)
      v559 = (labels / v788)
      v663 = (v788 - v788)
      params_1 = (1.0 - params_1)
      v794 = (v559 - labels)
      v954 = (v663 / labels)
      v332 = eval(params_1, v788, labels)
      v794 = (v559 / v954)
      v571 = (4.7420220375061035 * params)
      v419 = (v332 / labels)
      lr = 0.20240235328674316
      v615 = (v332 * data)
      v663 = (v615 - v788)
      v804 = (3.4247617721557617 + params_1)
      params_1 = compute_grad(v571, v615, labels)
      v788 = (v788 + v788)
      params_1 = (params - params_1)
      v90 = (labels / 155.4664306640625)
      v371 = (data * v788)
      v435 = (v794 * v332)
      v420 = (0.425348162651062 + v788)
      v770 = (params - params)
      v282 = (steps * v794)
      v571 = compute_grad(v804, v615, labels)
      v770 = (lr + v770)
      params_1 = (params - 0.574072539806366)
      params_1 = (params / v804)
      v770 = (params_1 * v804)
      v634 = (v90 - v954)
      v435 = (v420 * v90)
      v937 = (labels * labels)
      v282 = (labels - v663)
      v813 = (v937 / v420)
      v423 = (v435 / labels)
      v282 = (v559 / 1.0)
      v804 = (v770 * v804)
      v663 = (lr + v788)
      v559 = (v954 * v937)
      v332 = eval(v571, v371, labels)
      v788 = (v788 + v371)
      gradient = compute_grad(v770, v788, labels)
      v457 = compute_grad(v571, v615, v937)
      v634 = (v282 - v559)
      v143 = (v90 - steps)
      v677 = eval(v770, v615, v937)
      v571 = (gradient - steps)
      v332 = eval(v804, v788, labels)
      update = (gradient * lr)
      params = (params - update)
      params_1 = (params_1 + params_1)
      v954 = (v788 + v954)
      v527 = (steps + v371)
      v845 = (lr / params)
      return (params, params_1)
    """)


_SIMPLIFIED_CODE_1 = core.normalize_code("""
    def train1(params: Params, params_1: Params, steps: Int, data: Data, labels: Labels):
      v788 = (data + data)
      params_1 = (1.0 - params_1)
      lr = 0.20240235328674316
      v804 = (3.4247617721557617 + params_1)
      v788 = (v788 + v788)
      v371 = (data * v788)
      params_1 = (params / v804)
      v770 = (params_1 * v804)
      v788 = (v788 + v371)
      gradient = compute_grad(v770, v788, labels)
      update = (gradient * lr)
      params = (params - update)
      params_1 = (params_1 + params_1)
      return (params, params_1)
    """)


class CoreTest(unittest.TestCase):

  def assertLen(self, x, y):
    self.assertEqual(len(x), y)
  
  def test_funcall_generation(self):
    rng = jax.random.PRNGKey(0)
    namespace = core.Namespace(add_primitives=False)
    variables = [core.Symbol(x) for x in ['a', 'b', 'c', 'd']]
    variables.append(core.Atom(1.0))
    namespace[variables[0].name] = (jnp.ones([1, 2]), (jnp.ones([1, 2]), 1))
    namespace[variables[1].name] = (jnp.zeros([1, 2]), (jnp.zeros([1, 2]), 2))
    namespace[variables[2].name] = 1
    namespace[variables[3].name] = 2
    namespace['x'] = None
    namespace['y'] = 'distractor'
    tree_fn = core.Function(
        core.tree_add, 2, [core.is_numeric] * 2)
    tree_fn_name = 'tree_add'
    namespace[tree_fn_name] = tree_fn
    fn = core.Function(
        core.add, 2, [core.is_numeric] * 2)
    fn_name = 'add'
    namespace[fn_name] = fn
    generated_funcalls = list(core.FunCall.random_generate(rng, namespace))
    expected_funcalls = []

    # Tree ops can work between pytree and simple arrays.
    var_names = [var.to_string() for var in variables]
    for arg1 in var_names:
      for arg2 in var_names:
        expected_funcalls.append(
            '{}({}, {})'.format(tree_fn_name, arg1, arg2))

    # Simple add doesn't work across pytree and simple arrays.
    for arg1 in var_names[:2]:
      for arg2 in var_names[:2]:
        expected_funcalls.append(
            '{}({}, {})'.format(fn_name, arg1, arg2))
    for arg1 in var_names[2:]:
      for arg2 in var_names[2:]:
        expected_funcalls.append(
            '{}({}, {})'.format(fn_name, arg1, arg2))

    generated_funcalls = set([exp.to_string() for exp in generated_funcalls])
    expected_funcalls = set(expected_funcalls)
    self.assertEqual(generated_funcalls, expected_funcalls)

    # Different seed should have different candidate orders and
    # same seed show have the same order.
    rng1 = jax.random.PRNGKey(1)
    rng2 = jax.random.PRNGKey(2)
    generated_funcalls_1 = [
        exp.to_string() for exp in
        list(core.FunCall.random_generate(rng1, namespace))]
    generated_funcalls_2 = [
        exp.to_string() for exp in
        list(core.FunCall.random_generate(rng2, namespace))]
    generated_funcalls_3 = [
        exp.to_string() for exp in
        list(core.FunCall.random_generate(rng1, namespace))]
    self.assertEqual(generated_funcalls_1, generated_funcalls_3)
    self.assertNotEqual(generated_funcalls_1, generated_funcalls_2)
    # The order can be different but the contents should be the same
    # regardless of random seeds.
    self.assertEqual(set(generated_funcalls_1), set(generated_funcalls_2))
    self.assertEqual(set(generated_funcalls_1), set(generated_funcalls_3))

  def test_assignment_generation(self):
    rng = jax.random.PRNGKey(0)
    namespace = core.Namespace(add_primitives=False, max_num_var=1)
    var_names = ['a', 'b']
    namespace[var_names[0]] = 1.0
    namespace[var_names[1]] = (
        jnp.zeros([1, 2], dtype=jnp.float32),
        (jnp.zeros([1, 2], dtype=jnp.float32), 2.0))

    namespace['x'] = None
    namespace['y'] = 'distractor'
    tree_fn = core.Function(
        core.tree_add, 2, [core.is_numeric] * 2)
    tree_fn_name = 'tree_add'
    namespace[tree_fn_name] = tree_fn
    generated_exps = list(
        core.Assignment.random_generate(rng, namespace))
    generated_exps = set(
        [core.normalize_code(exp.to_string()) for exp in generated_exps])
    tmpl = '{} = tree_add({}, {})'
    var_names = ['a', 'b', 'v0']
    atom_string = core.Atom(1.0).to_string()
    expected_exps = []
    expected_exps.append(tmpl.format('b', 'b', 'b'))

    for arg in ['a', atom_string]:
      expected_exps.append(tmpl.format('b', 'b', arg))
      expected_exps.append(tmpl.format('b', arg, 'b'))

    for arg1 in  ['a', atom_string]:
      for arg2 in ['a', atom_string]:
        expected_exps.append(tmpl.format('a', arg1, arg2))

    for arg1 in ['a', 'b', atom_string]:
      for arg2 in ['a', 'b', atom_string]:
        expected_exps.append(tmpl.format('v0', arg1, arg2))

    expected_exps = set(
        [core.normalize_code(string) for string in expected_exps])
    print(generated_exps)
    print(expected_exps)
    self.assertEqual(generated_exps, expected_exps)

    # Different seed should have different candidate orders and
    # same seed show have the same order.
    rng1 = jax.random.PRNGKey(1)
    rng2 = jax.random.PRNGKey(2)
    generated_exps_1 = [
        exp.to_string() for exp in
        list(core.Assignment.random_generate(
            rng1, namespace))]
    generated_exps_2 = [
        exp.to_string() for exp in
        list(core.Assignment.random_generate(
            rng2, namespace))]
    generated_exps_3 = [
        exp.to_string() for exp in
        list(core.Assignment.random_generate(
            rng1, namespace))]
    self.assertEqual(generated_exps_1, generated_exps_3)
    self.assertNotEqual(generated_exps_1, generated_exps_2)
    # The order can be different but the contents should be
    # the same regardless of random seeds.
    self.assertEqual(set(generated_exps_1), set(generated_exps_2))
    self.assertEqual(set(generated_exps_1), set(generated_exps_3))

  def test_scalar_type_check(self):
    annotation = core.ExampleAnnotation(1)
    self.assertTrue(annotation.check(2))
    self.assertTrue(annotation.check(jnp.array(3)))
    self.assertFalse(annotation.check(2.0))
    self.assertFalse(annotation.check(jnp.array(3.0)))
    self.assertFalse(annotation.check(jnp.array([3, 4])))

  def test_array_type_check(self):
    annotation = core.ExampleAnnotation(jnp.ones([2, 3], dtype=jnp.float32))
    self.assertTrue(annotation.check(jnp.zeros([2, 3], dtype=jnp.float32)))
    # float32 and float64 should be treated as equal.
    self.assertTrue(annotation.check(jnp.zeros([2, 3], dtype=jnp.float64)))
    self.assertFalse(annotation.check(jnp.zeros([2, 4], dtype=jnp.float32)))
    self.assertFalse(annotation.check(jnp.zeros([2, 3], dtype=jnp.int32)))

  def test_pytree_type_check(self):
    annotation = core.ExampleAnnotation(
        {'a': 1, 'b': (jnp.array(1), jnp.zeros([1, 2]), 2)})
    v1 = {'a': 10, 'b': (jnp.array(4), jnp.ones([1, 2]), 3)}
    self.assertTrue(annotation.check(v1))
    v2 = {'c': 10, 'b': (jnp.array(4), jnp.ones([1, 2]), 3)}
    self.assertFalse(annotation.check(v2))
    v3 = {'a': 10.0, 'b': (jnp.array(4), jnp.ones([1, 2]), 3)}
    self.assertFalse(annotation.check(v3))
    v4 = {'a': 10, 'b': (jnp.array(4), jnp.ones([3, 5]), 3)}
    self.assertFalse(annotation.check(v4))
    v5 = {'a': 10, 'b': (jnp.array(4), jnp.ones([1, 2]), 3.0)}
    self.assertFalse(annotation.check(v5))

  def test_ignore_batch_axis(self):
    a1 = core.ExampleAnnotation(jnp.zeros([10, 3, 4]))
    a2 = core.ExampleAnnotation(jnp.zeros([10, 3, 4]), ignore_batch_axis=True)
    v1 = jnp.ones([3, 3, 4])
    self.assertFalse(a1.check(v1))
    self.assertTrue(a2.check(v1))

  def test_mutation(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      d = (a + b)
      c = (a - b)
      return c

    def _main_():
      return f(1, 1)
    """)
    namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
    program = core.Program.parse(code)
    string = program.to_string()
    self.assertEqual(core.normalize_code(string), code)
    _, success = program.execute(namespace)
    self.assertTrue(success)
    num_repeat = 3
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      success = program.mutate(rng, namespace)
      self.assertTrue(success)
    t2 = time.time()
    logging.info('Mutation time: %d sec', (t2-t1) / num_repeat)

  def test_random_deletion_1(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      d = (a + b)
      c = (a - b)
      e = (a * b)
      e = (a / e)
      return c

    def _main_():
      return f(1, 1)
    """)
    num_repeat = 5
    num_delete = 3
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      for _ in range(num_delete):
        program.random_delete(rng, namespace)
      mutated_code = core.normalize_code("""
      def f(a: Int, b: Int):
        c = (a - b)
        return c

      def _main_():
        return f(1, 1)
      """)
      string = program.to_string()
      self.assertEqual(core.normalize_code(string), mutated_code)
    t2 = time.time()
    logging.info('Deletion time: %d sec', (t2-t1) / (num_repeat * num_delete))

  def test_random_deletion_2(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      c = (a + b)
      c = (a - b)
      return c

    def _main_():
      return f(1, 1)
    """)
    num_repeat = 5
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      program.random_delete(rng, namespace)
      self.assertLen(program.body[0].body, 2)
    t2 = time.time()
    logging.info('Deletion time: %d sec', (t2-t1) / num_repeat)

  def test_random_deletion_not_affecting_conditional(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      if a < 2:
        c = b
      else:
        c = a
      c = (c + 1)
      return c
    """)
    code_after_deletion = core.normalize_code("""
    def f(a: Int, b: Int):
      if a < 2:
        c = b
      else:
        c = a
      return c
    """)
    expected_program_after_deletion = core.Program.parse(code_after_deletion)
    num_repeat = 5
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      program.random_delete(rng, namespace)
      self.assertLen(program.body[0].body, 2)
      self.assertEqual(program.to_string(),
                       expected_program_after_deletion.to_string())
    t2 = time.time()
    logging.info('Deletion time: %d sec', (t2 - t1) / num_repeat)

  def test_random_deletion_not_deleting_conditional(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      if a < 2:
        c = b
      else:
        c = a
      return c
    """)
    code_after_deletion = core.normalize_code("""
    def f(a: Int, b: Int):
      if a < 2:
        c = b
      else:
        c = a
      return c
    """)
    expected_program_after_deletion = core.Program.parse(code_after_deletion)
    num_repeat = 5
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      program.random_delete(rng, namespace)
      self.assertLen(program.body[0].body, 2)
      self.assertEqual(program.to_string(),
                       expected_program_after_deletion.to_string())
    t2 = time.time()
    logging.info('Deletion time: %d sec', (t2 - t1) / num_repeat)

  def test_random_insertion(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      c = (a + b)
      c = (a - b)
      return c

    def _main_():
      return f(1, 1)
    """)
    num_repeat = 3
    num_insert = 2
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      for _ in range(num_insert):
        program.random_insert(rng, namespace)
      self.assertLen(program.body[0].body, 3 + num_insert)
    t2 = time.time()
    logging.info('insertion time: %d sec', (t2-t1) / (num_repeat * num_insert))

  def test_random_argument_modification(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      c = (a - b)
      return c

    def _main_():
      return f(1, 1)
    """)
    num_repeat = 3
    num_modify = 2
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      for _ in range(num_modify):
        previous = program.to_string()
        program.random_modify(rng, namespace)
        new = program.to_string()
        self.assertNotEqual(previous, new)
      self.assertLen(program.body[0].body, 2)
    t2 = time.time()
    logging.info('insertion time: %d sec', (t2-t1) / (num_repeat * num_modify))

  def test_random_atom_modification(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      c = 1.0
      return c

    def _main_():
      return f(1, 1)
    """)
    num_repeat = 3
    num_modify = 2
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      for _ in range(num_modify):
        previous = program.to_string()
        program.random_modify(rng, namespace)
        new = program.to_string()
        self.assertNotEqual(previous, new)
      self.assertLen(program.body[0].body, 2)
    t2 = time.time()
    logging.info('insertion time: %d sec', (t2-t1) / (num_repeat * num_modify))

  def test_random_compare_op_modification(self):
    code = core.normalize_code("""
    def f(a: Int, b: Int):
      if a < 2:
        c = b
      else:
        c = a
      return c
    """)
    num_repeat = 3
    num_modify = 2
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      for _ in range(num_modify):
        conditional = program.body[0].body[0]
        old_compare_op = conditional.test.to_string()
        old_true_branch = core.block2str(conditional.true_branch)
        old_false_branch = core.block2str(conditional.false_branch)
        program.random_modify(rng, namespace)
        new_compare_op = conditional.test.to_string()
        new_true_branch = core.block2str(conditional.true_branch)
        new_false_branch = core.block2str(conditional.false_branch)
        self.assertNotEqual(old_compare_op, new_compare_op)
        self.assertEqual(old_true_branch, new_true_branch)
        self.assertEqual(old_false_branch, new_false_branch)
      self.assertLen(program.body[0].body, 2)
    t2 = time.time()
    logging.info('insertion time: %d sec',
                 (t2 - t1) / (num_repeat * num_modify))

  def test_basic_ops(self):
    code = core.normalize_code("""
    def f(a, b):
      c = (a + b)
      c = (a - b)
      c = (a * b)
      c = (a / b)
      c = (a and b)
      c = (a or b)
      c = (a == b)
      c = (a != b)
      c = (a > b)
      c = (a >= b)
      c = (a < b)
      c = (a <= b)
      return c

    def _main_():
      f(0, 1)
    """)
    namespace = core.Namespace()
    program = core.Program.parse(code)
    string = program.to_string()
    print(string)
    print(core.normalize_code(string))
    self.assertEqual(core.normalize_code(string), code)
    result, success = program.execute(namespace)
    print(result)
    self.assertTrue(success)

  def test_function(self):
    code = core.normalize_code("""
    def f(a):
      b = (a + 1)
      return b

    def _main_():
      return f(1)
    """)
    namespace = core.Namespace()
    program = core.Program.parse(code)
    string = program.to_string()
    self.assertEqual(core.normalize_code(string), code)
    result, success = program.execute(namespace)
    self.assertTrue(success)
    self.assertEqual(result, 2)
    fn = namespace['_main_']
    self.assertEqual(fn(), 2)
    fn = namespace['f']
    self.assertEqual(fn(2), 3)

  def test_function_assign_multiple_targets(self):
    code = core.normalize_code("""
    def f(a, b):
      c = (a * 2)
      d = (b + 1)
      return (c, d)

    def _main_():
      (c, d) = f(1, 2)
      e = (c + 1)
      f = (d - 1)
      return (e, f)
    """)
    namespace = core.Namespace()
    program = core.Program.parse(code)
    string = program.to_string()
    print(string)
    self.assertEqual(core.normalize_code(string), code)
    result, success = program.execute(namespace)
    self.assertTrue(success)
    self.assertEqual(result, [3, 2])
    fn = namespace['_main_']
    self.assertEqual(fn(), [3, 2])
    fn = namespace['f']
    self.assertEqual(fn(2, 1), [4, 2])

  def test_function_annotation(self):
    code = core.normalize_code("""
    def f(a: Int):
      b = (a + 1)
      return b

    def _main_():
      return f(1)
    """)
    namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
    program = core.Program.parse(code)
    string = program.to_string()
    self.assertEqual(core.normalize_code(string), code)
    result, success = program.execute(namespace)
    self.assertTrue(success)
    self.assertEqual(result, 2)
    fn = namespace['_main_']
    self.assertEqual(fn(), 2)
    fn = namespace['f']
    self.assertEqual(fn(2), 3)

  def test_conditional(self):
    code = core.normalize_code("""
    def f(a):
      b = a
      if (a > 1):
        b = (a + 100)
      else:
        if (a >= 0):
          b = (a + 10)
      return b

    def _main_():
      return ((f(2) + f(0)) + f(-2))
    """)
    namespace = core.Namespace()
    program = core.Program.parse(code)
    string = program.to_string()
    self.assertEqual(core.normalize_code(string), code)
    result, success = program.execute(namespace)
    print(result)
    self.assertTrue(success)
    self.assertEqual(result, 110)
    fn = namespace['_main_']
    self.assertEqual(fn(), 110)
    fn = namespace['f']
    self.assertEqual(fn(2), 102)

  def test_conditional_with_multiple_return_values(self):
    code = core.normalize_code("""
    def f1(a, b):
      return a + 1, b + 1

    def f2(a, b):
      return a - 1, b - 1

    def f(a, b):
      if (a > 1):
        c, d = f1(a, b)
      else:
        c, d = f2(a, b)
      return c, d

    def _main_():
      return f(2, 3)
    """)
    namespace = core.Namespace()
    program = core.Program.parse(code)
    result, success = program.execute(namespace)
    print(result)
    self.assertTrue(success)
    self.assertEqual(result, [3, 4])
    fn = namespace['_main_']
    self.assertEqual(fn(), [3, 4])
    fn = namespace['f']
    self.assertEqual(fn(1, 2), [0, 1])
    self.assertEqual(fn(2, 3), [3, 4])

  def test_for_loop(self):
    code = core.normalize_code("""
    def f():
      sum = 0
      for i in range(10):
        sum = (sum + i)
      return sum

    def _main_():
      x = 0
      for (i, j) in zip(range(3), range(10, 13)):
        x = ((i + j) + x)
      return (x + f())
    """)
    namespace = core.Namespace()
    program = core.Program.parse(code)
    string = program.to_string()
    self.assertEqual(core.normalize_code(string), code)
    result, success = program.execute(namespace)
    print(result)
    self.assertTrue(success)
    self.assertEqual(result, 81)
    fn = namespace['f']
    self.assertEqual(fn(), 45)

  @unittest.skip
  def test_recursive_funcall(self):
    code = core.normalize_code("""
    def f1(a):
      return f2(a-1)

    def f2(b):
      if b > 1:
        return f1(b)
      else:
        return b

    def _main_():
      return f1(10) + f1(0)
    """)
    namespace = core.Namespace()
    program = core.Program.parse(code)
    result, success = program.execute(namespace)
    self.assertTrue(success)
    self.assertEqual(result, 0)

  def test_namespace(self):
    parent_ns = core.Namespace()
    parent_ns['a'] = 1
    parent_ns['b'] = 1
    ns = parent_ns.child()
    ns['a'] = 2
    self.assertEqual(parent_ns['a'], 1)
    self.assertEqual(ns['a'], 2)
    self.assertIn('b', ns)

    ns_copy = ns.copy()
    self.assertIs(ns_copy.parent, ns.parent)
    self.assertIsNot(ns_copy.mapping, ns.mapping)
    self.assertEqual(ns_copy.mapping, ns.mapping)

  def test_abstract_execution(self):
    code = core.normalize_code("""
    def f(a):
      b = (a + 1)
      return b

    def _main_():
      return f(1)
    """)
    program = core.Program.parse(code)
    namespace = core.Namespace()
    result, success = program.execute(namespace, abstract=True)
    print(result)
    self.assertTrue(success)
    self.assertEqual(result.shape, ())
    self.assertEqual(result.dtype, jnp.int32)

  def test_jax_program_execution(self):
    ((train_data, train_labels,
      valid_data, valid_labels), params) = get_mock_setup()

    t1 = time.time()
    code = core.normalize_code("""
    def train1(params: Params, steps: Int, data: Data, labels: Labels, lr: Float):
      gradient = compute_grad(params, data, labels)
      params = (params - (gradient * lr))
      return params

    def _main_(init_params, train_data, train_labels, valid_data, valid_labels):
      lr = 0.1
      params = init_params
      loss = eval(params, valid_data, valid_labels)
      for i in range(2):
        params = train1(params, i, train_data, train_labels, lr)
        loss = eval(params, valid_data, valid_labels)
      return loss
    """)
    program = core.Program.parse(code)

    params_annotation = core.ExampleAnnotation(params)
    data_annotation = core.ExampleAnnotation(train_data, ignore_batch_axis=True)
    labels_annotation = core.ExampleAnnotation(
        train_labels, ignore_batch_axis=True)
    compute_grad_function = core.Function(
        compute_grad, 3,
        annotations=[params_annotation, data_annotation, labels_annotation])
    eval_function = core.Function(
        eval_model, 3,
        annotations=[params_annotation, data_annotation, labels_annotation])
    init_namespace = core.Namespace(
        dict(compute_grad=compute_grad_function,
             eval=eval_function,
             Params=params_annotation,
             Int=core.ExampleAnnotation(1),
             Data=data_annotation,
             Labels=labels_annotation,
             Float=core.ExampleAnnotation(1.0)))

    args = [params, train_data, train_labels, valid_data, valid_labels]
    loss, success = program.execute(
        namespace=init_namespace.copy(), args=args)
    t2 = time.time()
    logging.info('%d sec used in training.', t2-t1)
    self.assertTrue(success)
    self.assertLess(loss, 2.0)

  def test_jax_program_mutation(self):
    ((train_data, train_labels,
      valid_data, valid_labels), params) = get_mock_setup()

    code = core.normalize_code("""
    def train1(params: Params, steps: Int, data: Data, labels: Labels, lr: Float):
      gradient = compute_grad(params, data, labels)
      params = (params - (gradient * lr))
      return params

    def _main_(init_params, train_data, train_labels, valid_data, valid_labels):
      lr = 0.1
      params = init_params
      loss = eval(params, valid_data, valid_labels)
      for i in range(2):
        params = train1(params, i, train_data, train_labels, lr)
        loss = eval(params, valid_data, valid_labels)
      return loss
    """)
    program = core.Program.parse(code)

    params_annotation = core.ExampleAnnotation(params)
    data_annotation = core.ExampleAnnotation(
        train_data, ignore_batch_axis=True)
    labels_annotation = core.ExampleAnnotation(
        train_labels, ignore_batch_axis=True)
    compute_grad_function = core.Function(
        compute_grad, 3,
        annotations=[params_annotation, data_annotation, labels_annotation])
    eval_function = core.Function(
        eval_model, 3,
        annotations=[params_annotation, data_annotation, labels_annotation])
    init_namespace = core.Namespace(
        dict(compute_grad=compute_grad_function,
             eval=eval_function,
             Params=params_annotation,
             Int=core.ExampleAnnotation(1),
             Data=data_annotation,
             Labels=labels_annotation,
             Float=core.ExampleAnnotation(1.0)))

    args = [params, train_data, train_labels, valid_data, valid_labels]
    num_repeat = 3
    num_insert = 2
    t1 = time.time()
    for i in range(num_repeat):
      namespace = init_namespace.copy()
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      for _ in range(num_insert):
        program.random_insert(rng, namespace, args=args)
        print('=' * 20)
        print(program.to_string())
        print('=' * 20)
      self.assertLen(program.body[0].body, 3 + num_insert)
    t2 = time.time()
    logging.info('insertion time: %d sec', (t2-t1) / (num_repeat * num_insert))

  def test_jax_functionality(self):
    t1 = time.time()
    (train_data, train_labels,
     valid_data, valid_labels) = get_mock_regression_dataset(1)
    t2 = time.time()
    logging.info('%d sec used creating dataset.', t2-t1)
    lr = 0.1

    t1 = time.time()
    model = SimpleNN()
    params = model.init(
        jax.random.PRNGKey(0), train_data.shape)
    t2 = time.time()
    logging.info('%d sec used in initialization.', t2-t1)

    t1 = time.time()
    for _ in range(2):
      grad = compute_grad(params, train_data, train_labels)
      params = core.apply_binary_op(
          core.sub, params, core.apply_binary_op(core.mult, lr, grad))
      loss = eval_model(params, train_data, train_labels)
    t2 = time.time()
    logging.info('%d sec used in training.', t2-t1)

    loss = eval_model(params, valid_data, valid_labels)
    self.assertLess(loss, 2.0)

  def test_remove_redundancy(self):
    code = _CODE_1
    program = core.Program.parse(code)
    program.simplify()
    simplified_code = program.to_string()

    ground_truth_code = _SIMPLIFIED_CODE_1
    ground_truth_code = core.Program.parse(ground_truth_code).to_string()
    self.assertEqual(simplified_code, ground_truth_code)

  def test_remove_redundancy_in_branches(self):
    code = core.normalize_code("""
    def f(a):
      if a < 1:
        b = 1
        b = a * 2
      else:
        b = a * 0.5
        b = a - 1
      return b
    """)
    program = core.Program.parse(code)
    program.simplify()
    simplified_code = program.to_string()

    ground_truth_code = core.normalize_code("""
    def f(a):
      if a < 1:
        b = a * 2
      else:
        b = a - 1
      return b
    """)
    ground_truth_code = core.Program.parse(ground_truth_code).to_string()
    self.assertEqual(simplified_code, ground_truth_code)

  def test_program_hash(self):
    program_1 = core.Program.parse(_CODE_1)
    hash_1, _ = program_1.compute_hash()

    program_2 = core.Program.parse(_SIMPLIFIED_CODE_1)
    hash_2, _ = program_2.compute_hash()

    self.assertEqual(hash_1, hash_2)

    code_1 = core.normalize_code("""
    def f1(a, b):
      c = a + b
      return c

    def f2(a, b):
      c = a + b
      return c

    def f3(a, b):
      d = a + a
      return d

    def f4(a, b):
      c = f1(a, b)
      d = c + a
      return d, b

    def f5(a, b):
      c = f1(a, b)
      d = c + a
      return d

    def f6(a, b):
      c = f4(a, b)
      c = c + 1
      return c
    """)
    program_1 = core.Program.parse(code_1)
    program_hash_1, hash_dict_1 = program_1.compute_hash()

    code_2 = core.normalize_code("""
    def f1(a, b):
      c = a - b
      return c

    def f2(a, b):
      c = a + b
      d = 1 + c
      return c

    def f3(c, d):
      e = c + c
      return e

    def f4(a, b):
      c = f2(a, b)
      e = a * b
      d = c + a
      return d, b

    def f5(a, b):
      c = f1(a, b)
      d = c + a
      return d

    def f6(a, b):
      c = f4(a, b)
      c = c + 1
      return c
    """)
    program_2 = core.Program.parse(code_2)
    program_hash_2, hash_dict_2 = program_2.compute_hash()

    self.assertNotEqual(program_hash_1, program_hash_2)

    self.assertEqual(hash_dict_1['f2'], hash_dict_2['f2'])
    self.assertNotEqual(hash_dict_1['f1'], hash_dict_2['f1'])
    # Check that the name of the variables doesn't matter in hash.
    self.assertEqual(hash_dict_1['f3'], hash_dict_2['f3'])
    # Check that the hash considers the equivalence of the dependency,
    # in this case the call of `f1` and `f2` in the definition of `f4`.
    self.assertEqual(hash_dict_1['f4'], hash_dict_2['f4'])
    self.assertNotEqual(hash_dict_1['f5'], hash_dict_2['f5'])
    # Check nested function calls because f6 calls f4.
    self.assertEqual(hash_dict_1['f6'], hash_dict_2['f6'])

  def test_program_hash_with_conditional_statement(self):
    code_with_branches = core.normalize_code("""
    def f(a):
      if a < 1:
        b = a * 2
        b = b - 1
      else:
        b = a - 1
        b = a * b
      return b
    """)
    program_with_branches = core.Program.parse(code_with_branches)
    hash_with_branches, _ = program_with_branches.compute_hash()

    code_without_branch_1 = core.normalize_code("""
    def f(a):
      b = a * 2
      b = b - 1
      b = a - 1
      b = a * b
      return b
    """)
    program_without_branch_1 = core.Program.parse(code_without_branch_1)
    hash_without_branch_1, _ = program_without_branch_1.compute_hash()

    self.assertNotEqual(hash_with_branches, hash_without_branch_1)

    code_without_branch_2 = core.normalize_code("""
    def f(a):
      b = a * 2
      b = b - 1
      return b
    """)
    program_without_branch_2 = core.Program.parse(code_without_branch_2)
    hash_without_branch_2, _ = program_without_branch_2.compute_hash()

    self.assertNotEqual(hash_with_branches, hash_without_branch_2)

    code_without_branch_3 = core.normalize_code("""
    def f(a):
      b = a - 1
      b = a * b
      return b
    """)
    program_without_branch_3 = core.Program.parse(code_without_branch_3)
    hash_without_branch_3, _ = program_without_branch_3.compute_hash()

    self.assertNotEqual(hash_with_branches, hash_without_branch_3)

  def test_nested_abstract_execution(self):
    code = core.normalize_code("""
    def f1(a):
      b = (a + 1)
      return b

    def f2(a):
      c = (a * 2)
      return c

    def f3(a):
      b = f1(a)
      b = f2(b)
      return b

    def _main_():
      return f3(1)
    """)
    program = core.Program.parse(code)
    namespace = core.Namespace()
    result, success = program.execute(namespace, abstract=True)
    print(result)
    self.assertTrue(success)
    self.assertEqual(result.shape, ())
    self.assertEqual(result.dtype, jnp.int32)

  def test_nested_funcall_random_insertion(self):
    code = core.normalize_code("""
    def f1(a: Int):
      b = (a + 1)
      b = (a - 1)
      return b

    def f2(a: Int):
      c = (a * 2)
      c = (a * 4)
      return c

    def f3(a: Int):
      b = f1(a)
      b = f2(b)
      c = b + 1
      return b

    def _main_():
      return f3(1)
    """)
    num_repeat = 3
    num_insert = 2
    t1 = time.time()
    for i in range(num_repeat):
      namespace = core.Namespace(dict(Int=core.ExampleAnnotation(1)))
      rng = jax.random.PRNGKey(i)
      program = core.Program.parse(code)
      for _ in range(num_insert):
        program.random_insert(rng, namespace)
      self.assertEqual(sum([len(fundef.body) for fundef in program.body]),
                       11 + num_insert)
    t2 = time.time()
    logging.info('insertion time: %d sec', (t2-t1) / (num_repeat * num_insert))

  def test_nested_funcall_mutation(self):
    code = core.normalize_code("""
    def f1(a: Int):
      b = (a + 1)
      b = (a - 1)
      return b

    def f2(a: Int):
      c = (a * 2)
      c = (a * 4)
      return c

    def f3(a: Int):
      b = f1(a)
      b = f2(b)
      c = b + 1
      return b

    def _main_():
      return f3(1)
    """)
    program = core.Program.parse(code)
    namespace = core.Namespace()
    namespace['Int'] = core.ExampleAnnotation(1)
    for i in range(3):
      rng = jax.random.PRNGKey(i)
      success = program.random_insert(rng, namespace)
      print(f'insert {i}:')
      print(program.to_string())
      self.assertTrue(success)
      success = program.random_delete(rng, namespace)
      print(f'delete {i}:')
      print(program.to_string())
      self.assertTrue(success)
      success = program.random_modify(rng, namespace)
      print(f'modify {i}:')
      print(program.to_string())
      self.assertTrue(success)

  def test_collect_scalars(self):
    code = core.normalize_code("""
    def f1(a: Int):
      b = (a + 1)
      b = (a - 2)
      return b

    def f2(a: Int, b: Int):
      c = (a * 3)
      c = (b * 4)
      return c

    def f3(a: Int):
      b = f1(5)
      b = f2(6, 7)
      c = 8 + 9
      return b
    """)
    program = core.Program.parse(code)
    hparams = program.collect_hparams()
    self.assertLen(hparams, 9)

  def test_collect_scalars_with_conditional_statement(self):
    code = core.normalize_code("""
    def f(a):
      if a < 2:
        b = a + 1
      else:
        b = a - 1
      return b
    """)
    program = core.Program.parse(code)
    hparams = program.collect_hparams()
    self.assertLen(hparams, 3)

  def test_hparam_tuning(self):
    code = core.normalize_code("""
    def f1(a: Int):
      b = (a + 1)
      b = (a - 2)
      return b

    def f2(a: Int, b: Int):
      c = (a * 3)
      c = (b * 4)
      return c

    def f3(a: Int):
      b = f1(5)
      b = f2(6, 7)
      c = 8 + 9
      return b
    """)
    program = core.Program.parse(code)
    rng = jax.random.PRNGKey(0)
    hparams = program.collect_hparams()
    old_values = [float(h.value) for h in hparams]
    for _ in range(500):
      rng, sub_rng = jax.random.split(rng)
      program.hparam_tune(sub_rng)
    new_values = [float(h.value) for h in hparams]
    for v1, v2 in zip(old_values, new_values):
      self.assertNotEqual(v1, v2)

  def test_hparam_tuning_with_conditional_statement(self):
    code = core.normalize_code("""
    def f(a):
      if a < 2:
        b = a + 1
      else:
        b = a - 1
      return b
    """)
    program = core.Program.parse(code)
    rng = jax.random.PRNGKey(0)
    hparams = program.collect_hparams()
    old_values = [float(h.value) for h in hparams]
    for _ in range(500):
      rng, sub_rng = jax.random.split(rng)
      program.hparam_tune(sub_rng)
    new_values = [float(h.value) for h in hparams]
    for v1, v2 in zip(old_values, new_values):
      self.assertNotEqual(v1, v2)

  def test_export_function(self):
    def hero_compile(program_string):
      program = core.Program.parse(program_string)
      namespace = core.Namespace()
      namespace.ignore_arg_annotations = True
      namespace.update(fn_lib.get_math_fns())
      _, success = program.execute(namespace)
      if not success:
        raise ValueError('The hero compilation of the given program failed.')
      train_fn = namespace['train']
      return train_fn

    train_fn = hero_compile(_ADAM_STRING_ARG_ANNOTATION)
    params = {'w1': jnp.array([1.0]), 'w2': jnp.array([2.0])}
    m = {'w1': jnp.array([1.0]), 'w2': jnp.array([2.0])}
    v = {'w1': jnp.array([1.0]), 'w2': jnp.array([2.0])}
    g = {'w1': jnp.array([1.0]), 'w2': jnp.array([2.0])}
    update, new_m, new_v = train_fn(params, m, v, g)
    print(update)
    print(new_m)
    print(new_v)


if __name__ == '__main__':
  unittest.main()
