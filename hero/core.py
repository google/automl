"""The core functionality for supporting Hero search space.

1. Parse a program from string.
2. Execute a program in a given namespace, including both concrete and 
   abstract execution.
3. Mutate a program.
4. Turn a program to string for visualization and serialization.

The main classes are:
1. Program: the main component that is parsed from python code, and
            can execute or mutate in a given namespace.
2. Namespace: a mapping between names and values just like in most programming
              languages. It defines an environment that a program executes
              and mutates in. Note that a program will mutate during the
              search, but not the namespace. The namespace is initially
              specified by the user and can be changed during the execution of
              a program.
3. Expression: a program is made of expressions of various types, for example,
               function calls, literal values, conditionals, loops, function
               definitions, etc. Each Expression subclass should define how
               it will be constructed, executed, visualized and how it can be
               changed during mutations.
"""

import abc
import ast
import collections
import hashlib
import itertools
import time
import typing as t

from absl import logging
import jax
import jax.numpy as jnp

# Temporary symbols used to handle conditional statements.
CONDITIONAL = 'Conditional'

# Special symbols.
NONE = 'None'
MAIN = '_main_'

# Arithmetics.
ADD = '+'
SUB = '-'
MULT = '*'
DIV = '/'

# Comparison.
EQ = '=='
NEQ = '!='
LT = '<'
LTE = '<='
GT = '>'
GTE = '>='

# Bool ops.
AND = 'and'
OR = 'or'

# Unary ops.
INVERT = '~'
NOT = 'not'
UADD = '+'
USUB = '-'

# Primitive python functions.
ZIP = 'zip'
RANGE = 'range'

MAX = 'max'
MIN = 'min'

MAX_FLOAT = 1.0e10


# All the exception that will be captured during the execution.
def get_all_exceptions():
  return (UnknownVarError, TypeError, ValueError, IndexError,
          AttributeError, KeyError, RuntimeError)


class Program(object):
  """Program that can be executed and mutated given a namespace."""

  @classmethod
  def parse(cls, code: str):
    tree = ast.parse(code)
    return cls.transform(tree)

  @classmethod
  def transform(cls, tree: ast.Module):
    exps = []
    for node in tree.body:
      exps.append(transform_ast(node))
    return cls(exps)

  def __init__(self, exps):
    self.body = exps[:]

  @property
  def length(self):
    n = 0
    for exp in self.body:
      if isinstance(exp, FunDef):
        n += len(exp.body)
    return n

  def to_string(self):
    return '\n' + '\n\n'.join([exp.to_string() for exp in self.body]) + '\n'

  def copy(self):
    return self.parse(self.to_string())

  def execute(self, namespace, args=None, abstract=False):
    """Execute the program under the given namespace."""
    if abstract:
      namespace = namespace.to_abstract()
    result = None
    try:
      for exp in self.body:
        result = exp.execute(namespace)
      if MAIN in namespace:
        if args is None:
          result, success = namespace.get_function(MAIN)()
        else:
          result, success = namespace.get_function(MAIN)(*args)
      else:
        return result, True
    except get_all_exceptions() as e:
      return e, False
    return result, success

  def mutate(
      self, rng, namespace, args=None,
      insert_weight=1.0, delete_weight=1.0, modify_weight=1.0,
      max_program_len=-1, verbose=1):
    """Mutate the program using insertion, deletion or modification."""
    logging.info('=' * 50)
    logging.info(
        'program before mutation: \n%s', self.to_string())
    logging.info('=' * 50)
    # If the length already reached the maximum, then only allow modification
    # and deletion.
    if max_program_len > -1 and self.length >= max_program_len:
      insert_weight = 0.0
    rng, sub_rng = jax.random.split(rng)
    total_weight = float(insert_weight + delete_weight + modify_weight)
    if total_weight <= 0.0:
      raise ValueError('The sum of insert_weight, delete_weight, modify_weight'
                       ' should not be zero. Note insert_weight is '
                       'automatically set to zero when the '
                       'program length >= max_program_len and '
                       'max_program_len > -1.')
    p1 = delete_weight / total_weight
    p2 = (delete_weight + insert_weight) / total_weight
    if verbose: t1 = time.time()
    draw = jax.random.uniform(sub_rng)
    if draw < p1:
      action = 'deletion'
      success = self.random_delete(rng, namespace, args=args, verbose=verbose)
    elif draw < p2:
      action = 'insertion'
      success = self.random_insert(rng, namespace, args=args, verbose=verbose)
    else:
      action = 'modification'
      success = self.random_modify(rng, namespace, args=args, verbose=verbose)
    if verbose:
      if success:
        logging.info('%s succeeded!', action)
      else:
        logging.info('%s failed!', action)
      t2 = time.time()
      logging.info('%d secs used in %s!', t2 - t1, action)
    return success

  def hparam_tune(self, rng):
    """Mutate the scalars in the program to perform hyperparameter tuning."""
    hparams = self.collect_hparams()
    if hparams:
      ind = jax.random.randint(
          rng, (), minval=0, maxval=len(hparams))
      old_value = hparams[ind].value  # pytype: disable=unsupported-operands  # jax-types
      rng, sub_rng = jax.random.split(rng)
      hparams[ind].mutate(sub_rng)  # pytype: disable=unsupported-operands  # jax-types
      new_value = hparams[ind].value  # pytype: disable=unsupported-operands  # jax-types
      print(f'Scalar mutated from {old_value} to {new_value}.')
    else:
      print('No hard coded scalar available in the program!')

  def collect_hparams(self):
    hparams = []
    for exp in self.body:
      hparams += collect_scalars(exp)
    return hparams

  def get_fundef(self, name):
    fundefs = [exp for exp in self.body
               if isinstance(exp, FunDef) and exp.fn_name == name]
    if len(fundefs) != 1:
      raise ValueError(
          f'There should be exactly one function with the name {name}, '
          f'but {len(fundefs)} are found!')
    return fundefs[0]

  def mutate_function(self, name, rng, namespace,
                      insert_weight=1.0, delete_weight=1.0,
                      modify_weight=1.0):
    """Mutate the given function."""
    fundef = self.get_fundef(name)
    rng, sub_rng = jax.random.split(rng)
    total_weight = float(insert_weight + delete_weight + modify_weight)
    p1 = delete_weight / total_weight
    p2 = (delete_weight + insert_weight) / total_weight
    draw = jax.random.uniform(sub_rng)
    if draw < p1:
      for _ in fundef.try_next_deletion(sub_rng, namespace):
        _, success = self.execute(namespace.copy(), abstract=True)
        if success:
          return True
      return False
    elif draw < p2:
      for _ in fundef.try_next_insertion(sub_rng, namespace):
        _, success = self.execute(namespace.copy(), abstract=True)
        if success:
          return True
      return False
    else:
      rng, sub_rng = jax.random.split(rng)
      success = fundef.try_modification(sub_rng, namespace)
      _, success = self.execute(namespace.copy(), abstract=True)
      return success

  def compute_function_hash(self, name):
    _, hash_dict = self.compute_hash()
    return hash_dict[name]

  def mutate_function_until(self, name, rng, namespace, max_num_trials=1000,
                            insert_weight=1.0, delete_weight=1.0,
                            modify_weight=1.0):
    """Mutate the given function until its hash is different."""
    rng, sub_rng = jax.random.split(rng)
    old_hash = self.compute_function_hash(name)
    print(f'old function: {self.get_fundef(name).to_string()}')
    for _ in range(max_num_trials):
      success = self.mutate_function(name, sub_rng, namespace,
                                     insert_weight=insert_weight,
                                     delete_weight=delete_weight,
                                     modify_weight=modify_weight)
      if success:
        new_hash = self.compute_function_hash(name)
        if new_hash != old_hash:
          print(f'new function: {self.get_fundef(name).to_string()}')
          print(f'old hash: {old_hash}\nnew_hash: {new_hash}')
          return True
      else:
        return False
    return False

  def random_insert(self, rng, namespace, args=None, verbose=1):
    """Randomly insert an expression into the program."""
    fundefs = [exp for exp in self.body if isinstance(exp, FunDef)]
    rng, sub_rng = jax.random.split(rng)
    fundef_order = jax.random.permutation(
        sub_rng, jnp.arange(len(fundefs)))
    for i in fundef_order:
      fundef = fundefs[i]
      if fundef.fn_name == MAIN:
        # Don't mutate the main function since it usually
        # defines the fitness computation.
        continue
      rng, sub_rng = jax.random.split(rng)
      tmp_namespace = namespace.copy()
      for exp in self.body:
        if exp == fundef:
          break
        else:
          exp.execute(tmp_namespace)
      for _ in fundef.try_next_insertion(sub_rng, tmp_namespace):
        result, success = self.execute(
            namespace.copy(), args=args, abstract=True)
        if verbose:
          logging.info('=' * 50)
          logging.info(
              'attempted program after insertion: %s', self.to_string())
          logging.info(
              'attempted program result: %s', result)
          logging.info('=' * 50)
        if success:
          return True
    return False

  def random_delete(self, rng, namespace, args=None, verbose=1):
    """Randomly delete an expression from the program."""
    fundefs = [exp for exp in self.body if isinstance(exp, FunDef)]
    rng, sub_rng = jax.random.split(rng)
    fundef_order = jax.random.permutation(
        sub_rng, jnp.arange(len(fundefs)))
    for i in fundef_order:
      fundef = fundefs[i]
      if fundef.fn_name == MAIN or len(fundef.body) <= 1:
        # Don't mutate the main function since it usually
        # defines the fitness computation.
        continue
      rng, sub_rng = jax.random.split(rng)
      for _ in fundef.try_next_deletion(sub_rng, namespace):
        result, success = self.execute(
            namespace.copy(), args=args, abstract=True)
        if verbose:
          logging.info('=' * 50)
          logging.info(
              'attempted program after deletion: %s', self.to_string())
          logging.info('attempted program result: %s', result)
          logging.info('=' * 50)
        if success:
          return True
    return False

  def random_modify(self, rng, namespace, args=None, verbose=1):
    """Randomly modify an expression in the program."""
    fundefs = [exp for exp in self.body if isinstance(exp, FunDef)]
    rng, sub_rng = jax.random.split(rng)
    fundef_order = jax.random.permutation(
        sub_rng, jnp.arange(len(fundefs)))
    for i in fundef_order:
      fundef = fundefs[i]
      new_namespace = namespace.copy()
      for exp in self.body:
        if exp == fundef:
          break
        else:
          exp.execute(new_namespace)
      if fundef.fn_name == MAIN:
        # Don't mutate the main function since it is usually used to
        # defines the fitness computation.
        continue
      rng, sub_rng = jax.random.split(rng)
      success = fundef.try_modification(sub_rng, new_namespace)
      result, success = self.execute(
          namespace.copy(), args=args, abstract=True)
      if verbose:
        logging.info('=' * 50)
        logging.info(
            'attempted program after modification: %s', self.to_string())
        logging.info('attempted program result: %s', result)
        logging.info('=' * 50)
      return success

  def simplify(self):
    for exp in self.body:
      if isinstance(exp, FunDef):
        exp.simplify()

  def simplified(self):
    new = self.copy()
    new.simplify()
    return new

  def compute_hash(self):
    # Have a hash tracker as namespace to consider the dependency
    # between functions when computing the hash.
    hash_tracker = HashTracker()
    for exp in self.body:
      exp.execute(hash_tracker)
    return str2hash(str(sorted(hash_tracker.items()))), hash_tracker.to_dict()


def collect_scalars(exp):
  """Collect the hard coded scalars (Atom with numerical value)."""
  if isinstance(exp, Atom):
    if isinstance(exp.value, jnp.ndarray):
      return [exp]
  elif isinstance(exp, FunCall):
    scalars = []
    for a in exp.args:
      scalars += collect_scalars(a)
    return scalars
  elif isinstance(exp, FunDef):
    scalars = []
    for e in exp.body:
      scalars += collect_scalars(e)
    return scalars
  elif isinstance(exp, Assignment):
    return collect_scalars(exp.value)
  elif isinstance(exp, CompareOp):
    return collect_scalars(exp.left) + collect_scalars(exp.right)
  elif isinstance(exp, Conditional):
    scalars = collect_scalars(exp.test)
    for e in exp.true_branch:
      scalars += collect_scalars(e)
    for e in exp.false_branch:
      scalars += collect_scalars(e)
    return scalars
  else:
    return []


def candidate_insertions(rng, exp_list, namespace):
  """Iterate through possible insertions."""
  rng, sub_rng = jax.random.split(rng)
  exp_order = jax.random.permutation(sub_rng, jnp.arange(len(exp_list)+1))
  logging.info('insert attempt order: %s', exp_order)
  for i in exp_order:
    tmp_namespace = namespace.copy()
    for exp in exp_list[:i]:
      exp.execute(tmp_namespace)
    rng, sub_rng = jax.random.split(rng)
    for new_exp in candidate_new_exps(sub_rng, tmp_namespace):
      new_exp_list = exp_list[:i] + [new_exp] + exp_list[i:]
      logging.info('=' * 50)
      logging.info(
          'attempt to insert expression: %s.', new_exp.to_string())
      logging.info('=' * 50)
      yield new_exp_list


def candidate_deletions(rng, exp_list, namespace):
  """Iterate through the possible deletions."""
  del namespace
  rng, sub_rng = jax.random.split(rng)
  exp_order = jax.random.permutation(sub_rng, jnp.arange(len(exp_list)))
  logging.info('delete attempt order: %s', exp_order)
  for i in exp_order:
    if isinstance(exp_list[i], Conditional):
      if len(exp_order) == 1:
        # Don't mutate anything if the conditional statement is the only
        # statement left.
        return exp_list
      else:
        continue
    new_exp_list = exp_list[:i] + exp_list[i+1:]
    logging.info('=' * 50)
    logging.info(
        'attempt to delete expression: %s.', exp_list[i].to_string())
    logging.info('=' * 50)
    yield new_exp_list


def try_modification(rng, exp_list, namespace):
  """Randomly apply a modification."""
  rng, sub_rng = jax.random.split(rng)
  if not exp_list:
    return False
  exp_order = jax.random.permutation(sub_rng, len(exp_list))
  logging.info('modification attempt order: %s.', exp_order)
  for i in exp_order:
    tmp_namespace = namespace.copy()
    for exp in exp_list[:i]:
      exp.execute(tmp_namespace)
    rng, sub_rng = jax.random.split(rng)
    if exp_list[i].mutate(sub_rng, tmp_namespace):
      return True
  return False


def candidate_new_exps(rng, namespace):
  # Currently only supports creating new assignment expressions.
  yield from Assignment.random_generate(rng, namespace)


class Expression(metaclass=abc.ABCMeta):
  """Abstract class for expressions in the program."""

  @abc.abstractmethod
  def to_string(self, indent_level=0):
    raise NotImplementedError

  @abc.abstractmethod
  def execute(self, namespace):
    raise NotImplementedError

  def simplify(self, namespace):
    pass


class Atom(Expression):
  """An expression that represents a literal value."""

  @classmethod
  def random_generate(cls, rng):
    value = jax.random.normal(rng)
    return Atom(value)

  def __init__(self, value):
    if isinstance(value, (int, float)):
      self.value = jnp.array(value)
    else:
      self.value = value

  def to_string(self, indent_level=0):
    if isinstance(self.value, str):
      result = "'{}'".format(self.value)
    else:
      # Check whether it is a scalar and use scientific notation.
      if self.value.shape and self.value.dtype == jnp.float32:
        result = '{:.2e}'.format(self.value)
      else:
        result = '{}'.format(self.value)
    return apply_indent(result, indent_level)

  def execute(self, namespace):
    if isinstance(namespace, DependencyTracker):
      return set()
    elif isinstance(namespace, HashTracker):
      return Hash(self.to_string())
    else:
      return self.value

  def mutate(self, rng, namespace=None):
    del namespace
    multiplier = jnp.power(2, jax.random.normal(rng))
    self.value *= multiplier
    # Bound the mutated value to avoid infinity issues.
    self.value = jnp.array(max(-MAX_FLOAT, min(MAX_FLOAT, self.value)))
    return True


class Symbol(object):
  """A symbol that can be a variable name or a keyword."""

  def __init__(self, name):
    self.name = name

  def to_string(self, indent_level=0):
    return apply_indent(self.name, indent_level)

  def execute(self, namespace):
    if isinstance(namespace, HashTracker):
      if self.name in namespace:
        return namespace[self.name]
      else:
        # If symbol not in namespace then assume it is predefined,
        # and use the name to compute hash.
        return Hash(f'symbol_{self.name}')
    else:
      return namespace[self.name]

  def mutate(self, rng, namespace=None):
    # Symbol objects doesn't support mutations.
    del rng
    del namespace
    return True


def random_iter(rng, seq):
  if not seq:
    return []
  order = jax.random.permutation(rng, len(seq))
  for i in order:
    yield seq[i]


class FunCall(Expression):
  """An expression that calls a function."""

  @classmethod
  def transform(cls, node: ast.Call):
    function = transform_ast(node.func)
    args = [transform_ast(a) for a in node.args]
    return cls(function, args)

  @classmethod
  def transform_binop(cls, node: ast.BinOp):
    left = transform_ast(node.left)
    right = transform_ast(node.right)
    op_map = {ast.Add: '+',
              ast.Sub: '-',
              ast.Mult: '*',
              ast.Div: '/'}
    op = op_map[type(node.op)]
    return cls(Symbol(op), [left, right])

  @classmethod
  def random_generate(cls, rng, namespace, allow_new_atom=True):
    """Randomly generate a function call expression."""
    variables = namespace.items()
    functions = [x for x in variables if isinstance(x[-1], Function)]
    rng, sub_rng = jax.random.split(rng)
    fn_order = jax.random.permutation(
        sub_rng, jnp.arange(len(functions)))
    for i in fn_order:
      fname, f = functions[i]
      logging.info('Attempt to insert a call to %s.', fname)
      all_valid_args = f.collect_valid_args(
          namespace, allow_new_atom=allow_new_atom)
      rng, sub_rng = jax.random.split(rng)
      for args in random_iter(sub_rng, all_valid_args):
        yield FunCall(Symbol(fname), args)

  def __init__(self, function, args):
    self.function = function
    self.args = args

  def execute(self, namespace):
    func = self.function.execute(namespace)
    args = [a.execute(namespace) for a in self.args]
    if isinstance(namespace, DependencyTracker):
      dependency = set()
      if not isinstance(func, set):
        print('namespace:')
        for k, v in namespace.items():
          print(f'{k}: {v}')
        raise ValueError(f'Func is not set: {func}')
      dependency = dependency.union(func)
      for a in args:
        assert isinstance(a, set)
        dependency = dependency.union(a)
      return dependency
    elif isinstance(namespace, HashTracker):
      return func.combine(args)
    else:
      return func(*args)

  def to_string(self, indent_level=0):
    fn_name = self.function.to_string()
    if fn_name in [ADD, SUB, MULT, DIV]:
      result = '({} {} {})'.format(
          self.args[0].to_string(),
          self.function.to_string(),
          self.args[1].to_string())
    else:
      args = items2str(self.args)
      result = '{}({})'.format(self.function.to_string(), args)
    return apply_indent(result, indent_level)

  def mutate(self, rng, namespace):
    """Randomly change the function call expression."""
    if not self.args:
      return False
    else:
      rng, sub_rng = jax.random.split(rng)
      order = jax.random.permutation(sub_rng, len(self.args))
      for i in order:
        arg = self.args[i]
        if isinstance(arg, Symbol):
          replacement = random_replace_symbol(rng, arg, namespace)
          if replacement:
            self.args[i] = replacement
            return True
        elif isinstance(arg, Atom):
          arg.mutate(rng)
          return True
      return False


def random_replace_symbol(rng, symbol, namespace):
  """Randomly select a symbol from the namespace with the same type/shape."""
  valid_replacements = []
  value = namespace[symbol.name]
  for k, v in namespace.items():
    if (k != symbol.name) and match(v, value):
      valid_replacements.append(k)
  if valid_replacements:
    selected_id = jax.random.randint(
        rng, (), minval=0, maxval=len(valid_replacements))
    return Symbol(valid_replacements[selected_id])  # pytype: disable=unsupported-operands  # jax-types
  else:
    return None


def apply_indent(string, indent_level):
  if indent_level:
    return '  ' * indent_level + string
  else:
    return string


def same_shape(shape1, shape2):
  flat_a, tree_def_a = jax.tree_flatten(shape1)
  flat_b, tree_def_b = jax.tree_flatten(shape2)
  return ((tree_def_a == tree_def_b) and
          all(map(lambda a, b: a == b, flat_a, flat_b)))


def same_dtype(dtype1, dtype2):
  flat_a, tree_def_a = jax.tree_flatten(dtype1)
  flat_b, tree_def_b = jax.tree_flatten(dtype2)
  return ((tree_def_a == tree_def_b) and
          all(map(simple_same_dtype, flat_a, flat_b)))


def simple_same_dtype(dtype1, dtype2):
  """Check whether the two dtype are the same."""
  floats = [jnp.float32, jnp.float64]
  ints = [jnp.int32, jnp.int64]
  uints = [jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]
  numbers = floats + ints + uints
  if dtype1 not in numbers:
    raise ValueError('Unknown dtype: {}'.format(dtype1))
  if dtype2 not in numbers:
    raise ValueError('Unknown dtype: {}'.format(dtype2))
  if dtype1 in floats:
    return dtype2 in floats
  elif dtype1 in ints:
    return dtype2 in ints
  elif dtype1 in uints:
    return dtype2 in uints
  else:
    raise ValueError('Unknown dtype: {}'.format(dtype1))


def remove_axis(shape, axis=0):
  return jax.tree_map(lambda x: x[:axis] + x[axis:], shape)


class Annotation(metaclass=abc.ABCMeta):
  """An annotation of a function argument that is generated from an example.

  To make it easy to specify the signature of a function, we allow annotating
  the argument with a syntax similar to the Python type annotation. However, the
  annotation should be an Annotation object instead of a type object here.
  For example, to specify a function that takes in an integer and a float as
  argument, we can specify:
  ```
    def f(a: Int, b: Float):
      return a + b
  ```
  where `Int` and `Float` are two Annotation objects defined by the user, which
  checks whether the arguments are of the right type.
  """

  @abc.abstractmethod
  def check(self, value):
    """Check whether a value fits the annotation."""
    raise NotImplementedError


class ExampleAnnotation(Annotation):
  """An Annotation that is generated from an example value.

  An Annotation object that is created from a given example value. For example,
  to check whether an argument is of the type Int, we can create
  an example annotation like
  ```
  Int = ExampleAnnotation(1)
  ```
  or to check whether an argument is a 3x4 matrix of float32, we can create
  ```
  Data = ExampleAnnotation(jnp.zeros([3, 4], dtype=jnp.float32))
  ```
  Sometimes we don't care about the first axis which is the batch axis, and only
  want to check the other axes, then we can set `ignore_batch_axis` to True.

  This is especially handy when we want to check for a complex structure of
  arrays. For example, the parameters of a neural net might be represented as:
  ```
    params = {'layer1_weight': XXX
              'layer1_bias': XXX
              ...}
  ```
  We can create an annotation that checks for the same structure and shape/dtype
  by `ExampleAnnotation(params)`.
  """

  def __init__(self, value, ignore_batch_axis=False):
    if isinstance(value, (int, float)):
      self.value = jnp.array(value)
    else:
      self.value = value
    self.numeric = is_numeric(value)
    self.ignore_batch_axis = ignore_batch_axis

  def check(self, value):
    """Check whether a given value fits the annotation."""
    if self.numeric:
      return (is_numeric(value) and
              same_shape(
                  get_shape(value,
                            ignore_batch_axis=self.ignore_batch_axis),
                  get_shape(self.value,
                            ignore_batch_axis=self.ignore_batch_axis)) and
              same_dtype(get_dtype(value), get_dtype(self.value)))
    else:
      return isinstance(value, type(self.value))

  def __repr__(self):
    if self.numeric:
      return '<numeric shape: {} dtype: {}>'.format(
          get_shape(self.value), get_dtype(self.value))
    else:
      return '<{}>'.format(type(self.value))

  def example_value(self):
    return self.value


class FunDef(Expression):
  """An expression that defines a function."""

  @classmethod
  def transform(cls, node: ast.FunctionDef):
    name = Symbol(node.name)
    args = [(Symbol(arg.arg), transform_ast(arg.annotation))
            for arg in node.args.args]
    body = []
    for n in node.body:
      body.append(transform_ast(n))
    return cls(name, args, body)

  def __init__(self, fn_symbol: Symbol,
               args: t.List[t.Tuple[Symbol, Expression]],
               body: t.List[Expression]):
    self.fn_symbol = fn_symbol
    self.args = args
    self.body = body
    self.fn_name = fn_symbol.name

  def execute(self, namespace):
    if isinstance(namespace, HashTracker):
      return self.compute_hash(namespace)
    else:
      return self._execute(namespace)

  def _execute(self, namespace):
    """Execute the function definition on the given namespace."""
    def func(*args):
      new_namespace = namespace.child()
      if len(self.args) != len(args):
        raise ValueError(
            ('The number of arguments ({}) provided '
             'does not match the definition of {} ({})!').format(
                 len(args), self.fn_name, len(self.args)))
      for (arg_symbol, annotation_exp), arg_value in zip(self.args, args):
        if not namespace.ignore_arg_annotations:
          annotation = annotation_exp.execute(namespace)
          if ((annotation is not None) and (not annotation.check(arg_value))):
            raise ValueError(
                'Argument value {} does not match annotation {}!'.format(
                    arg_value, annotation))
        new_namespace[arg_symbol.name] = arg_value
      value = None
      for exp in self.body:
        value = exp.execute(new_namespace)
      return value

    namespace[self.fn_symbol.name] = Function(func, len(self.args))
    return namespace[self.fn_symbol.name]

  def to_string(self, indent_level=0):
    head = 'def {}({}):\n'.format(
        self.fn_symbol.to_string(),
        args2str(self.args))
    rest = block2str(self.body, indent_level+1)
    result = head + rest
    return apply_indent(result, indent_level)

  def try_next_insertion(self, rng, namespace):
    """Iterate through the possible insertions."""
    rng, sub_rng = jax.random.split(rng)
    body = self.body
    new_namespace = namespace.child()
    # Add each argument with its annotations.
    for arg_symbol, annotation_exp in self.args:
      annotation = annotation_exp.execute(new_namespace)
      assert annotation is not None
      new_namespace[arg_symbol.name] = annotation.example_value()
    for new_exp_list in candidate_insertions(
        sub_rng, body[:-1], new_namespace.copy()):
      self.body = new_exp_list + [self.body[-1]]
      _, success = self.check(namespace)
      if success:
        yield
    # Recover the original body if none of the insertions succeed.
    self.body = body

  def try_next_deletion(self, rng, namespace):
    """Iterate through the possible deletions."""
    if len(self.body) == 1:
      raise StopIteration
    rng, _ = jax.random.split(rng)
    body = self.body
    for new_exp_list in candidate_deletions(rng, body[:-1], namespace):
      self.body = new_exp_list + [self.body[-1]]
      _, success = self.check(namespace)
      if success:
        yield
    # Recover the original body if none of the insertions succeed.
    self.body = body

  def try_modification(self, rng, namespace):
    body = self.body
    new_namespace = namespace.child()
    # Add each argument with its annotations.
    for arg_symbol, annotation_exp in self.args:
      annotation = annotation_exp.execute(new_namespace)
      assert annotation is not None
      new_namespace[arg_symbol.name] = annotation.example_value()
    return try_modification(rng, body[:-1], new_namespace)

  def check(self, namespace):
    """Check whether the given function can be executed without error."""
    new_namespace = namespace.child()
    # Add each argument with its annotations.
    for arg_symbol, annotation_exp in self.args:
      annotation = annotation_exp.execute(new_namespace)
      assert annotation is not None
      new_namespace[arg_symbol.name] = annotation.example_value()
    try:
      result = None
      for exp in self.body:
        result = exp.execute(new_namespace)
    except get_all_exceptions() as e:
      return e, False
    return result, True

  def simplify(self):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Simplify the function definition by removing redundant expressions."""
    new_namespace = DependencyTracker()
    result = None
    for exp in self.body:
      result = exp.execute(new_namespace)
    new_body = []
    for exp in self.body[:-1]:
      if exp in result:
        exp.simplify(result)
        new_body.append(exp)
    return_exp = self.body[-1]
    new_body.append(return_exp)
    self.body = new_body
    return result

  def compute_hash(self, namespace):
    new_namespace = namespace.child()
    for i, (arg_symbol, _) in enumerate(self.args):
      new_namespace[arg_symbol.name] = Hash(f'{i}_arg')
    for exp in self.body:
      result = exp.execute(new_namespace)
    flat_result = jax.tree_util.tree_leaves(result)
    final_result = Hash(''.join([r.value for r in flat_result]))
    namespace[self.fn_symbol.name] = final_result
    return final_result


class Return(Expression):
  """An expression that returns the value.

  The same role as return statement in a function definition. However, we
  enforce each function to be pure and that the `Return` expression can only be
  used as the last expression in a function definition. It is just an identity
  function that returns the value.
  """

  @classmethod
  def transform(cls, node: ast.Return):
    return cls(transform_ast(node.value))

  def __init__(self, value: Expression):
    self.value = value

  # Because we assume `return` is always the last expression in a function
  # definition, its implementation is simplified as an identity function,
  # without the need to handle exiting from the function call.
  def execute(self, namespace):
    return_value = self.value.execute(namespace)
    if isinstance(namespace, DependencyTracker):
      dependency = set()
      for leave in jax.tree_leaves(return_value):
        dependency = dependency.union(leave)
      return dependency
    else:
      return return_value

  def to_string(self, indent_level=0):
    result = 'return {}\n'.format(self.value.to_string())
    return apply_indent(result, indent_level)


class Sequence(Expression):
  """A sequence like list or tuple."""

  @classmethod
  def transform(cls, node):
    return cls([transform_ast(item) for item in node.elts])

  def __init__(self, elements):
    self.elements = elements

  def execute(self, namespace):
    return [e.execute(namespace) for e in self.elements]

  def to_string(self, indent_level=0):
    raise NotImplementedError


class Tuple(Sequence):

  def to_string(self, indent_level=0):
    item_string = items2str(self.elements)
    if len(self.elements) == 1:
      item_string += ','
    result = '({})'.format(item_string)
    return apply_indent(result, indent_level)


class List(Sequence):

  def to_string(self, indent_level=0):
    item_string = items2str(self.elements)
    result = '[{}]'.format(item_string)
    return apply_indent(result, indent_level)


def match(a, b):
  flat_a, tree_def_a = jax.tree_flatten(a)
  flat_b, tree_def_b = jax.tree_flatten(b)
  return (tree_def_a == tree_def_b and
          all(map(simple_match, flat_a, flat_b)))


def simple_match(a, b):
  if is_simple_numeric(a):
    return (is_simple_numeric(b) and
            get_shape(a) == get_shape(b) and
            get_dtype(a) == get_dtype(b))
  else:
    return isinstance(a, type(b)) and isinstance(b, type(a))


class Assignment(Expression):
  """An assignment expression."""

  @classmethod
  def transform(cls, node: ast.Assign):
    if len(node.targets) > 1:
      target = Tuple([transform_ast(target) for target in node.targets])
    else:
      target = transform_ast(node.targets[0])
    value = transform_ast(node.value)
    return cls(target, value)

  @classmethod
  def random_generate(cls, rng, namespace):
    """Randomly generate an assignment expression."""
    rng, sub_rng = jax.random.split(rng)
    for exp in FunCall.random_generate(sub_rng, namespace):
      result = exp.execute(namespace)
      rng, sub_rng = jax.random.split(rng)
      new_name = namespace.generate_new_name(sub_rng)
      if new_name:
        valid_targets = [new_name]
      else:
        valid_targets = []
      for k, v in namespace.items():
        if match(v, result):
          valid_targets.append(k)
      rng, sub_rng = jax.random.split(rng)
      for target in random_iter(sub_rng, valid_targets):
        yield Assignment(Symbol(target), exp)

  def __init__(self, target, value):
    self.target = target
    self.value = value

  def execute(self, namespace):
    value = self.value.execute(namespace)
    target = extract_tuple_target(self.target)
    if isinstance(namespace, DependencyTracker):
      # Include condition dependencies to all branch statements.
      if CONDITIONAL in namespace:
        value = value.union(namespace[CONDITIONAL])
      jax.tree_map(lambda x: x.add(self), value)
      if isinstance(target, (tuple, list)):
        target_symbols = jax.tree_util.tree_leaves(target)
        value = split_dependencies(value, len(target_symbols))
    elif isinstance(namespace, HashTracker):
      # Include condition hash to all branch statements.
      if CONDITIONAL in namespace:
        value = value.union(namespace[CONDITIONAL])
      if isinstance(target, (tuple, list)):
        target_symbols = jax.tree_util.tree_leaves(target)
        value = split_hash(value, len(target_symbols))
    else:
      if isinstance(target, (tuple, list)):
        value = tuple(value)

    assign_value(self.target, value, namespace)

  def to_string(self, indent_level=0):
    result = '{} = {}\n'.format(
        self.target.to_string(),
        self.value.to_string())
    return apply_indent(result, indent_level)

  def mutate(self, rng, namespace):
    return self.value.mutate(rng, namespace)


def assign_value(target, value, namespace):
  """Assign the value to the given target in the namespace."""
  target = extract_tuple_target(target)
  if isinstance(target, Symbol):
    namespace[target.name] = value
  elif isinstance(target, (tuple, list)):
    target_symbols, treedef = jax.tree_util.tree_flatten(target)
    target_values = treedef.flatten_up_to(value)
    for ts, val in zip(target_symbols, target_values):
      namespace[ts.name] = val
  else:
    raise ValueError('Invalid target when assigning values!')


def extract_tuple_target(exp):
  if isinstance(exp, Symbol):
    return exp
  elif isinstance(exp, Tuple):
    return tuple([extract_tuple_target(x) for x in exp.elements])
  else:
    raise ValueError('Not a valid tuple target: {}!'.format(exp))


class BinOp(Expression):
  """Expression using primitive binary ops like +, -, / and *."""

  @classmethod
  def transform(cls, node: ast.BinOp):
    left = transform_ast(node.left)
    right = transform_ast(node.right)
    op_map = {ast.Add: '+',
              ast.Sub: '-',
              ast.Mult: '*',
              ast.Div: '/'}
    op = op_map[type(node.op)]
    return cls(op, left, right)

  def __init__(self, op, left, right):
    self.op = op
    self.left = left
    self.right = right

  def execute(self, namespace):
    op = namespace[self.op]
    left = self.left.execute(namespace)
    right = self.right.execute(namespace)
    return op(left, right)

  def to_string(self, indent_level=0):
    result = '({} {} {})'.format(
        self.left.to_string(), self.op, self.right.to_string())
    return apply_indent(result, indent_level)


class UnaryOp(Expression):
  """Expression using primitive binary ops like +, -, / and *."""

  @classmethod
  def transform(cls, node: ast.UnaryOp):
    arg = transform_ast(node.operand)
    op_map = {ast.Invert: INVERT,
              ast.Not: NOT,
              ast.UAdd: UADD,
              ast.USub: USUB}
    op = op_map[type(node.op)]
    return cls(op, arg)

  def __init__(self, op, arg):
    self.op = op
    self.arg = arg

  def execute(self, namespace):
    op_map = {
        INVERT: invert,
        NOT: negate,
        UADD: unary_add,
        USUB: unary_sub}
    op = op_map[self.op]
    arg = self.arg.execute(namespace)
    return op(arg)

  def to_string(self, indent_level=0):
    tmpl = '{} {}' if self.op == NOT else '{}{}'
    result = tmpl.format(self.op, self.arg.to_string())
    return apply_indent(result, indent_level)


class BoolOp(Expression):
  """Expression using primitive bool ops like `and` and `or`."""

  @classmethod
  def transform(cls, node: ast.BoolOp):
    values = [transform_ast(val) for val in node.values]
    if isinstance(node.op, ast.And):
      return cls(AND, values)
    elif isinstance(node.op, ast.Or):
      return cls(OR, values)
    else:
      raise ValueError('Unknown bool op: {}'.format(
          type(node.op)))

  def __init__(self, op, values):
    self.op = op
    self.values = values

  def execute(self, namespace):
    if self.op == AND:
      for value in self.values:
        if not value.execute(namespace):
          return False
      return True
    else:
      for value in self.values:
        if value.execute(namespace):
          return True
      return False

  def to_string(self, indent_level=0):
    op_str = ' {} '.format(self.op)
    result = '({})'.format(op_str.join([v.to_string() for v in self.values]))
    return apply_indent(result, indent_level)


class CompareOp(Expression):
  """A comparison expression."""

  @classmethod
  def transform(cls, node: ast.Compare):
    """Transform an ast.Compare node into a comparison expression."""
    left = transform_ast(node.left)
    op = node.ops[0]
    right = transform_ast(node.comparators[0])
    op_map = {ast.Eq: EQ,
              ast.NotEq: NEQ,
              ast.Lt: LT,
              ast.LtE: LTE,
              ast.Gt: GT,
              ast.GtE: GTE}
    return cls(op_map[type(op)], left, right)

  def __init__(self, op, left, right):
    self.op = op
    self.left = left
    self.right = right

  def execute(self, namespace):
    op_map = {
        EQ: eq,
        NEQ: neq,
        LT: lt,
        LTE: lte,
        GT: gt,
        GTE: gte}
    left = self.left.execute(namespace)
    right = self.right.execute(namespace)
    if isinstance(namespace, HashTracker):
      return Hash(f'{left.value}{self.op}{right.value}')
    elif isinstance(namespace, DependencyTracker):
      dependency = set()
      dependency = dependency.union(left)
      dependency = dependency.union(right)
      return dependency
    else:
      return op_map[self.op](left, right)

  def mutate(self, rng, namespace=None):
    """Randomly change the CompareOp expression."""
    rng, sub_rng = jax.random.split(rng)
    childs = [self.left, self.op, self.right]
    order = jax.random.permutation(sub_rng, len(childs))
    for i in order:
      child = childs[i]
      if isinstance(child, Symbol):
        replacement = random_replace_symbol(rng, child, namespace)
        if replacement:
          if child is self.left:
            self.left = replacement
          elif child is self.right:
            self.right = replacement
          return True
      elif isinstance(child, Atom):
        child.mutate(rng)
        return True
      elif isinstance(child, str):
        op_list = [op for op in [EQ, NEQ, LT, LTE, GT, GTE] if op != self.op]
        selected_id = jax.random.randint(rng, (), minval=0, maxval=len(op_list))
        self.op = op_list[selected_id]  # pytype: disable=unsupported-operands  # jax-types
        return True
    return False

  def to_string(self, indent_level=0):
    result = '({} {} {})'.format(
        self.left.to_string(),
        self.op,
        self.right.to_string())
    return apply_indent(result, indent_level)


def invert(a):
  return ~a


def negate(a):
  return not a


def unary_add(a):
  return a


def unary_sub(a):
  return -a


def eq(a, b):
  return a == b


def neq(a, b):
  return a != b


def lt(a, b):
  return a < b


def lte(a, b):
  return a <= b


def gt(a, b):
  return a > b


def gte(a, b):
  return a >= b


def add(a, b):
  return a + b


def mult(a, b):
  return a * b


def sub(a, b):
  return a - b


def div(a, b):
  return a / b


def maximum(a, b):
  return jnp.maximum(a, b)


def minimum(a, b):
  return jnp.minimum(a, b)


def tree_max(a, b):
  return apply_binary_op(maximum, a, b)


def tree_min(a, b):
  return apply_binary_op(minimum, a, b)


def tree_add(a, b):
  return apply_binary_op(add, a, b)


def tree_mult(a, b):
  return apply_binary_op(mult, a, b)


def tree_sub(a, b):
  return apply_binary_op(sub, a, b)


def tree_div(a, b):
  return apply_binary_op(div, a, tree_add(b, 1e-8))


def apply_binary_op(op, a, b):
  """Apply a binary op to two pytrees or simple values."""
  flat_a, treedef_a = jax.tree_flatten(a)
  flat_b, treedef_b = jax.tree_flatten(b)
  flat_result = []
  treedef = treedef_a
  if len(flat_a) != len(flat_b):
    if len(flat_a) == 1:
      flat_a = flat_a * len(flat_b)
      treedef = treedef_b
    elif len(flat_b) == 1:
      flat_b = flat_b * len(flat_a)
    else:
      raise ValueError('Cannot apply binary op!')
  for a_item, b_item in zip(flat_a, flat_b):
    flat_result.append(op(a_item, b_item))
  result = jax.tree_unflatten(treedef, flat_result)
  return result


class Conditional(Expression):
  """A conditionl expression."""

  @classmethod
  def transform(cls, node: ast.If):
    """Turn an ast.If node into a conditional expression."""
    test = transform_ast(node.test)
    true_branch = []
    for n in node.body:
      true_branch.append(transform_ast(n))

    false_branch = []
    for n in node.orelse:
      false_branch.append(transform_ast(n))

    return cls(test, true_branch, false_branch)

  def __init__(self, test, true_branch, false_branch):
    self.test = test
    self.true_branch = true_branch
    self.false_branch = false_branch

  def execute(self, namespace):
    test = self.test.execute(namespace)

    if isinstance(namespace, HashTracker):
      if CONDITIONAL not in namespace:
        h = test
      else:
        h = test.union(namespace[CONDITIONAL])
      self._execute_both_branches(namespace, h)
      # We don't need to return anything because the hashes are tracked in the
      # namespace.
      return None
    elif isinstance(namespace, DependencyTracker):
      # We should add all dependencies used in the condition and the conditional
      # statement itself to the dependencies of each statement in both branches.
      dependency = test.union(set([self]))
      if CONDITIONAL in namespace:
        dependency = dependency.union(namespace[CONDITIONAL])
      self._execute_both_branches(namespace, dependency)
      # We don't need to return anything because the dependencies are tracked
      # in the namespace.
      return None
    else:

      def _execute_true_branch(_):
        value = None
        child_namespace = namespace.child()
        for exp in self.true_branch:
          value = exp.execute(child_namespace)
        child_namespace_copy = self._keep_only_variables(child_namespace)
        return value, child_namespace_copy

      def _execute_false_branch(_):
        value = None
        child_namespace = namespace.child()
        for exp in self.false_branch:
          value = exp.execute(child_namespace)
        child_namespace_copy = self._keep_only_variables(child_namespace)
        return value, child_namespace_copy

      # This is still not strictly equivalent to Python's if...else... semantic.
      # Because in theory we could have different set of variables assigned in
      # each branch, making the returned namespaces different shapes.
      value, child_namespace_copy = jax.lax.cond(
          test, _execute_true_branch, _execute_false_branch, operand=None)
      for k, v in child_namespace_copy.items():
        namespace[k] = v

      return value

  def _execute_both_branches(self, namespace, conditional_value):
    """Execute the both branches and merge namespaces at the end."""
    true_branch_namespace = namespace.child()
    true_branch_namespace[CONDITIONAL] = conditional_value
    for exp in self.true_branch:
      exp.execute(true_branch_namespace)
    false_branch_namespace = namespace.child()
    false_branch_namespace[CONDITIONAL] = conditional_value
    for exp in self.false_branch:
      exp.execute(false_branch_namespace)
    self._merge_single_namespace(namespace, true_branch_namespace,
                                 false_branch_namespace)
    self._merge_single_namespace(namespace, false_branch_namespace,
                                 true_branch_namespace)

  def _merge_single_namespace(self, namespace, child1_namespace,
                              child2_namespace):
    """Merge entries in the child namespaces to the original namespace."""
    child1_branch_keys = child1_namespace.direct_keys()
    child2_branch_keys = child2_namespace.direct_keys()
    for key in child1_branch_keys:
      child1_branch_value = child1_namespace[key]
      if key not in child2_branch_keys:
        self._propagate(namespace, False, child1_branch_value, key)
      else:
        union_value = child1_branch_value.union(child2_namespace[key])
        self._propagate(namespace, True, union_value, key)

  def _propagate(self, namespace, appear_in_both_branch, value, key):
    """Propagate the vars created in the branch namespace to the parent namespace."""
    if key not in namespace or appear_in_both_branch:
      namespace[key] = value
    else:
      namespace[key] = namespace[key].union(value)

  def _keep_only_variables(self, namespace):
    """Only keep variables and remove functions."""
    namespace_copy = dict()
    for key, value in namespace.mapping.items():
      if value is None or isinstance(value,
                                     (ExampleAnnotation, Function, type)):
        continue
      namespace_copy[key] = value
    return namespace_copy

  def simplify(self, namespace):
    """Simplify the conditional statement by removing redundant expressions."""
    new_true_branch = []
    for exp in self.true_branch:
      if exp in namespace:
        if isinstance(exp, Conditional):
          exp.simplify(namespace)
        new_true_branch.append(exp)
    self.true_branch = new_true_branch
    new_false_branch = []
    for exp in self.false_branch:
      if exp in namespace:
        if isinstance(exp, Conditional):
          exp.simplify(namespace)
        new_false_branch.append(exp)
    self.false_branch = new_false_branch

  def mutate(self, rng, namespace=None):
    # TODO(b/192982773): Consider mutating expressions in branches.
    return self.test.mutate(rng, namespace)

  def to_string(self, indent_level=0):
    test = self.test.to_string()
    true_branch = block2str(
        self.true_branch, indent_level+1)
    result = apply_indent(
        'if {}:\n{}'.format(test, true_branch), indent_level)
    if self.false_branch:
      false_branch = block2str(
          self.false_branch, indent_level+1)
      result += apply_indent(
          'else:\n{}'.format(false_branch), indent_level)
    return result


class ForLoop(Expression):
  """An for loop expression."""

  @classmethod
  def transform(cls, node: ast.For):
    iter_target = transform_ast(node.target)
    iter_value = transform_ast(node.iter)
    body = []
    for n in node.body:
      body.append(transform_ast(n))
    return cls(iter_target, iter_value, body)

  def __init__(self, iter_target, iter_value, body):
    self.iter_target = iter_target
    self.iter_value = iter_value
    self.body = body

  def execute(self, namespace):
    for iter_value in self.iter_value.execute(namespace):
      assign_value(self.iter_target, iter_value, namespace)
      for exp in self.body:
        result = exp.execute(namespace)
    return result

  def to_string(self, indent_level=0):
    head = 'for {} in {}:\n'.format(
        self.iter_target.to_string(),
        self.iter_value.to_string())
    rest = block2str(self.body, indent_level+1)
    result = head + rest
    return apply_indent(result, indent_level)


def transform_ast(node: ast.AST) -> Expression:
  """Transform an ast node into an expression."""
  if isinstance(node, ast.FunctionDef):
    return FunDef.transform(node)
  elif isinstance(node, ast.Return):
    return Return.transform(node)
  elif isinstance(node, ast.Call):
    return FunCall.transform(node)
  elif isinstance(node, ast.Expr):
    return transform_ast(node.value)
  elif isinstance(node, ast.Assign):
    return Assignment.transform(node)
  elif isinstance(node, ast.Name):
    return Symbol(node.id)
  elif isinstance(node, ast.Tuple):
    return Tuple.transform(node)
  elif isinstance(node, ast.List):
    return List.transform(node)
  elif isinstance(node, ast.Num):
    return Atom(value=node.n)
  elif isinstance(node, ast.BinOp):
    return FunCall.transform_binop(node)
  elif isinstance(node, ast.UnaryOp):
    return UnaryOp.transform(node)
  elif isinstance(node, ast.BoolOp):
    return BoolOp.transform(node)
  elif isinstance(node, ast.For):
    return ForLoop.transform(node)
  elif isinstance(node, ast.If):
    return Conditional.transform(node)
  elif isinstance(node, ast.Compare):
    return CompareOp.transform(node)
  elif isinstance(node, ast.Subscript):
    raise ValueError('Subscription are not supported in transformation yet.')
  elif isinstance(node, ast.Str):
    return Atom(node.s)
  elif node is None:
    return Symbol(NONE)
  else:
    raise ValueError('{} not supported yet!'.format(type(node)))


def items2str(exps):
  return ', '.join(
      [exp.to_string() for exp in exps])


def args2str(args):
  """Generate a string representation of the function arguments."""
  result_list = []
  for arg in args:
    if isinstance(arg, Symbol):
      result_list.append(arg.to_string())
    elif isinstance(arg, tuple):
      if is_none(arg[1]):
        result_list.append(arg[0].to_string())
      else:
        result_list.append(arg[0].to_string() + ': ' + arg[1].to_string())
    else:
      raise ValueError('Wrong type of args: {}'.format(type(arg)))
  return ', '.join(result_list)


def block2str(exps, indent_level=0):
  # Return keyword 'pass' if the block is empty.
  if not exps:
    return apply_indent('pass', indent_level)
  return ''.join(
      [exp.to_string(indent_level) for exp in exps])


def is_none(exp):
  return isinstance(exp, Symbol) and exp.name == NONE


def get_python_primitives():
  """Get primitives python functions to add to default namespace."""
  mapping = {}
  mapping[NONE] = None

  def binary_math_op(fn):
    return Function(fn, 2, [is_numeric] * 2)
  mapping[ADD] = binary_math_op(tree_add)
  mapping[SUB] = binary_math_op(tree_sub)
  mapping[MULT] = binary_math_op(tree_mult)
  mapping[DIV] = binary_math_op(tree_div)
  mapping[MAX] = binary_math_op(tree_max)
  mapping[MIN] = binary_math_op(tree_min)

  mapping[ZIP] = zip
  mapping[RANGE] = range
  return mapping


class ProgramError(Exception):
  pass


class UnknownVarError(Exception):
  pass


class Namespace(collections.abc.MutableMapping):
  """A namespace that maintains the mapping between names and their values.

  Like the namespace in programming languages, it maps names to some values.
  A program should be executed in a given namespace, and could modify it during
  the execution, for example, defining new variables or functions.
  """

  def __init__(self, mapping=None, add_primitives=True, max_num_var=1000,
               ignore_arg_annotations=False):
    self.max_num_var = max_num_var
    self.mapping = dict()
    self.parent = None
    self.ignore_arg_annotations = ignore_arg_annotations

    if add_primitives:
      for k, v in get_python_primitives().items():
        self[k] = v

    if mapping is not None:
      for k, v in mapping.items():
        if isinstance(v, (float, int)):
          self[k] = jnp.array(v)
        else:
          self[k] = v

  def __getitem__(self, key):
    if key in self.mapping:
      return self.mapping[key]
    elif self.parent is None:
      raise UnknownVarError('Unknown variable {}'.format(key))
    else:
      return self.parent[key]

  def __setitem__(self, key, value):
    # Don't fall back to the outer namespace.
    if isinstance(value, (float, int)):
      self.mapping[key] = jnp.array(value)
    else:
      self.mapping[key] = value

  def __delitem__(self, key):
    del self.mapping[key]

  def __iter__(self):
    if self.parent is None:
      return iter(self.mapping)
    else:
      return itertools.chain(iter(self.mapping), iter(self.parent))

  def __len__(self):
    if self.parent is None:
      return len(self.mapping)
    else:
      return len(self.mapping) + len(self.parent)

  def __contains__(self, key):
    if key in self.mapping:
      return True
    elif self.parent is None:
      return False
    else:
      return key in self.parent

  def copy(self):
    new_namespace = Namespace(self.mapping, add_primitives=False)
    new_namespace.parent = self.parent
    new_namespace.max_num_var = self.max_num_var
    new_namespace.ignore_arg_annotations = self.ignore_arg_annotations
    return new_namespace

  def child(self):
    new_namespace = Namespace(self.mapping)
    new_namespace.ignore_arg_annotations = self.ignore_arg_annotations
    new_namespace.parent = self
    return new_namespace

  def to_abstract(self):
    new_namespace = self.copy()
    if new_namespace.parent is not None:
      new_namespace.parent = new_namespace.parent.to_abstract()
    for k, v in new_namespace.mapping.items():
      if isinstance(v, Function):
        new_namespace[k] = v.to_abstract()
    return new_namespace

  def get_function(self, name):
    fn = self[name]
    def new_fn(*args):
      try:
        result = fn(*args)
        return result, True
      except get_all_exceptions() as e:
        return e, False
    return new_fn

  def direct_keys(self):
    return self.mapping.keys()

  def generate_new_name(self, rng):
    for i in jax.random.permutation(rng, jnp.arange(0, self.max_num_var)):
      name = 'v{}'.format(i)
      if name not in self:
        return name
    else:
      # Reuse the names when all the names are used.
      return name


class HashTracker(collections.abc.MutableMapping):
  """A mapping from symbols (variable names) to the hash of their values."""

  def __init__(self, namespace=None, parent=None):
    self.mapping = {}
    self.parent = parent
    if namespace:
      for k in namespace.mapping.keys():
        self.mapping[k] = Hash(k)
      if namespace.parent:
        self.parent = HashTracker(namespace.parent)

  def __getitem__(self, key):
    if key in self.mapping:
      return self.mapping[key]
    elif self.parent is None:
      raise UnknownVarError('Unknown variable {}'.format(key))
    else:
      return self.parent[key]

  def __setitem__(self, key, value):
    if isinstance(value, Hash):
      self.mapping[key] = value
    else:
      raise ValueError(
          (f'The value is not a Hash object, '
           f'whose type is instead {type(value)}.'))

  def __delitem__(self, key):
    del self.mapping[key]

  def __contains__(self, key):
    if key in self.mapping:
      return True
    elif self.parent is None:
      return False
    else:
      return key in self.parent

  def __iter__(self):
    if self.parent is None:
      return iter(self.mapping.keys())
    else:
      keys = set(self.mapping.keys())
      parent_keys = set(self.parent.keys())
      return iter(keys.union(parent_keys))

  def __len__(self):
    if self.parent is None:
      return len(self.mapping)
    else:
      keys = set(self.mapping.keys())
      parent_keys = set(self.parent.keys())
      return len(keys.union(parent_keys))

  def child(self):
    return HashTracker(parent=self)

  def to_dict(self):
    """Convert itself to a dictionary."""
    if self.parent:
      d = self.parent.to_dict()
      d.update(self.mapping)
    else:
      d = self.mapping.copy()

    new_d = {}
    for k, v in d.items():
      new_d[k] = v.value

    return new_d

  def direct_keys(self):
    return self.mapping.keys()


class DependencyTracker(Namespace):
  """A mock namespace for tracking dependencies."""

  def __init__(self, mapping=None):
    super().__init__(mapping, add_primitives=False)

  def __getitem__(self, key):
    if key in self.mapping:
      return self.mapping[key]
    elif self.parent is None:
      # Return empty set to indicate no dependencies.
      return set()
    else:
      return self.parent[key]

  def child(self):
    new_dependency_tracker = DependencyTracker(self.mapping)
    new_dependency_tracker.parent = self
    return new_dependency_tracker


def always_true(*args, **kwargs):
  del args
  del kwargs
  return True


def check_validity(fn, *args):
  try:
    jax.eval_shape(fn, *args)
  except (TypeError, ValueError) as e:
    logging.info('Function call is invalid due to %s.', e)
    return False
  return True


def validity_checker(fn):
  def f(*args):
    return check_validity(fn, *args)
  return f


def str2hash(string):
  return hashlib.md5(string.encode('utf-8')).hexdigest()


def split_hash(h, n):
  return tuple(Hash(h.value + str(i)) for i in range(n))


def split_dependencies(d, n):
  return tuple(d.copy() for _ in range(n))


class Hash(object):
  """A hash of a value generated during the computation of a program."""

  def __init__(self, string):
    self.value = str2hash(string)

  def combine(self, hash_list):
    hash_values = [h.value for h in hash_list]
    arg_string = ', '.join(hash_values)
    new_value = str2hash(f'{self.value}({arg_string})')
    return Hash(new_value)

  def union(self, other):
    new_value = str2hash(f'{self.value}{other.value}')
    return Hash(new_value)

  def __repr__(self):
    return f'<Hash: {self.value}>'


class Function(collections.abc.Callable):
  """An function object that handles concrete and abstract execution.

  The function object is a wrapper around a python callable.
  Besides the normal execution, it also handles:
  1. abstract execution that only computes the shape and dtype of the result.
  2. collect valid arguments from a given namespace, which is used to support
     the generation of new function calls.
  """

  def __init__(self, fn, num_args, check_arg_fns=None, annotations=None):
    self.fn = fn
    self.num_args = num_args
    if check_arg_fns is not None:
      self.check_arg_fns = check_arg_fns
    elif annotations is not None:
      self.check_arg_fns = [a.check for a in annotations]
    else:
      self.check_arg_fns = [always_true] * num_args

    self.check_all_args_fn = validity_checker(fn)

  def __call__(self, *args):
    return self.fn(*args)

  def to_abstract(self):

    def abstract_fn(*args, **kwargs):
      result = jax.eval_shape(self.fn, *args, **kwargs)
      return result

    new_function = Function(abstract_fn, self.num_args, self.check_arg_fns)
    new_function.check_all_args_fn = self.check_all_args_fn
    return new_function

  def collect_valid_args(self, namespace, allow_new_atom=True):
    """Collect all the valid arguments for the given function."""
    args_list = [[]]
    items = list(namespace.items())
    if allow_new_atom:
      items.append((None, 1.0))
    for i in range(self.num_args):
      new_args_list = []
      for k, v in items:
        if self.check_arg_fns[i](v):
          for args in args_list:
            new_args = args + [(k, v)]
            if ((len(new_args) < self.num_args) or
                self.check_all_args_fn(*[x[-1] for x in new_args])):
              new_args_list.append(new_args)
      args_list = [args for args in new_args_list if len(args) == (i+1)]

    # Retrieve the args.
    result = []
    for args in args_list:
      new_args = []
      for name, value in args:
        if name is None:
          new_args.append(Atom(value))
        else:
          new_args.append(Symbol(name))
      result.append(new_args)
    return result


# Type and shape related utilities.
def is_numeric(x):
  return is_simple_numeric(x) or is_compound_numeric(x)


def is_simple_numeric(x):
  return (isinstance(x, float) or
          isinstance(x, int) or
          (hasattr(x, 'shape') and hasattr(x, 'dtype')))


def is_compound_numeric(x):
  leaves = jax.tree_leaves(x)
  return leaves and all(map(is_simple_numeric, leaves))


def randint(key, minval, maxval):
  return jax.random.randint(key, [1], minval, maxval)[0]


def simple_get_shape(x, ignore_batch_axis=False):
  """Get the shape information of simple values like int, float or arrays."""
  if isinstance(x, (int, float)):
    if ignore_batch_axis:
      # int and float don't have a batch dimension,
      # so should not match anything, returning a None.
      return None
    else:
      return ()
  elif hasattr(x, 'shape') and hasattr(x, 'dtype'):
    if ignore_batch_axis:
      return x.shape[1:]
    else:
      return x.shape
  else:
    raise ValueError('input is not numeric!')


def simple_get_dtype(x):
  """Get the dtype information of simple values like int, float or arrays."""
  if isinstance(x, int):
    # jnp.zeros((), jnp.int32).dtype is different from jnp.int32.
    return jnp.zeros((), jnp.int32).dtype
  elif isinstance(x, float):
    # jnp.zeros((), jnp.float32).dtype is different from jnp.float32.
    return jnp.zeros((), jnp.float32).dtype
  elif hasattr(x, 'shape') and hasattr(x, 'dtype'):
    return x.dtype
  else:
    raise ValueError('input is not numeric!')


def get_shape_dtype(x):
  shape = jax.tree_map(simple_get_shape, x)
  dtype = jax.tree_map(simple_get_dtype, x)
  return shape, dtype


def get_shape(a, ignore_batch_axis=False):
  return jax.tree_map(
      lambda x: simple_get_shape(x, ignore_batch_axis=ignore_batch_axis), a)


def get_dtype(x):
  return jax.tree_map(simple_get_dtype, x)


def normalize_code(code: str):
  """Normalize the given code unnecessary indentations and newlines."""
  code = code.strip('\n')
  if not code:
    raise ValueError('The code is empty.')
  lines = code.split('\n')
  min_indent = -1
  new_lines = []
  for line in lines:
    if not line.strip():
      continue
    leading_spaces = len(line) - len(line.lstrip())
    if (min_indent < 0) or (leading_spaces < min_indent):
      min_indent = leading_spaces
    new_lines.append(line)

  code = '\n'.join([line[min_indent:] for line in new_lines])
  return code


class Environment(metaclass=abc.ABCMeta):
  """An environment contains the information to evaluate and mutate a program.

  Each experiment should implement a Environment object that defines how to
  evaluate and mutate a program, which provides the interface that
  a search algorithm can use.
  """

  @abc.abstractmethod
  def get_namespace(self):
    raise NotImplementedError

  @abc.abstractmethod
  def evaluate(self, program):
    raise NotImplementedError

  @abc.abstractmethod
  def mutate(self, program):
    raise NotImplementedError
