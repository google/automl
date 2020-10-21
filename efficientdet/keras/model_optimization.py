import functools

import tensorflow_model_optimization as tfmot


optimzation_methods = {
  'prune': tfmot.sparsity.keras.prune_low_magnitude
}

def set_config(configs):
  for key in configs:
    if key == 'prune':
      optimzation_methods[key] = functools.partial(
          tfmot.sparsity.keras.prune_low_magnitude,
          **configs[key])

def get_methods():
  return optimzation_methods

