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
"""Simply a language model."""

import os
import re
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import config_lib
import data_lib
import model_lib

_EXPERIMENT_CONFIG = flags.DEFINE_string(
    'experiment_config', 'TransformerLMTest', 'Name of the experiment config.')

_SHARDING_CONFIG = flags.DEFINE_string(
    'sharding_config', 'GSPMDSharding', 'Name of the sharding config.')

_EXPERIMENT_DIR = flags.DEFINE_string(
    'experiment_dir', '/tmp/simply_lm/', 'Path to save the experiment data.')

_MESH_SHAPE = flags.DEFINE_list(
    'mesh_shape',
    None,
    'Shape for the mesh, comma separated integers, e.g. 1,265,1',
)

_DCN_MESH_SHAPE = flags.DEFINE_list(
    'dcn_mesh_shape',
    None,
    'Shape for the dcn mesh, comma separated integers, e.g. 2,1,1',
)


def main(argv: Sequence[str]) -> None:
  del argv
  if mesh_shape := _MESH_SHAPE.value:
    mesh_shape = [int(i) for i in mesh_shape]
  if dcn_mesh_shape := _DCN_MESH_SHAPE.value:
    dcn_mesh_shape = [int(i) for i in dcn_mesh_shape]
  config = config_lib.ExperimentConfigRegistry.get_config(
      _EXPERIMENT_CONFIG.value)
  sharding_config = config_lib.ShardingConfigRegistry.get_config(
      _SHARDING_CONFIG.value)
  logging.info('config: %s', config)
  logging.info('sharding_config: %s', sharding_config)
  logging.info('mesh_shape: %s', mesh_shape)
  logging.info('dcn_mesh_shape: %s', dcn_mesh_shape)
  model_lib.run_experiment(
      config=config, sharding_config=sharding_config,
      mesh_shape=mesh_shape,
      dcn_mesh_shape=dcn_mesh_shape,
      create_dataset=data_lib.create_dataset,
      experiment_dir=_EXPERIMENT_DIR.value)


_TASK_HANDLE_RE = re.compile(r'(?:logs\.)?(\d+)\.(.*)\.([^.]+)\.\d+')

if __name__ == '__main__':
  app.run(main)
