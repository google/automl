# Copyright 2021 Google Research. All Rights Reserved.
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
"""EfficientNet V1 and V2 model configs."""
import functools
import re
import hparams
cfg_register = functools.partial(hparams.register, prefix='cfg:')


class BlockDecoder(object):
  """Block Decoder for readability."""

  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    return hparams.Config(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=int(options['s']),
        conv_type=int(options['c']) if 'c' in options else 0,
    )

  def _encode_block_string(self, block):
    """Encodes a block to a string."""
    args = [
        'r%d' % block.num_repeat,
        'k%d' % block.kernel_size,
        's%d' % block.strides,
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters,
        'c%d' % block.conv_type,
        'f%d' % block.fused_conv,
    ]
    if block.se_ratio > 0 and block.se_ratio <= 1:
      args.append('se%s' % block.se_ratio)
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of block.

    Returns:
      A list of namedtuples to represent blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent blocks arguments.
    Returns:
      a list of strings, each string is a notation of block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


#################### EfficientNet V1 configs ####################
v1_b0_block_str = [
    'r1_k3_s1_e1_i32_o16_se0.25',
    'r2_k3_s2_e6_i16_o24_se0.25',
    'r2_k5_s2_e6_i24_o40_se0.25',
    'r3_k3_s2_e6_i40_o80_se0.25',
    'r3_k5_s1_e6_i80_o112_se0.25',
    'r4_k5_s2_e6_i112_o192_se0.25',
    'r1_k3_s1_e6_i192_o320_se0.25',
]


efficientnetv1_params = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
}


def efficientnetv1_config(model_name='efficientnet-b0'):
  """EfficientNetV1 model config."""
  width_coefficient, depth_coefficient, isize, dropout_rate = (
      efficientnetv1_params[model_name])

  cfg = hparams.Config(
      model=dict(
          model_name=model_name,
          blocks_args=BlockDecoder().decode(v1_b0_block_str),
          width_coefficient=width_coefficient,
          depth_coefficient=depth_coefficient,
          dropout_rate=dropout_rate,
      ),
      eval=dict(isize=isize),
      train=dict(isize=0.8),  # 80% of eval size
      data=dict(augname='effnetv1_autoaug'),
  )
  return cfg


#################### EfficientNet V2 configs ####################
v2_base_block = [  # The baseline config for v2 models.
    'r1_k3_s1_e1_i32_o16_c1',
    'r2_k3_s2_e4_i16_o32_c1',
    'r2_k3_s2_e4_i32_o48_c1',
    'r3_k3_s2_e4_i48_o96_se0.25',
    'r5_k3_s1_e6_i96_o112_se0.25',
    'r8_k3_s2_e6_i112_o192_se0.25',
]


v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]


v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]


v2_l_block = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]

v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]
efficientnetv2_params = {
    # (block, width, depth, train_size, eval_size, dropout, randaug, mixup, aug)
    'efficientnetv2-s':  # 83.9% @ 22M
        (v2_s_block, 1.0, 1.0, 300, 384, 0.2, 10, 0, 'randaug'),
    'efficientnetv2-m':  # 85.2% @ 54M
        (v2_m_block, 1.0, 1.0, 384, 480, 0.3, 15, 0.2, 'randaug'),
    'efficientnetv2-l':  # 85.7% @ 120M
        (v2_l_block, 1.0, 1.0, 384, 480, 0.4, 20, 0.5, 'randaug'),

    'efficientnetv2-xl':
        (v2_xl_block, 1.0, 1.0, 384, 512, 0.4, 20, 0.5, 'randaug'),

    # For fair comparison to EfficientNetV1, using the same scaling and autoaug.
    'efficientnetv2-b0':  # 78.7% @ 7M params
        (v2_base_block, 1.0, 1.0, 192, 224, 0.2, 0, 0, 'effnetv1_autoaug'),
    'efficientnetv2-b1':  # 79.8% @ 8M params
        (v2_base_block, 1.0, 1.1, 192, 240, 0.2, 0, 0, 'effnetv1_autoaug'),
    'efficientnetv2-b2':  # 80.5% @ 10M params
        (v2_base_block, 1.1, 1.2, 208, 260, 0.3, 0, 0, 'effnetv1_autoaug'),
    'efficientnetv2-b3':  # 82.1% @ 14M params
        (v2_base_block, 1.2, 1.4, 240, 300, 0.3, 0, 0, 'effnetv1_autoaug'),
}


def efficientnetv2_config(model_name='efficientnetv2-s'):
  """EfficientNetV2 model config."""
  block, width, depth, train_size, eval_size, dropout, randaug, mix, aug = (
      efficientnetv2_params[model_name])

  cfg = hparams.Config(
      model=dict(
          model_name=model_name,
          blocks_args=BlockDecoder().decode(block),
          width_coefficient=width,
          depth_coefficient=depth,
          dropout_rate=dropout,
      ),
      train=dict(isize=train_size, stages=4, sched=True),
      eval=dict(isize=eval_size),
      data=dict(augname=aug, ram=randaug, mixup_alpha=mix, cutmix_alpha=mix),
  )
  return cfg


################################################################################
def get_model_config(model_name: str):
  """Main entry for model name to config."""
  if model_name.startswith('efficientnet-'):
    return efficientnetv1_config(model_name)
  if model_name.startswith('efficientnetv2-'):
    return efficientnetv2_config(model_name)
  raise ValueError(f'Unknown model_name {model_name}')
