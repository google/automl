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
r"""Utilities for dataset creation.
"""

import functools
import os
from typing import Mapping, MutableMapping, Union, Optional, Callable

import einops
import jax
import jax.numpy as jnp
import numpy as np
import seqio
import t5.data.preprocessors
import tensorflow as tf


################################################################################
# Type aliases.
Batch = MutableMapping[str, Union[np.ndarray, jnp.ndarray]]
Processor = Callable[[Batch], Batch]

################################################################################
# Tokenizers / vocabularies.

T5_CC_VOCAB = 'vb32000_t5_cc.model'
OPENMIX_V1_VOCAB = 'vb100864_openmix_v1.model'

ALL_VOCABS = [('vb32000_t5_cc', T5_CC_VOCAB),
              ('vb100864_openmix_v1', OPENMIX_V1_VOCAB)]

################################################################################
# PT datasets.


def add_pt_task_v1(name, source, vocab, add_eos=False,
                   use_reduce_concat_split=True):
  """Adds a seqio task for a PT dataset."""
  preprocessors = [
      functools.partial(
          t5.data.preprocessors.rekey,
          key_map={
              'inputs': None,
              'targets': 'text',
          },
      ),
      seqio.preprocessors.tokenize,
      # Note that append_eos will respect the `add_eos`` field in
      # `output_features``.
      seqio.preprocessors.append_eos,
  ]
  if use_reduce_concat_split:
    preprocessors += [
        t5.data.preprocessors.reduce_concat_tokens,
        t5.data.preprocessors.split_tokens_to_targets_length,
    ]
  seqio.TaskRegistry.remove(name)
  seqio.TaskRegistry.add(
      name,
      source=source,
      preprocessors=preprocessors,
      output_features={
          'targets': seqio.Feature(
              seqio.SentencePieceVocabulary(vocab),
              add_eos=add_eos, dtype=tf.int32
              ),
          },
  )


def add_lm1b_task():
  """Adds LM1B tasks."""
  lm1b_source = seqio.TfdsDataSource(
      tfds_name='lm1b:1.1.0',
      splits={
          'train': 'train[:90%]',
          'validation': 'train[90%:]',
          'test': 'test'})
  minilm1b_source = seqio.TfdsDataSource(
      tfds_name='lm1b:1.1.0',
      splits={
          'train': 'train[:500]',
          'validation': 'train[500:1000]',
          'test': 'test'})
  for name, source in [('lm1b', lm1b_source),
                       ('minilm1b', minilm1b_source)]:
    for vocab_name, vocab in ALL_VOCABS:
      task_name = f'{name}.{vocab_name}'
      add_pt_task_v1(task_name, source, vocab,
                     use_reduce_concat_split=False)

add_lm1b_task()


def add_c4_task():
  """Adds C4 tasks."""
  source = seqio.TfdsDataSource(tfds_name='c4:3.1.0')
  for vocab_name, vocab in ALL_VOCABS:
    task_name = f'c4.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab,
                   use_reduce_concat_split=True)

add_c4_task()


def add_imdb_reviews_task():
  """Adds imdb_reviews tasks."""
  source = seqio.TfdsDataSource(
      tfds_name='imdb_reviews:1.0.0',
      splits={
          'train': 'train[:90%]',
          'validation': 'train[90%:]',
          'test': 'test'})
  name = 'imdb_reviews'
  for vocab_name, vocab in ALL_VOCABS:
    task_name = f'{name}.{vocab_name}'
    add_pt_task_v1(task_name, source, vocab,
                   use_reduce_concat_split=False)

add_imdb_reviews_task()


# ###############################################################################
# # Dataset utilities.


class Dataset:
  """A wrapper of tf.data.Dataset to add processors with numpy and jax."""

  def __init__(
      self, tf_dataset: tf.data.Dataset,
      processors: Optional[list[Processor]] = None):
    self._tf_dataset = tf_dataset
    if processors is None:
      processors = []
    self._processors = processors

  def add_processor(self, processor: Processor):
    self._processors.append(processor)
    return self._processors

  def repeat(self, num_repeat):
    return self.copy(tf_dataset=self._tf_dataset.repeat(num_repeat))

  def copy(self, tf_dataset: Optional[tf.data.Dataset] = None,
           processors: Optional[list[Processor]] = None):
    if tf_dataset is None:
      tf_dataset = self._tf_dataset
    if processors is None:
      processors = self._processors
    return Dataset(tf_dataset, processors)

  def __iter__(self):
    def generator():
      for batch in self._tf_dataset.as_numpy_iterator():
        for processor in self._processors:
          batch = processor(batch)
        yield batch
    return generator()


def create_dataset(config, start_steps):
  """Creates the train and validation datasets from given config."""
  if config.batch_size % jax.device_count() == 0:
    local_batch_size = config.batch_size // jax.process_count()
  else:
    raise ValueError(f'Batch size {config.batch_size} must be divisible'
                     f' by total number of cores {jax.device_count()}.')

  if config.feature_converter_name == 'LMFeatureConverter':
    task_feature_lengths = {'targets': config.seq_len}
    feature_converter = seqio.LMFeatureConverter(pack=config.use_packing)
  elif config.feature_converter_name == 'PrefixLMFeatureConverter':
    task_feature_lengths = {
        'inputs': config.seq_len // 2, 'targets': config.seq_len // 2}
    feature_converter = seqio.PrefixLMFeatureConverter(
        pack=config.use_packing,
    )
  else:
    raise ValueError(
        f'Unsupported feature converter type: {config.feature_converter_name}'
    )
  tf_validation_set = None
  train_processors = []
  validation_processors = []
  if config.dataset_name.startswith('simply_det'):
    raise ValueError('not supported!')
  else:
    tf_train_set = seqio.get_dataset(
        config.dataset_name,
        task_feature_lengths=task_feature_lengths,
        dataset_split='train',
        shuffle=True,
        num_epochs=None,
        use_cached=False,
        seed=config.dataset_seed,
        batch_size=config.batch_size,
        feature_converter=feature_converter)
    train_processors.append(select_local_batch)
    if config.use_validation_set:
      if config.validation_eval_batch_size == -1:
        eval_batch_size = config.batch_size
      else:
        eval_batch_size = config.validation_eval_batch_size
      tf_validation_set = seqio.get_dataset(
          config.dataset_name,
          task_feature_lengths=task_feature_lengths,
          dataset_split='validation',
          shuffle=False,
          num_epochs=1,
          use_cached=False,
          seed=None,
          batch_size=eval_batch_size,
          # Do not use packing for validation set.
          feature_converter=feature_converter)
      validation_processors.append(select_local_batch)

  if config.add_chat_loss_mask:
    vocab = seqio.SentencePieceVocabulary(config.vocab_path)
    mask_start_id = vocab.encode(config.mask_start_token)[0]
    mask_end_id = vocab.encode(config.mask_end_token)[0]
    add_chat_loss_mask_fn = jax.jit(functools.partial(
        add_chat_loss_mask,
        mask_start_id=mask_start_id,
        mask_end_id=mask_end_id))
    train_processors.append(add_chat_loss_mask_fn)
    validation_processors.append(add_chat_loss_mask_fn)

  train_set = Dataset(tf_train_set, train_processors)
  if tf_validation_set is not None:
    validation_set = Dataset(tf_validation_set, validation_processors)
  else:
    validation_set = None

  return train_set, validation_set


def select_local_batch(batch: Batch) -> Batch:
  """Selects the batch for the given process."""
  select_local_array_fn = functools.partial(
      select_local_array,
      process_index=jax.process_index(),
      num_processes=jax.process_count())
  new_batch = jax.tree_util.tree_map(select_local_array_fn, batch)
  return new_batch


def select_local_array(
    array: np.ndarray,
    process_index: int,
    num_processes: int) -> np.ndarray:
  """Selects the batch for the given process."""
  batch_size = array.shape[0]
  assert batch_size % num_processes == 0
  local_batch_size = batch_size // num_processes
  start_index = process_index * local_batch_size
  end_index = start_index + local_batch_size
  return array[start_index:end_index]


def create_chat_loss_mask(token_ids, mask_start_id, mask_end_id):
  def f(carry, a):
    new_carry = jnp.where(
        a == mask_end_id, -2, jnp.where(a == mask_start_id, -1, carry))
    return new_carry, carry

  token_ids = einops.rearrange(token_ids, 'b t -> t b')
  result = jax.lax.scan(f, jnp.full(token_ids.shape[1], -2), token_ids)[1] + 2
  return einops.rearrange(result, 't b -> b t')


def add_chat_loss_mask(batch, mask_start_id, mask_end_id):
  batch['decoder_loss_weights'] = create_chat_loss_mask(
      batch['decoder_target_tokens'], mask_start_id=mask_start_id,
      mask_end_id=mask_end_id) * batch['decoder_loss_weights']
  return batch
