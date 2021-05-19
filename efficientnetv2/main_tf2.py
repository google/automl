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
"""A simple script to train efficient net with tf2/keras."""

import copy
import os
import re

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import cflags
import datasets
import effnetv2_configs
import effnetv2_model
import hparams
import utils
FLAGS = flags.FLAGS


def build_tf2_optimizer(learning_rate,
                        optimizer_name='rmsprop',
                        decay=0.9,
                        epsilon=0.001,
                        momentum=0.9):
  """Build optimizer."""
  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    logging.info('Using Momentum optimizer')
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp optimizer')
    optimizer = tf.keras.optimizers.RMSprop(learning_rate, decay, momentum,
                                            epsilon)
  elif optimizer_name == 'adam':
    logging.info('Using Adam optimizer')
    optimizer = tf.keras.optimizers.Adam(learning_rate)
  else:
    raise Exception('Unknown optimizer: %s' % optimizer_name)

  return optimizer


class TrainableModel(effnetv2_model.EffNetV2Model):
  """Wraps efficientnet to make a keras trainable model.

  Handles efficientnet's multiple outputs and adds weight decay.
  """

  def __init__(self,
               model_name='efficientnetv2-s',
               model_config=None,
               name=None,
               weight_decay=0.0):
    super().__init__(
        model_name=model_name,
        model_config=model_config,
        name=name or model_name)

    self.weight_decay = weight_decay

  def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
    """Return regularization l2 loss loss."""
    var_match = re.compile(regex)
    return weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in self.trainable_variables
        if var_match.match(v.name)
    ])

  def train_step(self, data):
    features, labels = data
    images, labels = features['image'], labels['label']

    with tf.GradientTape() as tape:
      pred = self(images, training=True)[0]
      pred = tf.cast(pred, tf.float32)
      loss = self.compiled_loss(
          labels,
          pred,
          regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    features, labels = data
    images, labels = features['image'], labels['label']
    pred = self(images, training=False)[0]
    pred = tf.cast(pred, tf.float32)

    self.compiled_loss(
        labels,
        pred,
        regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}


def main(_) -> None:
  config = copy.deepcopy(hparams.base_config)
  config.override(effnetv2_configs.get_model_config(FLAGS.model_name))
  config.override(datasets.get_dataset_config(FLAGS.dataset_cfg))
  config.override(FLAGS.hparam_str)
  config.model.num_classes = config.data.num_classes
  strategy = config.runtime.strategy
  if strategy == 'tpu' and not config.model.bn_type:
    config.model.bn_type = 'tpu_bn'

  # log and save config.
  logging.info('config=%s', str(config))
  if 'train' in FLAGS.mode:
    if not tf.io.gfile.exists(FLAGS.model_dir):
      tf.io.gfile.makedirs(FLAGS.model_dir)
    config.save_to_yaml(os.path.join(FLAGS.model_dir, 'config.yaml'))

  if strategy == 'tpu':
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
  elif strategy == 'gpus':
    ds_strategy = tf.distribute.MirroredStrategy()
    logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
  else:
    if tf.config.list_physical_devices('GPU'):
      ds_strategy = tf.distribute.MirroredStrategy(['GPU:0'])
    else:
      ds_strategy = tf.distribute.MirroredStrategy(['CPU:0'])

  with ds_strategy.scope():
    train_split = config.train.split or 'train'
    eval_split = config.eval.split or 'eval'
    num_train_images = config.data.splits[train_split].num_images
    num_eval_images = config.data.splits[eval_split].num_images

    train_size = config.train.isize
    eval_size = config.eval.isize
    if train_size <= 16.:
      train_size = int(eval_size * train_size) // 16 * 16

    image_dtype = None
    if config.runtime.mixed_precision:
      image_dtype = 'bfloat16' if strategy == 'tpu' else 'float16'
      precision = 'mixed_bfloat16' if strategy == 'tpu' else 'mixed_float16'
      policy = tf.keras.mixed_precision.Policy(precision)
      tf.keras.mixed_precision.set_global_policy(policy)

    model = TrainableModel(
        config.model.model_name,
        config.model,
        weight_decay=config.train.weight_decay)

    if config.train.ft_init_ckpt:  # load pretrained ckpt for finetuning.
      model(tf.ones([1, 224, 224, 3]))
      ckpt = config.train.ft_init_ckpt
      utils.restore_tf2_ckpt(model, ckpt, exclude_layers=('_head', 'optimizer'))

    steps_per_epoch = num_train_images // config.train.batch_size
    total_steps = steps_per_epoch * config.train.epochs

    scaled_lr = config.train.lr_base * (config.train.batch_size / 256.0)
    scaled_lr_min = config.train.lr_min * (config.train.batch_size / 256.0)
    learning_rate = utils.WarmupLearningRateSchedule(
        scaled_lr,
        steps_per_epoch=steps_per_epoch,
        decay_epochs=config.train.lr_decay_epoch,
        warmup_epochs=config.train.lr_warmup_epoch,
        decay_factor=config.train.lr_decay_factor,
        lr_decay_type=config.train.lr_sched,
        total_steps=total_steps,
        minimal_lr=scaled_lr_min)

    optimizer = build_tf2_optimizer(
        learning_rate, optimizer_name=config.train.optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.train.label_smoothing, from_logits=True),
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='acc_top1'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc_top5')
        ],
    )

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(FLAGS.model_dir, 'ckpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir, update_freq=100)
    rstr_callback = tf.keras.callbacks.experimental.BackupAndRestore(
        backup_dir=FLAGS.model_dir)

    def get_dataset(training):
      """A shared utility to get input dataset."""
      if training:
        return ds_strategy.distribute_datasets_from_function(
            datasets.build_dataset_input(
                True, train_size, image_dtype, FLAGS.data_dir, train_split,
                config.data).distribute_dataset_fn(config.train.batch_size))
      else:
        return ds_strategy.distribute_datasets_from_function(
            datasets.build_dataset_input(
                False, eval_size, image_dtype, FLAGS.data_dir, eval_split,
                config.data).distribute_dataset_fn(config.eval.batch_size))

    if FLAGS.mode == 'traineval':
      model.fit(
          get_dataset(training=True),
          epochs=config.train.epochs,
          steps_per_epoch=steps_per_epoch,
          validation_data=get_dataset(training=False),
          validation_steps=num_eval_images // config.eval.batch_size,
          callbacks=[ckpt_callback, tb_callback, rstr_callback],
          # don't log spam if running on tpus
          verbose=2 if strategy == 'tpu' else 1,
      )
    elif FLAGS.mode == 'train':
      model.fit(
          get_dataset(training=True),
          epochs=config.train.epochs,
          steps_per_epoch=steps_per_epoch,
          callbacks=[ckpt_callback, tb_callback, rstr_callback],
          verbose=2 if strategy == 'tpu' else 1,
      )
    elif FLAGS.mode == 'eval':
      for ckpt in tf.train.checkpoints_iterator(
          FLAGS.model_dir, timeout=60 * 60 * 24):
        model.load_weights(ckpt)
        eval_results = model.evaluate(
            get_dataset(training=False),
            batch_size=config.eval.batch_size,
            steps=num_eval_images // config.eval.batch_size,
            callbacks=[tb_callback, rstr_callback],
            verbose=2 if strategy == 'tpu' else 1,
        )

        try:
          current_epoch = int(os.path.basename(ckpt).split('-')[1])
        except IndexError:
          logging.info('%s has no epoch info: stop!', ckpt)
          break

        logging.info('Epoch: %d, total %d', current_epoch, config.train.epochs)
        if current_epoch >= config.train.epochs:
          break
    else:
      raise ValueError(f'Invalid mode {FLAGS.mode}')


if __name__ == '__main__':
  cflags.define_flags()
  app.run(main)
