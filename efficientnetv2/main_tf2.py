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
from typing import Tuple, Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

import cflags
import datasets
import effnetv2_configs
import effnetv2_model
import hparams
import utils
FLAGS = flags.FLAGS

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
      pred = self(images, training=True)
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
    pred = self(images, training=False)
    pred = tf.cast(pred, tf.float32)

    self.compiled_loss(
        labels,
        pred,
        regularization_losses=[self._reg_l2_loss(self.weight_decay)])

    self.compiled_metrics.update_state(labels, pred)
    return {m.name: m.result() for m in self.metrics}


def main(_) -> None:
  trainer = Trainer(hparams.base_config)

  # strategy context manager. 
  with trainer.ds_strategy.scope():
    # setup training regime.
    trainer.setup()
    # run trainer.
    trainer.run()

class Trainer:
  """A simple class for training Efficientnetv2 in tf2. 

  Attributes:
    config: Config. a configuration class.
    mode: Flags.mode.
    strategy: str. Training strategy. default('gpu').
    ds_strategy: tf.distributed. distributed strategy object.

  """
  def __init__(self, config):
    """__init__.

    Args:
        config: hparams base_config.
    """
    self.config = self.get_config(config)
    self.mode = FLAGS.mode
    self.strategy = self.config.runtime.strategy

    # set config model batch normalization.
    if self.strategy == 'tpu' and not self.config.model.bn_type:
      self.config.model.bn_type = 'tpu_bn'

    # log and save config.
    logging.info('config=%s', str(self.config))
    if 'train' in FLAGS.mode:
      if not tf.io.gfile.exists(FLAGS.model_dir):
        tf.io.gfile.makedirs(FLAGS.model_dir)
      self.config.save_to_yaml(os.path.join(FLAGS.model_dir, 'config.yaml'))

    self.ds_strategy = self.get_strategy()
    self._steps_per_epoch = None
    self._total_steps = None

  @property
  def steps_per_epoch(self):
    return self._steps_per_epoch

  @property
  def total_steps(self):
    return self._total_steps

  @staticmethod
  def get_optimizer_tf2(learning_rate: float,
      optimizer_name: str = 'rmsprop',
      decay: float=0.9,
      epsilon: float=0.001,
      momentum : float=0.9) -> Any:

    """ Function to build tf2 optimizer for training. """
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

  def num_samples(self) -> Tuple[int, int]:
    """Helper function to get num_samples"""
    train_split = self.config.train.split or 'train'
    eval_split = self.config.eval.split or 'eval'
    num_train_images = self.config.data.splits[train_split].num_images
    num_eval_images = self.config.data.splits[eval_split].num_images
    return num_train_images, num_eval_images

  def setup(self) -> None:
    """ Function for setting up model and training setup. """
    logging.info('Setting up training regime for efficientnetv2')

    self.num_train_images, self.num_eval_images = self.num_samples()
    self._steps_per_epoch = self.num_train_images // self.config.train.batch_size
    self._total_steps = self._steps_per_epoch * self.config.train.epochs
    self.train_size = self.config.train.isize
    self.eval_size = self.config.eval.isize

    if self.train_size <= 16.:
      self.train_size = int(self.eval_size * self.train_size) // 16 * 16

    # set the precision for training.
    self.set_precision()

    # get efficientnetv2 model
    self.model = self.get_model()
    

  def get_callbacks(self, istrain : bool = True) -> Tuple[Any, Any, Any]:
    """ Get tensorflow callbacks. """
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir, update_freq=100)

    rstr_callback = utils.ReuableBackupAndRestore(backup_dir=FLAGS.model_dir)

    if not istrain:
      return tb_callback, rstr_callback

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(FLAGS.model_dir, 'ckpt-{epoch:d}'),
        verbose=1,
        save_weights_only=True)

    return ckpt_callback, tb_callback, rstr_callback
    

  def get_model(self) -> TrainableModel:
    """ A helper function to get efficientnetv2 model. """
    model = TrainableModel(
        self.config.model.model_name,
        self.config.model,
        weight_decay=self.config.train.weight_decay)

    if self.config.train.ft_init_ckpt:  # load pretrained ckpt for finetuning.
      ckpt = self.config.train.ft_init_ckpt
      model(tf.keras.Input([None, None, 3]))
      utils.restore_tf2_ckpt(model, ckpt, exclude_layers=('_fc', 'optimizer'))


    scaled_lr = self.config.train.lr_base * (self.config.train.batch_size / 256.0)
    scaled_lr_min = self.config.train.lr_min * (self.config.train.batch_size / 256.0)
    learning_rate = utils.WarmupLearningRateSchedule(
        scaled_lr,
        steps_per_epoch=self._steps_per_epoch,
        decay_epochs=self.config.train.lr_decay_epoch,
        warmup_epochs=self.config.train.lr_warmup_epoch,
        decay_factor=self.config.train.lr_decay_factor,
        lr_decay_type=self.config.train.lr_sched,
        total_steps=self.total_steps,
        minimal_lr=scaled_lr_min)

    optimizer = self.get_optimizer_tf2(
        learning_rate, optimizer_name=self.config.train.optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=self.config.train.label_smoothing, from_logits=True),
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='acc_top1'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc_top5')
        ],
    )
    return model

  def train_multi_stage(self, fit_config: dict) -> None:
    """ Function to train efficientnetv2 in multi stages. """

    total_stages = self.config.train.stages
    ibase = self.config.data.ibase or (self.train_size / 2)

    if self.config.train.sched:
      ram_list = np.linspace(5, self.config.data.ram, total_stages)
      mixup_list = np.linspace(0, self.config.data.mixup_alpha, total_stages)
      cutmix_list = np.linspace(0, self.config.data.cutmix_alpha, total_stages)

    for stage in range(total_stages):
      ratio = float(stage + 1) / float(total_stages)
      start_epoch = int(
          float(stage) / float(total_stages) * self.config.train.epochs)
      end_epoch = int(ratio * self.config.train.epochs)
      _image_size = int(ibase + (self.train_size - ibase) * ratio)

      if self.config.train.sched:
        self.config.data.ram = ram_list[stage]
        self.config.data.mixup_alpha = mixup_list[stage]
        self.config.data.cutmix_alpha = cutmix_list[stage]

      model.fit(
          self.get_dataset(training=True, image_size=_image_size, config=self.config),
          initial_epoch=start_epoch,
          epochs=end_epoch,
          **fit_config
      )

  def evaluate(self) -> None:
    """ Evaluation helper function. """
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=60 * 60 * 24):
      model.load_weights(ckpt)
      eval_results = model.evaluate(
          self.get_dataset(training=False, image_size=eval_size, config=self.config),
          batch_size=self.config.eval.batch_size,
          steps=self.num_eval_images // self.config.eval.batch_size,
          callbacks=self.get_callbacks(istrain=False),
          verbose=2 if self.strategy == 'tpu' else 1,
      )

      try:
        current_epoch = int(os.path.basename(ckpt).split('-')[1])
      except IndexError:
        logging.info('%s has no epoch info: stop!', ckpt)
        break

      logging.info('Epoch: %d, total %d', current_epoch, self.config.train.epochs)
      if current_epoch >= self.config.train.epochs:
        break
    else:
      raise ValueError(f'Invalid mode {FLAGS.mode}')

  def run(self):
    """ Run function executes the training with give mode. """
    if self.mode != 'eval':
      # don't log spam if running on tpus
      verbose = 2 if self.strategy == 'tpu' else 1
      # initial setup for model fit config.
      _fit_config =  {
          'epochs': self.config.train.epochs,
          'steps_per_epoch': self.steps_per_epoch,
          'callbacks': [self.get_callbacks()],
          'verbose': verbose}

      # override fit_config with corresponding mode.
      if self.mode == 'traineval':
        _val_dataset = self.get_dataset(training=False, image_size=self.eval_size, config=self.config)
        _train_dataset = self.get_dataset(training=True, image_size=self.train_size, config=self.config)
        _fit_config.update({'train_data': _train_dataset, 
            'validation_data': _val_dataset, 
            'validation_steps' : self.num_eval_images // self.config.eval.batch_size,})

      elif self.mode == 'train':
        if not self.config.train.stages:
          _train_dataset = self.get_dataset(training=True, image_size=self.train_size, config=self.config)
          _fit_config.update({'train_data': _train_dataset})

        else:
          # train model on multi stages.
          self.train_multi_stage(_fit_config)

      # training the model with corresponding to mode.
      if not self.config.train.stages:
        model.fit(_fit_config)

    elif self.mode == 'eval':
      self.evaluate()


  def get_dataset(self, training: bool, image_size: Tuple[int, int, int], config: Config) -> Any:
    """A shared utility to get input dataset."""
    if training:
      return self.strategy.distribute_datasets_from_function(
          datasets.build_dataset_input(
              True, image_size, self.image_dtype, FLAGS.data_dir, train_split,
              config.data).distribute_dataset_fn(config.train.batch_size))
    return self.strategy.distribute_datasets_from_function(
        datasets.build_dataset_input(
            False, image_size, self.image_dtype, FLAGS.data_dir, eval_split,
            config.data).distribute_dataset_fn(config.eval.batch_size))


  def set_precision(self) -> None:
    """ Sets precision for the training session. """
    self.image_dtype = 'bfloat16' if self.strategy == 'tpu' else 'float16'
    precision = 'mixed_bfloat16' if self.strategy == 'tpu' else 'mixed_float16'
    logging.info('Training precision set to %s', precision)
    policy = tf.keras.mixed_precision.Policy(precision)
    tf.keras.mixed_precision.set_global_policy(policy)


  def get_config(self, base_config: Config) -> Config:
    """ Get training config. """
    config = copy.deepcopy(base_config)
    config.override(effnetv2_configs.get_model_config(FLAGS.model_name))
    config.override(datasets.get_dataset_config(FLAGS.dataset_cfg))
    config.override(FLAGS.hparam_str)
    config.model.num_classes = config.data.num_classes
    return config

  def get_strategy(self) -> None:
    """ Get ds_strategy for the training session. """

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
    return ds_strategy










if __name__ == '__main__':
  cflags.define_flags()
  app.run(main)
