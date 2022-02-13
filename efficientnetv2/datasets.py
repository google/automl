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
"""Datasets input pipeline."""
import copy
import functools
import os
from absl import logging
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import hparams
import preprocessing
ds_register = functools.partial(hparams.register, prefix='ds:')


# Input pipelines are slightly different (with regards to shuffling and
# preprocessing) between training and evaluation.
def build_dataset_input(is_training, image_size, image_dtype, data_dir, split,
                        data_config):
  """Generate input for training and eval."""
  ds_cls = get_dataset_class(data_config.ds_name)
  return ds_cls(
      is_training=is_training,
      data_dir=data_dir,
      split=split,
      cache=data_config.cache,
      image_size=image_size,
      image_dtype=image_dtype,
      augname=data_config.augname,
      mixup_alpha=data_config.mixup_alpha,
      cutmix_alpha=data_config.cutmix_alpha,
      ra_num_layers=data_config.ra_num_layers,
      ra_magnitude=data_config.ram)


def build_image_serving_input_fn(image_size, batch_size=None):
  """Builds a serving input fn for raw images."""

  def _image_serving_input_fn():
    """Serving input fn for raw images."""

    def _preprocess_image(image_bytes):
      """Preprocess a single raw image."""
      image = preprocessing.preprocess_image(
          image_bytes,
          is_training=False,
          image_size=image_size)
      return image

    image_bytes_list = tf.placeholder(
        shape=[batch_size],
        dtype=tf.string,
    )
    images = tf.map_fn(
        _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
    return tf.estimator.export.ServingInputReceiver(
        images, {'image_bytes': image_bytes_list})
  return _image_serving_input_fn


class ImageNetInput():
  """Generates ImageNet input_fn from a series of TFRecord files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """
  cfg = hparams.Config(
      data_dir=None,
      num_classes=1000,
      multiclass=False,
      tfds_split=None,
      splits=dict(
          train=dict(
              num_images=1_256_144, files='train*', slice=slice(20, None)),
          minival=dict(num_images=25_021, files='train*', slice=slice(0, 20)),
          eval=dict(num_images=50_000, files='val*', slice=slice(0, None)),
          trainval=dict(
              num_images=1_281_167, files='train*', slice=slice(0, None)),
      ),
  )

  def __init__(self,
               is_training,
               image_dtype=False,
               data_dir=None,
               split=None,
               image_size=224,
               cache=False,
               augname=None,
               mixup_alpha=None,
               cutmix_alpha=None,
               ra_num_layers=None,
               ra_magnitude=None,
               transpose_image=False,
               debug=False):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      image_dtype: iamge dtype. If None, use tf.float32.
      data_dir: `str` for the directory of the training and validation data;
      split: `str`, dataset split, common values {train, eval, test, traineval}.
      image_size: `int` for image size (both width and height).
      cache: if true, fill the dataset by repeating from its cache.
      augname: `string` that is the name of the augmentation method
          to apply to the image. `autoaugment` if AutoAugment is to be used or
          `randaugment` if RandAugment is to be used. If the value is `None` no
          no augmentation method will be applied applied. See autoaugment.py
          for more details.
      mixup_alpha: float to control the strength of Mixup regularization, set
          to 0.0 to disable.
      cutmix_alpha: float to control cutmix, set to 0.0 or None to disable.
      ra_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      ra_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
      transpose_image: Whether to transpose the image. Useful for the "double
        transpose" trick for improved input throughput.
      debug: bool, If true, use deterministic behavior and add orig_image.
    """
    self.is_training = is_training
    self.image_dtype = image_dtype or tf.float32
    self.image_size = image_size
    self.augname = augname
    self.mixup_alpha = mixup_alpha
    self.cutmix_alpha = cutmix_alpha
    self.ra_num_layers = ra_num_layers
    self.ra_magnitude = ra_magnitude
    self.split = split or ('train' if is_training else 'eval')

    self.data_dir = data_dir or self.cfg.data_dir
    self.cache = cache
    self.transpose_image = transpose_image

    # for input pipeline performance.
    self.file_buffer_size_m = None
    self.shuffle_size_k = 128

    self.debug = debug
    self.orig_image = False
    # randomness
    self.shuffle_files = False if debug else True
    self.shuffle_seed = 1111 if debug else None

  def image_preprocessing(self, image):
    return preprocessing.preprocess_image(
        image,
        image_size=self.image_size,
        is_training=self.is_training,
        image_dtype=self.image_dtype,
        augname=self.augname,
        ra_num_layers=self.ra_num_layers,
        ra_magnitude=self.ra_magnitude)

  @property
  def split_info(self):
    return self.cfg.splits[self.split]

  def set_shapes(self, batch_size, features, labels):
    """Statically set the batch_size dimension."""
    features['image'].set_shape(features['image'].get_shape().merge_with(
        tf.TensorShape([batch_size, None, None, None])))
    labels['label'].set_shape(labels['label'].get_shape().merge_with(
        tf.TensorShape([batch_size, None])))
    return features, labels

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3], self.image_dtype)

  def cutmix_mask(self, alpha, h, w):
    """Returns image mask for CutMix."""
    r_x = tf.random.uniform([], 0, w, tf.int32)
    r_y = tf.random.uniform([], 0, h, tf.int32)

    area = tf.distributions.Beta(alpha, alpha).sample()
    patch_ratio = tf.cast(tf.math.sqrt(1 - area), tf.float32)
    r_w = tf.cast(patch_ratio * tf.cast(w, tf.float32), tf.int32)
    r_h = tf.cast(patch_ratio * tf.cast(h, tf.float32), tf.int32)
    bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
    bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
    bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
    bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

    # Create the binary mask.
    pad_left = bbx1
    pad_top = bby1
    pad_right = tf.maximum(w - bbx2, 0)
    pad_bottom = tf.maximum(h - bby2, 0)
    r_h = bby2 - bby1
    r_w = bbx2 - bbx1

    mask = tf.pad(
        tf.ones((r_h, r_w)),
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
        mode='CONSTANT',
        constant_values=0)
    mask.set_shape((h, w))
    return mask[..., None]  # Add channel dim.

  def cutmix(self, image, label, mask):
    """Applies CutMix regularization to a batch of images and labels.

    Reference: https://arxiv.org/pdf/1905.04899.pdf

    Arguments:
      image: a Tensor of batched images.
      label: a Tensor of batched labels.
      mask: a Tensor of batched masks.

    Returns:
      A new dict of features with updated images and labels with the same
      dimensions as the input with CutMix regularization applied.
    """
    # actual area of cut & mix pixels
    mix_area = tf.reduce_sum(mask) / tf.cast(tf.size(mask), mask.dtype)
    mask = tf.cast(mask, image.dtype)
    mixed_image = (1. - mask) * image + mask * image[::-1]
    mix_area = tf.cast(mix_area, label.dtype)
    mixed_label = (1. - mix_area) * label + mix_area * label[::-1]

    return mixed_image, mixed_label

  def mixup(self, batch_size, alpha, image, label):
    """Applies Mixup regularization to a batch of images and labels.

    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
      Mixup: Beyond Empirical Risk Minimization.
      ICLR'18, https://arxiv.org/abs/1710.09412

    Arguments:
      batch_size: The input batch size for images and labels.
      alpha: Float that controls the strength of Mixup regularization.
      image: a Tensor of batched images.
      label: a Tensor of batch labels.

    Returns:
      A new dict of features with updated images and labels with the same
      dimensions as the input with Mixup regularization applied.
    """
    mix_weight = tf.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
    img_weight = tf.cast(
        tf.reshape(mix_weight, [batch_size, 1, 1, 1]), image.dtype)
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    image = image * img_weight + image[::-1] * (1. - img_weight)
    label_weight = tf.cast(mix_weight, label.dtype)
    label = label * label_weight + label[::-1] * (1 - label_weight)
    return image, label

  def mixing(self, batch_size, mixup_alpha, cutmix_alpha, features, labels):
    """Applies mixing regularization to a batch of images and labels.

    Arguments:
      batch_size: The input batch size for images and labels.
      mixup_alpha: Float that controls the strength of Mixup regularization.
      cutmix_alpha: FLoat that controls the strenght of Cutmix regularization.
      features: a dict of batched images.
      labels: a dict of batched labels.

    Returns:
      A new dict of features with updated images and labels with the same
      dimensions as the input.
    """
    image, label = features['image'], labels['label']
    if mixup_alpha and cutmix_alpha:
      # split the batch half-half, and aplly mixup and cutmix for each half.
      bs = batch_size // 2
      img1, lab1 = self.mixup(bs, mixup_alpha, image[:bs], label[:bs])
      img2, lab2 = self.cutmix(image[bs:], label[bs:],
                               features['cutmix_mask'][bs:])
      features['image'] = tf.concat([img1, img2], axis=0)
      labels['label'] = tf.concat([lab1, lab2], axis=0)
    elif mixup_alpha:
      features['image'], labels['label'] = self.mixup(batch_size, mixup_alpha,
                                                      image, label)
    elif cutmix_alpha:
      features['image'], labels['label'] = self.cutmix(
          image, label, features['cutmix_mask'])
    return features, labels

  def dataset_parser(self, value):
    """See base class."""
    if self.data_dir == 'null' or not self.data_dir:
      labels = tf.constant(0., tf.float32, (self.cfg.num_classes,))
      return {'image': value}, {'label': labels}

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image = self.image_preprocessing(image_bytes)
    # The labels will be in range [1,1000], 0 is reserved for background
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)
    label -= 1  # get rid of the background.
    onehot_label = tf.one_hot(label, self.cfg.num_classes)
    features = {'image': image}
    labels = {'label': onehot_label}
    if self.cutmix_alpha:
      features['cutmix_mask'] = self.cutmix_mask(  #
          self.cutmix_alpha, self.image_size, self.image_size)
    if self.debug and self.orig_image:
      features['orig_image'] = tf.image.decode_image(image_bytes)
    return features, labels

  def fetch_dataset(self, filename):
    buffer_size = (self.file_buffer_size_m or 8) * 1024 * 1024  # 8 MiB
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if self.data_dir == 'null' or not self.data_dir:
      logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    filenames = tf.io.gfile.glob(
        os.path.join(self.data_dir, self.split_info['files']))
    filenames = sorted(filenames)[self.split_info['slice']]
    for f in filenames[:5]:
      logging.info('datafiles: %s', f)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = dataset.shard(num_hosts, index)

    # file-level shuffle
    if self.is_training and self.shuffle_files:
      num_files_per_shard = (len(filenames) + num_hosts - 1) // num_hosts
      dataset = dataset.shuffle(num_files_per_shard, seed=self.shuffle_seed)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    # Read the data from disk in parallel
    dataset = dataset.interleave(
        self.fetch_dataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=self.debug)

    if self.is_training and self.cache:
      dataset = dataset.cache().shuffle(
          self.shuffle_size_k * 1024, seed=self.shuffle_seed).repeat()
    else:
      dataset = dataset.shuffle(
          self.shuffle_size_k * 1024, seed=self.shuffle_seed)
    return dataset

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.estimator.tpu.RunConfig for details.
    batch_size = params['batch_size']

    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1

    return self._input_fn(batch_size, current_host, num_hosts)

  def _input_fn(self, batch_size, current_host, num_hosts):
    """Creates a dataset for the specified host."""
    dataset = self.make_source_dataset(current_host, num_hosts)
    dataset = dataset.map(
        self.dataset_parser,
        num_parallel_calls=tf.data.AUTOTUNE).batch(
            batch_size, drop_remainder=True)

    # Apply Mixup
    if self.is_training and (self.mixup_alpha or self.cutmix_alpha):
      dataset = dataset.map(
          functools.partial(self.mixing, batch_size, self.mixup_alpha,
                            self.cutmix_alpha),
          num_parallel_calls=tf.data.AUTOTUNE)

    # Assign static batch size dimension
    dataset = dataset.map(
        functools.partial(self.set_shapes, batch_size),
        num_parallel_calls=tf.data.AUTOTUNE)

    def transpose_image(features):
      # NHWC -> HWCN
      features['image'] = tf.transpose(features['image'], [1, 2, 3, 0])
      return features

    if self.transpose_image:
      dataset = dataset.map(
          lambda features, labels: (transpose_image(features), labels),
          num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.deterministic = self.debug
    options.threading.max_intra_op_parallelism = 1
    options.threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)
    return dataset

  def distribute_dataset_fn(self, global_batch_size):
    """Dataset for tf.distribute.Strategy.distribute_datasets_from_function."""

    def dataset_fn(input_context):
      return self._input_fn(
          input_context.get_per_replica_batch_size(global_batch_size),
          input_context.input_pipeline_id, input_context.num_input_pipelines)

    return dataset_fn


class ImageNet21kInput(ImageNetInput):
  """Generates input_fn from ImageNet21k TFRecord files."""
  cfg = copy.deepcopy(ImageNetInput.cfg)
  cfg.update(
      dict(
          data_dir=None,
          num_classes=21843,
          multiclass=True,
          splits=dict(
              train=dict(
                  num_images=12_720_275,
                  files='imagenet21k*',
                  slice=slice(20, None)),
              minival=dict(
                  num_images=25089, files='imagenet21k*', slice=slice(16, 20)),
              eval=dict(
                  num_images=100357, files='imagenet21k*', slice=slice(0, 16)),
          ),
      ))

  def dataset_parser(self, value):
    """See base class."""
    if self.data_dir == 'null' or not self.data_dir:
      values = tf.constant([2, 3], tf.int64)
      fake_labels = tf.sparse.to_dense(
          tf.sparse.SparseTensor(
              tf.expand_dims(values, -1), tf.ones_like(values),
              [self.cfg.num_classes]))
      return {'image': value}, {'label': fake_labels}

    keys_to_features = {
        'id':
            tf.io.FixedLenFeature([], tf.string),
        'image':
            tf.io.FixedLenFeature([], tf.string),
        'labels':
            # tf.io.VarLenFeature(tf.int64),
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }

    parsed = tf.io.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image'], shape=[])
    image = self.image_preprocessing(image_bytes)
    labels = parsed['labels']
    labels = tf.sparse.SparseTensor(
        tf.expand_dims(labels, -1), tf.ones_like(labels),
        [self.cfg.num_classes])
    labels = tf.sparse.to_dense(labels)
    features = {'image': image}
    labels = {'label': labels}
    if self.debug:
      features['orig_image'] = tf.image.decode_image(image_bytes)
    return features, labels


class CIFAR10Input(ImageNetInput):
  """Cifar10 input from tfds."""
  cfg = copy.deepcopy(ImageNetInput.cfg)
  cfg.update(
      dict(
          try_gcs=True,
          data_dir=None,
          num_classes=10,
          multiclass=False,
          tfds_name='cifar10',
          splits=dict(
              train=dict(num_images=50000, tfds_split='train', slice=''),
              minival=dict(num_images=10000, tfds_split='test', slice=''),
              eval=dict(num_images=10000, tfds_split='test', slice=''),
          )))

  def preprocess(self, features):
    """The preprocessing function."""
    image = self.image_preprocessing(features['image'])
    new_features = {'image': image}
    if self.debug:
      new_features['orig_image'] = features['image']
    new_label = {'label': tf.one_hot(features['label'], self.cfg.num_classes)}
    return new_features, new_label

  def _input_fn(self, batch_size, current_host, num_hosts):
    logging.info('use tfds: %s[%s]', self.cfg.tfds_name,
                 self.cfg.splits[self.split]['tfds_split'])
    ds = tfds.load(
        self.cfg.tfds_name, split=self.cfg.splits[self.split]['tfds_split'], try_gcs=self.cfg.try_gcs)
    ds = ds.shard(num_hosts, current_host)
    if self.is_training:
      if self.cache:
        ds = ds.cache().shuffle(1024 * 16, seed=self.shuffle_seed).repeat()
      else:
        ds = ds.shuffle(self.shuffle_size_k * 1024, seed=self.shuffle_seed)
      ds = ds.repeat()

    ds = ds.map(
        self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(1)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(1)

    options = tf.data.Options()
    options.autotune.enabled = True
    return ds.with_options(options)


class CIFAR100Input(CIFAR10Input):
  """Cifar100 input from tfds."""
  cfg = copy.deepcopy(CIFAR10Input.cfg)
  cfg.update(
      dict(
          num_classes=100,
          tfds_name='cifar100',
          splits=dict(
              train=dict(num_images=50000, tfds_split='train', slice=''),
              minival=dict(num_images=10000, tfds_split='test', slice=''),
              eval=dict(num_images=10000, tfds_split='test', slice=''),
          )))


class FlowersInput(CIFAR10Input):
  """Flowers input from tfds."""
  cfg = copy.deepcopy(CIFAR10Input.cfg)
  cfg.update(
      dict(
          num_classes=102,
          tfds_name='oxford_flowers102',
          splits=dict(
              train=dict(
                  num_images=1020, tfds_split='train+validation', slice=''),
              minival=dict(num_images=1020, tfds_split='validation', slice=''),
              eval=dict(num_images=6149, tfds_split='test', slice=''),
          )))


class TFFlowersInput(CIFAR10Input):
  """TFFlowers input from tfds gcs."""
  cfg = copy.deepcopy(CIFAR10Input.cfg)
  cfg.update(
      dict(
          num_classes=5,
          tfds_name='tf_flowers',
          try_gcs=True,
          splits=dict(
              train=dict(num_images=2569, tfds_split='train[:70%]'),
              minival=dict(num_images=1101, tfds_split='train[30%:]'),
              eval=dict(num_images=1101, tfds_split='train[30%:]'),
          )))


class CarsInput(CIFAR10Input):
  """Car input from tfds."""
  cfg = copy.deepcopy(CIFAR10Input.cfg)
  cfg.update(
      dict(
          num_classes=196,
          tfds_name='cars196',
          splits=dict(
              train=dict(num_images=8_144, tfds_split='train', slice=''),
              minival=dict(num_images=8_041, tfds_split='test', slice=''),
              eval=dict(num_images=8_041, tfds_split='test', slice=''),
          )))


class ImageNetTfdsInput(CIFAR10Input):
  """ImageNet TFDS input from tfds."""
  cfg = copy.deepcopy(CIFAR10Input.cfg)
  cfg.update(
      dict(
          data_dir=None,
          num_classes=1000,
          multiclass=False,
          tfds_name='imagenet2012',
          splits=dict(
              train=dict(num_images=1_256_144, tfds_split='train[:98%]'),
              minival=dict(num_images=25_021, tfds_split='train[2%:]'),
              eval=dict(num_images=50_000, tfds_split='validation'),
              trainval=dict(num_images=1_281_167, tfds_split='train'),
          )))


def get_dataset_class(ds_name):
  return {
      'imagenet': ImageNetInput,
      'imagenet21k': ImageNet21kInput,
      'imagenettfds': ImageNetTfdsInput,
      'cifar10': CIFAR10Input,
      'cifar100': CIFAR100Input,
      'flowers': FlowersInput,
      'tfflowers': TFFlowersInput,
      'cars': CarsInput,
  }[ds_name]


################# Dataset training configs ####################
@ds_register
class ImageNet():
  """ImageNet train/eval configs."""
  cfg = hparams.Config(
      data=dict(
          ds_name='imagenet',
          multiclass=False,
      ),
      train=dict(
          epochs=350,
          lr_base=0.016,
          lr_warmup_epoch=5,
          lr_sched='exponential',
          label_smoothing=0.1,
      ),
      eval=dict(
          batch_size=8,
      ),
  )


@ds_register
class ImageNet21k():
  """ImageNet21k train/eval configs."""
  cfg = hparams.Config(
      model=dict(
          dropout_rate=0.000001,
          survival_prob=1.0,
      ),
      data=dict(
          ds_name='imagenet21k',
          multiclass=True,
          augname=None,  # Disable all augmentation and mixup.
          mixup_alpha=0,
          cutmix_alpha=0,
      ),
      train=dict(
          epochs=60,
          lr_base=0.008,
          lr_warmup_epoch=1,
          lr_sched='cosine',
          label_smoothing=0.0,
          isize=224,
          stages=0,  # do not apply staged training.
          sched=False,
      ),
      eval=dict(
          batch_size=128,
          isize=224,
      ),
  )


@ds_register
class ImagenetFt(ImageNet):
  """Finetune imagenet configs."""
  # Finetune should have less regularization due to the limited training steps.
  cfg = hparams.Config(
      model=dict(
          dropout_rate=0.000001,
          survival_prob=0.8,
      ),
      train=dict(
          batch_size=512,
          stages=0,
          epochs=15,
          optimizer='rmsprop',
          lr_sched='constant',
          lr_base=0.0005,
          lr_warmup_epoch=1,
          ema_decay=0.9996,
          weight_decay=1e-5,
          label_smoothing=0.1,
          min_steps=10000,
          isize=1.0,
      ),
      data=dict(
          ds_name='imagenettfds',
          augname='ft',
          mixup_alpha=0,
          cutmix_alpha=0,
      ),
  )


@ds_register
class Cifar10Ft(ImagenetFt):
  """Finetune cifar10 configs."""
  cfg = copy.deepcopy(ImagenetFt.cfg)
  cfg.data.ds_name = 'cifar10'


@ds_register
class Cifar100Ft(Cifar10Ft):
  """Finetune cifar100 configs."""
  cfg = copy.deepcopy(Cifar10Ft.cfg)
  cfg.data.override(dict(ds_name='cifar100'))


@ds_register
class FlowersFt(Cifar10Ft):
  """Finetune flower configs."""
  cfg = copy.deepcopy(Cifar10Ft.cfg)
  cfg.data.override(dict(ds_name='flowers'))

@ds_register
class TFFlowersFt(Cifar10Ft):
  """Finetune tfflower configs."""
  cfg = copy.deepcopy(Cifar10Ft.cfg)
  cfg.data.override(dict(ds_name='tfflowers'))

@ds_register
class CarsFt(Cifar10Ft):
  """Finetune car configs."""
  cfg = copy.deepcopy(Cifar10Ft.cfg)
  cfg.data.override(dict(ds_name='cars'))


################################################################################
def get_dataset_config(name, prefix='ds:'):
  """Main entry for dataset config, e.g., ImageNet or Cifar10Ft."""
  cfg = hparams.lookup(name, prefix).cfg
  cfg.data.update(get_dataset_class(cfg.data.ds_name).cfg)
  return cfg
