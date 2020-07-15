from keras.efficientdet_keras import EfficientDetNet
import tensorflow as tf

NUM_CLASSES = 3
class EfficientDetSegmentation(EfficientDetNet):
  def build(self, input_shape):
    self.con2d_ts = []
    self.bns = []
    for _ in range(self.config.max_level-self.config.min_level):
      self.con2d_ts.append(tf.keras.layers.Conv2DTranspose(self.config.fpn_num_filters, 3, strides=2, padding='same', use_bias=False))
      self.bns.append(tf.keras.layers.BatchNormalization(momentum=self.config.momentum))

    self.last = tf.keras.layers.Conv2DTranspose(
      NUM_CLASSES, 3, strides=2,
      padding='same')
    super().build(input_shape)

  def call(self, inputs, training):
    config = self.config
    # call backbone network.
    self.backbone(inputs, training=training, features_only=True)
    all_feats = [
      inputs,
      self.backbone.endpoints['reduction_1'],
      self.backbone.endpoints['reduction_2'],
      self.backbone.endpoints['reduction_3'],
      self.backbone.endpoints['reduction_4'],
      self.backbone.endpoints['reduction_5'],
    ]
    feats = all_feats[config.min_level:config.max_level + 1]

    # Build additional input features that are not from backbone.
    for resample_layer in self.resample_layers:
      feats.append(resample_layer(feats[-1], training))

    # call feature network.
    feats = self.fpn_cells(feats, training)

    x = feats[-1]
    skips = list(reversed(feats[:-1]))

    for con2d_t, bn, skip in zip(self.con2d_ts, self.bns, skips):
      x = con2d_t(x)
      x = bn(x)
      x = tf.nn.relu(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])

    # This is the last layer of the model
    return self.last(x) # 64x64 -> 128x128

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

import tensorflow_datasets as tfds
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (512, 512))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (512, 512))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 8
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


model = EfficientDetSegmentation('efficientdet-d0')
model.build((BATCH_SIZE,512,512,3))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[])

model.save_weights("./test/segmentation")


print(create_mask(model(tf.ones((1, 512, 512, 3)), False)))
