# Lint as: python3
# Copyright 2020 Google Research. All Rights Reserved.
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
# =============================================================================
"""Postprocessing for anchor-based detection."""
from typing import List
from absl import logging
import tensorflow as tf

import anchors
import utils

T = tf.Tensor  # a shortcut for typing check.
MAX_BOXES_PER_IMAGE = 100


def clip_boxes(boxes: T, image_size: int) -> T:
  """Clip boxes to fit the image size."""
  image_size = utils.parse_image_size(image_size) * 2
  return tf.clip_by_value(boxes, [0], image_size)


def pad_zeros(inputs: T, max_output_size: int, indices=None) -> T:
  """Pad the inputs (or a subset specified by indices) to max_output_size."""
  if indices is not None:
    inputs = tf.gather(inputs, indices)
  padding_size = max_output_size - tf.shape(inputs)[0]
  if tf.rank(inputs) == 1:
    return tf.pad(inputs, [[0, padding_size]])
  if tf.rank(inputs) == 2:
    return tf.pad(inputs, [[0, padding_size], [0, 0]])
  raise ValueError('pad_zeros only support rank 1 or 2 inputs.')


def merge_class_box_level_outputs(params, cls_outputs, box_outputs) -> List[T]:
  """Concatenates class and box of all levels into one tensor."""
  cls_outputs_all, box_outputs_all = [], []
  batch_size = cls_outputs[0].shape[0]
  for level in range(0, params['max_level'] - params['min_level'] + 1):
    if params['data_format'] == 'channels_first':
      cls_outputs[level] = tf.transpose(cls_outputs[level], [0, 2, 3, 1])
      box_outputs[level] = tf.transpose(box_outputs[level], [0, 2, 3, 1])
    cls_outputs_all.append(
        tf.reshape(cls_outputs[level], [batch_size, -1, params['num_classes']]))
    box_outputs_all.append(tf.reshape(box_outputs[level], [batch_size, -1, 4]))
  return tf.concat(cls_outputs_all, 1), tf.concat(box_outputs_all, 1)


def topk_class_boxes(params, cls_outputs: T, box_outputs: T) -> List[T]:
  """Pick the topk class and box outputs."""
  batch_size = cls_outputs.shape[0]
  num_classes = params['num_classes']

  max_nms_inputs = params['nms_configs'].get('max_nms_inputs', 0)
  if max_nms_inputs > 0:
    # Prune anchors and detections to only keep max_nms_inputs.
    # Due to some issues, top_k is currently slow in graph model.
    logging.info('use max_nms_inputs for pre-nms topk.')
    cls_outputs_reshape = tf.reshape(cls_outputs, [batch_size, -1])
    _, cls_topk_indices = tf.math.top_k(cls_outputs_reshape,
                                        k=max_nms_inputs,
                                        sorted=False)
    indices = cls_topk_indices // num_classes
    classes = cls_topk_indices % num_classes
    cls_indices = tf.stack([indices, classes], axis=2)

    cls_outputs_topk = tf.gather_nd(cls_outputs, cls_indices, batch_dims=1)
    box_outputs_topk = tf.gather_nd(box_outputs,
                                    tf.expand_dims(indices, 2),
                                    batch_dims=1)
  else:
    logging.info('use max_reduce for pre-nms topk.')
    # Keep all anchors, but for each anchor, just keep the max probablity for
    # each class.
    cls_outputs_idx = tf.math.argmax(cls_outputs, axis=-1, output_type=tf.int32)
    num_anchors = cls_outputs.shape[1]

    classes = cls_outputs_idx
    indices = tf.tile(tf.expand_dims(tf.range(num_anchors), axis=0),
                      [batch_size, 1])
    cls_outputs_topk = tf.reduce_max(cls_outputs, -1)
    box_outputs_topk = box_outputs

  return cls_outputs_topk, box_outputs_topk, classes, indices


def pre_nms(params, cls_outputs, box_outputs) -> List[T]:
  """Detection post processing before nms.

  It takes the multi-level class and box predictions from network, merge them
  into unified tensors, and compute boxes, scores, and classes.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level
      of logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors].

  Returns:
    A tuple of (boxes, scores, classes).
  """
  cls_outputs, box_outputs = merge_class_box_level_outputs(
      params, cls_outputs, box_outputs)
  cls_outputs, box_outputs, classes, indices = topk_class_boxes(
      params, cls_outputs, box_outputs)

  # get boxes by apply bounding box regression to anchors.
  eval_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                 params['num_scales'], params['aspect_ratios'],
                                 params['anchor_scale'], params['image_size'])
  anchor_boxes = tf.gather(eval_anchors.boxes, indices)
  boxes = anchors.decode_box_outputs_tf(box_outputs, anchor_boxes)

  # convert logits to scores.
  scores = tf.math.sigmoid(cls_outputs)
  return boxes, scores, classes


def nms(params, boxes: T, scores: T, classes: T, padded: bool) -> List[T]:
  """Non-maximum suppression.

  Args:
    params: a dict of parameters.
    boxes: a tensor with shape [N, 4], where N is the number of boxes.
    scores: a tensor with shape [N].
    classes: a tensor with shape [N].

  Returns:
    A tuple (boxes, scores, classes, valid_lens), where valid_lens is a scalar
    denoting the valid length of boxes/scores/classes outputs.
  """
  logging.info('performing per-sample nms')
  nms_configs = params['nms_configs'] or dict()
  method = nms_configs.get('method', None)
  max_output_size = nms_configs.get('max_output_size', MAX_BOXES_PER_IMAGE)

  if method == 'hard' or not method:
    # hard nms.
    sigma = 0.0
    iou_thresh = nms_configs.get('iou_thresh', 0.5)
    score_thresh = nms_configs.get('score_thresh', float('-inf'))
  elif method == 'gaussian':
    sigma = nms_configs.get('sigma', 0.5)
    iou_thresh = nms_configs.get('iou_thresh', 0.3)
    score_thresh = nms_configs.get('score_thresh', 0.001)
  else:
    raise ValueError('Inference has invalid nms method {}'.format(method))

  # TF API's sigma is twice as the paper's value, so here we divide it by 2:
  # https://github.com/tensorflow/tensorflow/issues/40253.
  nms_top_idx, nms_scores, nms_valid_lens = tf.raw_ops.NonMaxSuppressionV5(
      boxes=boxes,
      scores=scores,
      max_output_size=max_output_size,
      iou_threshold=iou_thresh,
      score_threshold=score_thresh,
      soft_nms_sigma=(sigma / 2),
      pad_to_max_output_size=padded)
  nms_boxes = tf.gather(boxes, nms_top_idx)
  nms_classes = tf.cast(tf.gather(classes, nms_top_idx) + 1, tf.float32)
  return nms_boxes, nms_scores, nms_classes, nms_valid_lens


def postprocess_global(params, cls_outputs, box_outputs, img_scales=None):
  """Post processing with global NMS.

  A fast but less accurate version of NMS. The idea is to treat the scores for
  different classes in a unified way, and perform NMS globally for all classes.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level
      of logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors].
    img_scales: scaling factor or the final image and bounding boxes.

  Returns:
    A tuple of batch level (boxes, scores, classess, valid_len) after nms.
  """
  boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)

  # A list of batched boxes, scores, and classes.
  nms_boxes_bs, nms_scores_bs, nms_classes_bs, nms_valid_len_bs = [], [], [], []
  batch_size = boxes.shape[0]
  for i in range(batch_size):
    padded = batch_size > 1  # only pad if batch size > 1 for simplicity.
    nms_boxes, nms_scores, nms_classes, nms_valid_len = nms(
        params, boxes[i], scores[i], classes[i], padded)
    nms_boxes = clip_boxes(nms_boxes, params['image_size'])

    nms_boxes_bs.append(nms_boxes)
    nms_scores_bs.append(nms_scores)
    nms_classes_bs.append(nms_classes)
    nms_valid_len_bs.append(nms_valid_len)

  nms_boxes_bs = tf.stack(nms_boxes_bs)
  nms_scores_bs = tf.stack(nms_scores_bs)
  nms_classes_bs = tf.stack(nms_classes_bs)
  nms_valid_len_bs = tf.stack(nms_valid_len_bs)
  if img_scales is not None:
    scales = tf.expand_dims(tf.expand_dims(img_scales, -1), -1)
    nms_boxes_bs = nms_boxes_bs * scales
  return nms_boxes_bs, nms_scores_bs, nms_classes_bs, nms_valid_len_bs


def postprocess_per_class(params, cls_outputs, box_outputs, img_scales=None):
  """Post processing with per class NMS.

  An accurate but relatively slow version of NMS. The idea is to perform NMS for
  each class, and then combine them.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level
      of logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors].
    img_scales: scaling factor or the final image and bounding boxes.

  Returns:
    A tuple of batch level (boxes, scores, classess, valid_len) after nms.
  """
  boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)

  nms_boxes_bs, nms_scores_bs, nms_classes_bs, nms_valid_len_bs = [], [], [], []
  batch_size = boxes.shape[0]
  for i in range(batch_size):
    boxes_i, scores_i, classes_i = boxes[i], scores[i], classes[i]
    nms_boxes_cls, nms_scores_cls, nms_classes_cls = [], [], []
    nms_valid_len_cls = []
    for cid in range(params['num_classes']):
      indices = tf.squeeze(tf.where(classes_i == cid))
      if indices.shape.as_list() == 0:
        continue
      classes_cls = tf.gather(classes_i, indices)
      boxes_cls = tf.gather(boxes_i, indices)
      scores_cls = tf.gather(scores_i, indices)

      nms_boxes, nms_scores, nms_classes, nms_valid_len = nms(
          params, boxes_cls, scores_cls, classes_cls, False)
      nms_boxes_cls.append(nms_boxes)
      nms_scores_cls.append(nms_scores)
      nms_classes_cls.append(nms_classes)
      nms_valid_len_cls.append(nms_valid_len)

    nms_boxes_cls = tf.concat(nms_boxes_cls, 0)
    nms_scores_cls = tf.concat(nms_scores_cls, 0)
    nms_classes_cls = tf.concat(nms_classes_cls, 0)
    nms_valid_len_cls = tf.concat(nms_valid_len_cls, 0)
    # get top detections and pad to fix size.
    max_output_size = params['nms_configs'].get('max_output_size', 100)
    _, indices = tf.math.top_k(nms_scores_cls, k=max_output_size, sorted=True)

    nms_boxes_bs.append(pad_zeros(nms_boxes_cls, max_output_size, indices))
    nms_scores_bs.append(pad_zeros(nms_scores_cls, max_output_size, indices))
    nms_classes_bs.append(pad_zeros(nms_classes_cls, max_output_size, indices))
    nms_valid_len_bs.append(indices.shape[0])

  nms_scores_bs = tf.stack(nms_scores_bs)
  nms_classes_bs = tf.stack(nms_classes_bs)
  nms_boxes_bs = tf.stack(nms_boxes_bs)
  nms_valid_len_bs = tf.stack(nms_valid_len_bs)
  if img_scales:
    scales = tf.expand_dims(tf.expand_dims(img_scales, -1), -1)
    nms_boxes_bs = nms_boxes_bs * scales
  return nms_boxes_bs, nms_scores_bs, nms_classes_bs, nms_valid_len_bs


def generate_detections(params, cls_outputs, box_outputs, img_scales, img_ids):
  """A legacy interface for generating [id, x, y, w, h, score, class]."""
  nms_boxes_bs, nms_scores_bs, nms_classes_bs, _ = postprocess_per_class(
      params, cls_outputs, box_outputs, img_scales)

  img_ids_bs = tf.cast(tf.expand_dims(img_ids, -1), nms_scores_bs.dtype)
  detections_bs = [
      img_ids_bs * tf.ones_like(nms_scores_bs),
      nms_boxes_bs[:, :, 1],
      nms_boxes_bs[:, :, 0],
      nms_boxes_bs[:, :, 3] - nms_boxes_bs[:, :, 1],
      nms_boxes_bs[:, :, 2] - nms_boxes_bs[:, :, 0],
      nms_scores_bs,
      nms_classes_bs,
  ]
  return tf.stack(detections_bs, axis=-1, name='detnections')
