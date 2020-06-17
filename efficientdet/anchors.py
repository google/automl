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
# ==============================================================================
"""Anchor definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import utils
from object_detection import argmax_matcher
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import region_similarity_calculator
from object_detection import target_assigner

# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0

# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5

# The maximum number of (anchor,class) pairs to keep for non-max suppression.
MAX_DETECTION_POINTS = 5000

# The maximum number of detections per image.
MAX_DETECTIONS_PER_IMAGE = 100

# The minimal score threshold.
MIN_SCORE_THRESH = 0.4


def sigmoid(x):
  """Sigmoid function for use with Numpy for CPU evaluation."""
  return 1 / (1 + np.exp(-x))


def decode_box_outputs(rel_codes, anchors):
  """Transforms relative regression coordinates to absolute positions.

  Network predictions are normalized and relative to a given anchor; this
  reverses the transformation and outputs absolute coordinates for the input
  image.

  Args:
    rel_codes: box regression targets.
    anchors: anchors on all feature levels.
  Returns:
    outputs: bounding boxes.

  """
  ycenter_a = (anchors[0] + anchors[2]) / 2
  xcenter_a = (anchors[1] + anchors[3]) / 2
  ha = anchors[2] - anchors[0]
  wa = anchors[3] - anchors[1]
  ty, tx, th, tw = rel_codes

  w = np.exp(tw) * wa
  h = np.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return np.column_stack([ymin, xmin, ymax, xmax])


def decode_box_outputs_tf(rel_codes, anchors):
  """Transforms relative regression coordinates to absolute positions.

  Network predictions are normalized and relative to a given anchor; this
  reverses the transformation and outputs absolute coordinates for the input
  image.

  Args:
    rel_codes: box regression targets.
    anchors: anchors on all feature levels.
  Returns:
    outputs: bounding boxes.
  """
  ycenter_a = (anchors[..., 0] + anchors[..., 2]) / 2
  xcenter_a = (anchors[..., 1] + anchors[..., 3]) / 2
  ha = anchors[..., 2] - anchors[..., 0]
  wa = anchors[..., 3] - anchors[..., 1]
  ty, tx, th, tw = tf.unstack(rel_codes, num=4, axis=-1)

  w = tf.math.exp(tw) * wa
  h = tf.math.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return tf.stack([ymin, xmin, ymax, xmax], axis=-1)


def diou_nms(dets, iou_thresh=None):
  """DIOU non-maximum suppression.

  diou = iou - square of euclidian distance of box centers
     / square of diagonal of smallest enclosing bounding box

  Reference: https://arxiv.org/pdf/1911.08287.pdf

  Args:
    dets: detection with shape (num, 5) and format [x1, y1, x2, y2, score].
    iou_thresh: IOU threshold,

  Returns:
    numpy.array: Retained boxes.
  """
  iou_thresh = iou_thresh or 0.5
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  center_x = (x1 + x2) / 2
  center_y = (y1 + y2) / 2

  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    intersection = w * h
    iou = intersection / (areas[i] + areas[order[1:]] - intersection)

    smallest_enclosing_box_x1 = np.minimum(x1[i], x1[order[1:]])
    smallest_enclosing_box_x2 = np.maximum(x2[i], x2[order[1:]])
    smallest_enclosing_box_y1 = np.minimum(y1[i], y1[order[1:]])
    smallest_enclosing_box_y2 = np.maximum(y2[i], y2[order[1:]])

    square_of_the_diagonal = (
        (smallest_enclosing_box_x2 - smallest_enclosing_box_x1)**2 +
        (smallest_enclosing_box_y2 - smallest_enclosing_box_y1)**2)

    square_of_center_distance = ((center_x[i] - center_x[order[1:]])**2 +
                                 (center_y[i] - center_y[order[1:]])**2)

    # Add 1e-10 for numerical stability.
    diou = iou - square_of_center_distance / (square_of_the_diagonal  + 1e-10)
    inds = np.where(diou <= iou_thresh)[0]
    order = order[inds + 1]
  return dets[keep]


def hard_nms(dets, iou_thresh=None):
  """The basic hard non-maximum suppression.

  Args:
    dets: detection with shape (num, 5) and format [x1, y1, x2, y2, score].
    iou_thresh: IOU threshold,

  Returns:
    numpy.array: Retained boxes.
  """
  iou_thresh = iou_thresh or 0.5
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    intersection = w * h
    overlap = intersection / (areas[i] + areas[order[1:]] - intersection)

    inds = np.where(overlap <= iou_thresh)[0]
    order = order[inds + 1]

  return dets[keep]


def soft_nms(dets, nms_configs):
  """Soft non-maximum suppression.

  [1] Soft-NMS -- Improving Object Detection With One Line of Code.
    https://arxiv.org/abs/1704.04503

  Args:
    dets: detection with shape (num, 5) and format [x1, y1, x2, y2, score].
    nms_configs: a dict config that may contain the following members
      * method: one of {`linear`, `gaussian`, 'hard'}. Use `gaussian` if None.
      * iou_thresh (float): IOU threshold, only for `linear`, `hard`.
      * sigma: Gaussian parameter, only for method 'gaussian'.
      * score_thresh (float): Box score threshold for final boxes.

  Returns:
    numpy.array: Retained boxes.
  """
  method = nms_configs.get('method', 'gaussian')
  # Default sigma and iou_thresh are from the original soft-nms paper.
  sigma = nms_configs.get('sigma', 0.5)
  iou_thresh = nms_configs.get('iou_thresh', 0.3)
  score_thresh = nms_configs.get('score_thresh', 0.001)

  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  # expand dets with areas, and the second dimension is
  # x1, y1, x2, y2, score, area
  dets = np.concatenate((dets, areas[:, None]), axis=1)

  retained_box = []
  while dets.size > 0:
    max_idx = np.argmax(dets[:, 4], axis=0)
    dets[[0, max_idx], :] = dets[[max_idx, 0], :]
    retained_box.append(dets[0, :-1])

    xx1 = np.maximum(dets[0, 0], dets[1:, 0])
    yy1 = np.maximum(dets[0, 1], dets[1:, 1])
    xx2 = np.minimum(dets[0, 2], dets[1:, 2])
    yy2 = np.minimum(dets[0, 3], dets[1:, 3])

    w = np.maximum(xx2 - xx1 + 1, 0.0)
    h = np.maximum(yy2 - yy1 + 1, 0.0)
    inter = w * h
    iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

    if method == 'linear':
      weight = np.ones_like(iou)
      weight[iou > iou_thresh] -= iou[iou > iou_thresh]
    elif method == 'gaussian':
      weight = np.exp(-(iou * iou) / sigma)
    else:  # traditional nms
      weight = np.ones_like(iou)
      weight[iou > iou_thresh] = 0

    dets[1:, 4] *= weight
    retained_idx = np.where(dets[1:, 4] >= score_thresh)[0]
    dets = dets[retained_idx + 1, :]

  return np.vstack(retained_box)


def nms(dets, nms_configs):
  """Non-maximum suppression.

  Args:
    dets: detection with shape (num, 5) and format [x1, y1, x2, y2, score].
    nms_configs: a dict config that may contain parameters.

  Returns:
    numpy.array: Retained boxes.
  """

  nms_configs = nms_configs or {}
  method = nms_configs.get('method', None)

  if method == 'hard' or not method:
    return hard_nms(dets, nms_configs.get('iou_thresh', None))

  if method == 'diou':
    return diou_nms(dets, nms_configs.get('iou_thresh', None))

  if method in ('linear', 'gaussian'):
    return soft_nms(dets, nms_configs)

  raise ValueError('Unknown NMS method: {}'.format(method))


def _generate_anchor_configs(feat_sizes, min_level, max_level, num_scales,
                             aspect_ratios):
  """Generates mapping from output level to a list of anchor configurations.

  A configuration is a tuple of (num_anchors, scale, aspect_ratio).

  Args:
      feat_sizes: list of dict of integer numbers of feature map sizes.
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect ratio anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

  Returns:
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.
  """
  anchor_configs = {}
  for level in range(min_level, max_level + 1):
    anchor_configs[level] = []
    for scale_octave in range(num_scales):
      for aspect in aspect_ratios:
        anchor_configs[level].append(
            ((feat_sizes[0]['height'] / float(feat_sizes[level]['height']),
              feat_sizes[0]['width'] / float(feat_sizes[level]['width'])),
             scale_octave / float(num_scales), aspect))
  return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
  """Generates multiscale anchor boxes.

  Args:
    image_size: tuple of integer numbers of input image size.
    anchor_scale: float number representing the scale of size of the base
      anchor to the feature stride 2^level.
    anchor_configs: a dictionary with keys as the levels of anchors and
      values as a list of anchor configuration.

  Returns:
    anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
      feature levels.
  Raises:
    ValueError: input size must be the multiple of largest feature stride.
  """
  boxes_all = []
  for _, configs in anchor_configs.items():
    boxes_level = []
    for config in configs:
      stride, octave_scale, aspect = config
      base_anchor_size_x = anchor_scale * stride[1] * 2**octave_scale
      base_anchor_size_y = anchor_scale * stride[0] * 2**octave_scale
      anchor_size_x_2 = base_anchor_size_x * aspect[0] / 2.0
      anchor_size_y_2 = base_anchor_size_y * aspect[1] / 2.0

      x = np.arange(stride[1] / 2, image_size[1], stride[1])
      y = np.arange(stride[0] / 2, image_size[0], stride[0])
      xv, yv = np.meshgrid(x, y)
      xv = xv.reshape(-1)
      yv = yv.reshape(-1)

      boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                         yv + anchor_size_y_2, xv + anchor_size_x_2))
      boxes = np.swapaxes(boxes, 0, 1)
      boxes_level.append(np.expand_dims(boxes, axis=1))
    # concat anchors on the same level to the reshape NxAx4
    boxes_level = np.concatenate(boxes_level, axis=1)
    boxes_all.append(boxes_level.reshape([-1, 4]))

  anchor_boxes = np.vstack(boxes_all)
  return anchor_boxes


def _generate_detections_tf(cls_outputs,
                            box_outputs,
                            anchor_boxes,
                            indices,
                            classes,
                            image_id,
                            image_scale,
                            image_size,
                            min_score_thresh=MIN_SCORE_THRESH,
                            max_boxes_to_draw=MAX_DETECTIONS_PER_IMAGE,
                            soft_nms_sigma=0.25,
                            iou_threshold=0.5):
  """Generates detections with model outputs and anchors.

  Args:
    cls_outputs: a numpy array with shape [N, 1], which has the highest class
      scores on all feature levels. The N is the number of selected
      top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)
    box_outputs: a numpy array with shape [N, 4], which stacks box regression
      outputs on all feature levels. The N is the number of selected top-k
      total anchors on all levels. (k being MAX_DETECTION_POINTS)
    anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
      feature levels. The N is the number of selected top-k total anchors on
      all levels.
    indices: a numpy array with shape [N], which is the indices from top-k
      selection.
    classes: a numpy array with shape [N], which represents the class
      prediction on all selected anchors from top-k selection.
    image_id: an integer number to specify the image id.
    image_scale: a float tensor representing the scale between original image
      and input image for the detector. It is used to rescale detections for
      evaluating with the original groundtruth annotations.
    image_size: a tuple (height, width) or an integer for image size.
    min_score_thresh: A float representing the threshold for deciding when to
      remove boxes based on score.
    max_boxes_to_draw: Max number of boxes to draw.
    soft_nms_sigma: A scalar float representing the Soft NMS sigma parameter;
      See Bodla et al, https://arxiv.org/abs/1704.04503).  When
        `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
        NMS.
    iou_threshold: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.

  Returns:
    detections: detection results in a tensor with each row representing
      [image_id, ymin, xmin, ymax, xmax, score, class]
  """
  if not image_size:
    raise ValueError('tf version generate_detection needs non-empty image_size')

  logging.info('Using tf version of post-processing.')
  anchor_boxes = tf.gather(anchor_boxes, indices)

  scores = tf.math.sigmoid(cls_outputs)
  # apply bounding box regression to anchors
  boxes = decode_box_outputs_tf(box_outputs, anchor_boxes)
  # TF API is slightly different from paper, here we follow the paper value:
  # https://github.com/tensorflow/tensorflow/issues/40253.
  top_detection_idx, scores = tf.image.non_max_suppression_with_scores(
      boxes,
      scores,
      max_boxes_to_draw,
      iou_threshold=iou_threshold,
      score_threshold=min_score_thresh,
      soft_nms_sigma=soft_nms_sigma)
  boxes = tf.gather(boxes, top_detection_idx)

  image_size = utils.parse_image_size(image_size)
  detections = tf.stack([
      tf.cast(tf.tile(image_id, tf.shape(top_detection_idx)), tf.float32),
      tf.clip_by_value(boxes[:, 0], 0, image_size[0]) * image_scale,
      tf.clip_by_value(boxes[:, 1], 0, image_size[1]) * image_scale,
      tf.clip_by_value(boxes[:, 2], 0, image_size[0]) * image_scale,
      tf.clip_by_value(boxes[:, 3], 0, image_size[1]) * image_scale,
      scores,
      tf.cast(tf.gather(classes, top_detection_idx) + 1, tf.float32)
  ], axis=1)
  return detections


def _generate_detections(cls_outputs, box_outputs, anchor_boxes, indices,
                         classes, image_id, image_scale, num_classes,
                         max_boxes_to_draw, nms_configs):
  """Generates detections with model outputs and anchors.

  Args:
    cls_outputs: a numpy array with shape [N, 1], which has the highest class
      scores on all feature levels. The N is the number of selected
      top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)
    box_outputs: a numpy array with shape [N, 4], which stacks box regression
      outputs on all feature levels. The N is the number of selected top-k
      total anchors on all levels. (k being MAX_DETECTION_POINTS)
    anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
      feature levels. The N is the number of selected top-k total anchors on
      all levels.
    indices: a numpy array with shape [N], which is the indices from top-k
      selection.
    classes: a numpy array with shape [N], which represents the class
      prediction on all selected anchors from top-k selection.
    image_id: an integer number to specify the image id.
    image_scale: a float tensor representing the scale between original image
      and input image for the detector. It is used to rescale detections for
      evaluating with the original groundtruth annotations.
    num_classes: a integer that indicates the number of classes.
    max_boxes_to_draw: max number of boxes to draw per image.
    nms_configs: A dict of NMS configs.

  Returns:
    detections: detection results in a tensor with each row representing
      [image_id, x, y, width, height, score, class]
  """
  anchor_boxes = anchor_boxes[indices, :]
  scores = sigmoid(cls_outputs)
  # apply bounding box regression to anchors
  boxes = decode_box_outputs(
      box_outputs.swapaxes(0, 1), anchor_boxes.swapaxes(0, 1))
  boxes = boxes[:, [1, 0, 3, 2]]
  # run class-wise nms
  detections = []
  for c in range(num_classes):
    indices = np.where(classes == c)[0]
    if indices.shape[0] == 0:
      continue
    boxes_cls = boxes[indices, :]
    scores_cls = scores[indices]
    # Select top-scoring boxes in each class and apply non-maximum suppression
    # (nms) for boxes in the same class. The selected boxes from each class are
    # then concatenated for the final detection outputs.
    all_detections_cls = np.column_stack((boxes_cls, scores_cls))
    top_detections_cls = nms(all_detections_cls, nms_configs)
    top_detections_cls[:, 2] -= top_detections_cls[:, 0]
    top_detections_cls[:, 3] -= top_detections_cls[:, 1]
    top_detections_cls = np.column_stack(
        (np.repeat(image_id, len(top_detections_cls)),
         top_detections_cls,
         np.repeat(c + 1, len(top_detections_cls)))
    )
    detections.append(top_detections_cls)

  def _generate_dummy_detections(number):
    detections_dummy = np.zeros((number, 7), dtype=np.float32)
    detections_dummy[:, 0] = image_id[0]
    detections_dummy[:, 5] = _DUMMY_DETECTION_SCORE
    return detections_dummy

  if detections:
    detections = np.vstack(detections)
    # take final 100 detections
    indices = np.argsort(-detections[:, -2])
    detections = np.array(
        detections[indices[0:max_boxes_to_draw]], dtype=np.float32)
    # Add dummy detections to fill up to 100 detections
    n = max(max_boxes_to_draw - len(detections), 0)
    detections_dummy = _generate_dummy_detections(n)
    detections = np.vstack([detections, detections_dummy])
  else:
    detections = _generate_dummy_detections(max_boxes_to_draw)

  detections[:, 1:5] *= image_scale

  return detections


class Anchors(object):
  """Multi-scale anchors class."""

  def __init__(self, min_level, max_level, num_scales, aspect_ratios,
               anchor_scale, image_size):
    """Constructs multiscale anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds two additional
        anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of tuples representing the aspect ratio anchors added
        on each level. For instances, aspect_ratios =
        [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
      anchor_scale: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: integer number or tuple of integer number of input image size.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_scale = anchor_scale
    self.image_size = utils.parse_image_size(image_size)
    self.feat_sizes = utils.get_feat_sizes(image_size, max_level)
    self.config = self._generate_configs()
    self.boxes = self._generate_boxes()

  def _generate_configs(self):
    """Generate configurations of anchor boxes."""
    return _generate_anchor_configs(self.feat_sizes, self.min_level,
                                    self.max_level, self.num_scales,
                                    self.aspect_ratios)

  def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
                                   self.config)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    return boxes

  def get_anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(object):
  """Labeler for multiscale anchor boxes."""

  def __init__(self, anchors, num_classes, match_threshold=0.5):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      anchors: an instance of class Anchors.
      num_classes: integer number representing number of classes in the dataset.
      match_threshold: float number between 0 and 1 representing the threshold
        to assign positive labels for anchors.
    """
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(
        match_threshold,
        unmatched_threshold=match_threshold,
        negatives_lower_than_unmatched=True,
        force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

    self._target_assigner = target_assigner.TargetAssigner(
        similarity_calc, matcher, box_coder)
    self._anchors = anchors
    self._match_threshold = match_threshold
    self._num_classes = num_classes

  def _unpack_labels(self, labels):
    """Unpacks an array of labels into multiscales labels."""
    labels_unpacked = collections.OrderedDict()
    anchors = self._anchors
    count = 0
    for level in range(anchors.min_level, anchors.max_level + 1):
      feat_size = anchors.feat_sizes[level]
      steps = feat_size['height'] * feat_size[
          'width'] * anchors.get_anchors_per_location()
      indices = tf.range(count, count + steps)
      count += steps
      labels_unpacked[level] = tf.reshape(
          tf.gather(labels, indices),
          [feat_size['height'], feat_size['width'], -1])
    return labels_unpacked

  def label_anchors(self, gt_boxes, gt_labels):
    """Labels anchors with ground truth inputs.

    Args:
      gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
        For each row, it stores [y0, x0, y1, x1] for four corners of a box.
      gt_labels: A integer tensor with shape [N, 1] representing groundtruth
        classes.
    Returns:
      cls_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of class logits at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      num_positives: scalar tensor storing number of positives in an image.
    """
    gt_box_list = box_list.BoxList(gt_boxes)
    anchor_box_list = box_list.BoxList(self._anchors.boxes)

    # cls_weights, box_weights are not used
    cls_targets, _, box_targets, _, matches = self._target_assigner.assign(
        anchor_box_list, gt_box_list, gt_labels)

    # class labels start from 1 and the background class = -1
    cls_targets -= 1
    cls_targets = tf.cast(cls_targets, tf.int32)

    # Unpack labels.
    cls_targets_dict = self._unpack_labels(cls_targets)
    box_targets_dict = self._unpack_labels(box_targets)
    num_positives = tf.reduce_sum(
        tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))

    return cls_targets_dict, box_targets_dict, num_positives

  def generate_detections(self,
                          cls_outputs,
                          box_outputs,
                          indices,
                          classes,
                          image_id,
                          image_scale,
                          image_size=None,
                          min_score_thresh=MIN_SCORE_THRESH,
                          max_boxes_to_draw=MAX_DETECTIONS_PER_IMAGE,
                          disable_pyfun=None,
                          nms_configs=None):
    """Generate detections based on class and box predictions."""
    if disable_pyfun:
      return _generate_detections_tf(
          cls_outputs,
          box_outputs,
          self._anchors.boxes,
          indices,
          classes,
          image_id,
          image_scale,
          image_size,
          min_score_thresh=min_score_thresh,
          max_boxes_to_draw=max_boxes_to_draw)
    else:
      logging.info('nms_configs=%s', nms_configs)
      return tf.py_func(
          functools.partial(_generate_detections, nms_configs=nms_configs), [
              cls_outputs,
              box_outputs,
              self._anchors.boxes,
              indices,
              classes,
              image_id,
              image_scale,
              self._num_classes,
              max_boxes_to_draw,
          ], tf.float32)
