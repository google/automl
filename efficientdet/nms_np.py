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
import numpy as np

# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -5.0

# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5

# The maximum number of (anchor,class) pairs to keep for non-max suppression.
MAX_DETECTION_POINTS = 5000


def sigmoid(x):
  """Sigmoid function for use with Numpy for CPU evaluation."""
  return 1 / (1 + np.exp(-x))


def decode_box_outputs_np(rel_codes, anchors):
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
  method = nms_configs['method']
  # Default sigma and iou_thresh are from the original soft-nms paper.
  sigma = nms_configs['sigma'] or 0.5
  iou_thresh = nms_configs['iou_thresh'] or 0.3
  score_thresh = nms_configs['score_thresh'] or 0.001

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
  method = nms_configs['method']

  if method == 'hard' or not method:
    return hard_nms(dets, nms_configs['iou_thresh'])

  if method == 'diou':
    return diou_nms(dets, nms_configs['iou_thresh'])

  if method in ('linear', 'gaussian'):
    return soft_nms(dets, nms_configs)

  raise ValueError('Unknown NMS method: {}'.format(method))


def per_class_nms(boxes, scores, classes, image_id, image_scale, num_classes,
                  max_boxes_to_draw, nms_configs):
  """Perform per class nms."""
  boxes = boxes[:, [1, 0, 3, 2]]
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
  boxes = decode_box_outputs_np(
      box_outputs.swapaxes(0, 1), anchor_boxes.swapaxes(0, 1))
  # run class-wise nms
  return per_class_nms(boxes, scores, classes, image_id, image_scale,
                       num_classes, max_boxes_to_draw, nms_configs)
