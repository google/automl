import math
import tensorflow.compat.v1 as tf

class ScaleLayer(tf.keras.layers.Layer):

  def __init__(self, init_value=1.0, **kwargs):
    super(ScaleLayer, self).__init__(**kwargs)
    self.init_value = init_value

  def build(self, input_shape):
    shape = [1] * len(input_shape)
    self.scale = self.add_weight(
      shape=shape,
      initializer=tf.initializers.constant(self.init_value),
      name='scale')

  def call(self, inputs):
    outputs = self.scale * inputs
    return outputs

class FCOSHead(tf.keras.Model):
  def __init__(self, num_classes, prefix="fcos_head"):
    super(FCOSHead, self).__init__()
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    self.cls_logits = tf.layers.Conv2D(
      num_classes,
      kernel_size=3,
      strides=1,
      padding="same",
      kernel_initializer=tf.initializers.random_normal(stddev=0.01),
      bias_initializer=tf.initializers.constant(bias_value),
      name=(prefix + "_cls_logits")
    )
    self.bbox_pred = tf.layers.Conv2D(
      4,
      kernel_size=3,
      strides=1,
      padding="same",
      kernel_initializer=tf.initializers.random_normal(stddev=0.01),
      name=(prefix + "_bbox_pred")
    )
    self.centerness = tf.layers.Conv2D(
      1,
      kernel_size=3,
      strides=1,
      padding="same",
      kernel_initializer=tf.initializers.random_normal(stddev=0.01),
      name=(prefix + "_centerness")
    )

  def call(self, cls_feature, bbox_feature, **kwargs):
    logits = self.cls_logits(cls_feature)
    centerness = self.centerness(cls_feature)
    bbox_reg = self.bbox_pred(bbox_feature)
    return logits, bbox_reg, centerness


def fcos_block(cls_features, bbox_features, params, prefix="fcos_head"):
  logits_list = []
  bbox_reg_list = []
  centerness_list = []
  fcos_head = FCOSHead(params['num_classes'], prefix)
  for level in range(params['min_level'], params['max_level'] + 1):
    logits, bbox_reg, centerness = fcos_head(cls_features[level], bbox_features[level])
    name = prefix + ("_scale%d" % level)
    scale_bbox_reg = ScaleLayer(name=name)(bbox_reg)
    scale_bbox_reg = tf.math.exp(scale_bbox_reg)
    logits_list.append(logits)
    bbox_reg_list.append(scale_bbox_reg)
    centerness_list.append(centerness)
  return [logits_list, bbox_reg_list, centerness_list]


def compute_locations(feature, stride, dim_height=1, dim_width=2):
  w = tf.shape(feature)[dim_width]
  h = tf.shape(feature)[dim_height]

  shifts_x = tf.cast(tf.range(0, w * stride, stride, dtype=tf.int32), tf.float32)
  shifts_y = tf.cast(tf.range(0, h * stride, stride, dtype=tf.int32), tf.float32)

  shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
  shift_x = tf.reshape(shift_x, [-1])
  shift_y = tf.reshape(shift_y, [-1])
  locations = tf.stack([shift_x, shift_y], -1) + tf.cast(stride // 2, tf.float32)
  return locations

INF = 100000000

def bbox_regress_target(bboxes, points):
  '''
  Args:
      bboxes: [batch_size, m, 4+]
      points: [n, 2]
  Returns:
      reg_target: [batch_size, n, m, 4]
  '''
  xs = points[:, 0][tf.newaxis, :, tf.newaxis]
  ys = points[:, 1][tf.newaxis, :, tf.newaxis]
  l = xs - tf.expand_dims(bboxes[:, :, 1], 1)
  t = ys - tf.expand_dims(bboxes[:, :, 0], 1)
  r = tf.expand_dims(bboxes[:, :, 3], 1) - xs
  b = tf.expand_dims(bboxes[:, :, 2], 1) - ys
  reg_target = tf.stack([l, t, r, b], -1)
  return reg_target


def bbox_area(bboxes):
  '''
  Args:
      bboxes: [batch_size, m, 4+]
  Returns:
      area: [batch_size, m]
  '''
  width = bboxes[:, :, 2] - bboxes[:, :, 0] + 1
  height = bboxes[:, :, 3] - bboxes[:, :, 1] + 1
  return width * height


def compute_targets_for_locations(locations, gt_boxes, object_sizes_of_interest):
  '''
  Args:
      locations: [n, 2]
      gt_boxes: [batch_size, m, 4]
      object_sizes_of_interest: List[2]
  Returns:
      cls_target: [batch_size, n]
      reg_target: [batch_size, n, 4]
  '''
  n = tf.shape(locations)[0]
  m = tf.shape(gt_boxes)[1]
  batch_size = tf.shape(gt_boxes)[0]

  reg_target = bbox_regress_target(gt_boxes, locations)  # [batch_size, n, m, 4]

  is_in_boxes = tf.reduce_min(reg_target, 3) > 0  # [batch_size, n, m]

  # limit the regression range for each location
  max_reg_target = tf.reduce_max(reg_target, 3)  # [batch_size, n, m]
  is_cared_in_the_level = tf.logical_and(
    (max_reg_target >= object_sizes_of_interest[0]),
    (max_reg_target <= object_sizes_of_interest[1])
  )  # [batch_size, n, m]

  #
  is_valid = tf.broadcast_to(gt_boxes[:, tf.newaxis, :, 6] > 0, [batch_size, n, m])

  area = bbox_area(gt_boxes)  # [batch_size, m]
  locations_to_gt_area = tf.broadcast_to(area[:, tf.newaxis, :], [batch_size, n, m])
  locations_to_gt_area = tf.where_v2(
    (is_in_boxes & is_cared_in_the_level & is_valid),
    locations_to_gt_area,
    tf.constant(INF, dtype=tf.float32)
  )

  # if there are still more than one objects for a location,
  # we choose the one with minimal area
  locations_to_gt_inds = tf.argmin(locations_to_gt_area, 2)  # [batch_size, n]
  locations_to_min_area = tf.reduce_min(locations_to_gt_area, 2)  # [batch_size, n]

  reg_target = tf.gather_nd(
    reg_target,
    locations_to_gt_inds[:, :, tf.newaxis],
    batch_dims=2)
  cls_target = tf.gather_nd(
    gt_boxes[:, :, 6],
    locations_to_gt_inds[:, :, tf.newaxis],
    batch_dims=1)
  cls_target = tf.where_v2(
    locations_to_min_area < tf.constant(INF - 1, dtype=tf.float32),
    cls_target,
    tf.constant(0, dtype=tf.float32)
  )  # TODO mask assignment

  return cls_target, reg_target


def prepare_targets(points, gt_boxes):
  '''
      points: List[Tensor]
  '''
  object_sizes_of_interest = [
    [0, 64],
    [64, 128],
    [128, 256],
    [256, 512],
    [512, INF],
  ]
  cls_targets = []
  reg_targets = []
  for lvl, points_lvl in enumerate(points):
    cls_target_lvl, reg_target_lvl = compute_targets_for_locations(
      points_lvl, gt_boxes, object_sizes_of_interest[lvl]
    )
    cls_targets.append(cls_target_lvl)
    reg_targets.append(reg_target_lvl)
  return cls_targets, reg_targets

def iou_loss(pred, target, weight=None):
  '''
  Args:
      pred: [n, 4]
      target: [n, 4]
      weight: [n]
  Returns:
      loss: average iou loss
  '''
  pred_left = pred[:, 0]
  pred_top = pred[:, 1]
  pred_right = pred[:, 2]
  pred_bottom = pred[:, 3]

  target_left = target[:, 0]
  target_top = target[:, 1]
  target_right = target[:, 2]
  target_bottom = target[:, 3]

  target_area = (target_left + target_right) * \
                (target_top + target_bottom)
  pred_area = (pred_left + pred_right) * \
              (pred_top + pred_bottom)

  w_intersect = tf.minimum(pred_left, target_left) + \
                tf.minimum(pred_right, target_right)
  h_intersect = tf.minimum(pred_bottom, target_bottom) + \
                tf.minimum(pred_top, target_top)

  area_intersect = w_intersect * h_intersect
  area_union = target_area + pred_area - area_intersect

  ratio = (area_intersect + 1.0) / (area_union + 1.0)

  losses = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))

  if weight is not None:
    return tf.reduce_sum(losses * weight) / tf.reduce_sum(weight)
  else:
    return tf.reduce_mean(losses)


def focal_loss_sigmoid(pred,
                       target,
                       gamma=2.0,
                       alpha=0.25,
                       weight=None):
  # predict probability
  pred_sigmoid = tf.sigmoid(pred)

  # focal weight
  pt = tf.where_v2(
    tf.equal(target, 1.0),
    1.0 - pred_sigmoid,
    pred_sigmoid)
  alpha_weight = (alpha * target) + ((1 - alpha) * (1 - target))
  focal_weight = alpha_weight * tf.pow(pt, gamma)
  if weight is not None:
    focal_weight = focal_weight * weight

  # loss
  bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred)
  loss = bce * focal_weight
  loss = tf.reduce_sum(loss)

  return loss

def compute_centerness_targets(reg_targets):
  left_right_min = tf.minimum(reg_targets[:, 0], reg_targets[:, 2])
  left_right_max = tf.maximum(reg_targets[:, 0], reg_targets[:, 2])
  top_bottom_min = tf.minimum(reg_targets[:, 1], reg_targets[:, 3])
  top_bottom_max = tf.maximum(reg_targets[:, 1], reg_targets[:, 3])
  centerness = (left_right_min / left_right_max) * (top_bottom_min / top_bottom_max)
  return tf.sqrt(centerness)

def fcos_loss(box_cls,
              box_regression,
              centerness,
              labels,
              params):
  locations = []
  num_classes = params['num_classes']
  batch_size = params['batch_size']

  # TODO compute locations
  strides = [item.shape[2] for item in box_cls][::-1]
  for lvl, stride in enumerate(strides):
    locations.append(compute_locations(box_cls[lvl], stride))

  cls_targets, reg_targets = prepare_targets(locations, labels['groundtruth_data'])

  # concat over level
  box_cls_flatten = []
  box_regression_flatten = []
  centerness_flatten = []
  cls_ind_flatten = []
  reg_targets_flatten = []
  for l in range(len(cls_targets)):
    box_cls_flatten.append(tf.reshape(box_cls[l], [-1, num_classes]))
    box_regression_flatten.append(tf.reshape(box_regression[l], [-1, 4]))
    cls_ind_flatten.append(tf.reshape(cls_targets[l], [-1]))
    reg_targets_flatten.append(tf.reshape(reg_targets[l], [-1, 4]))
    centerness_flatten.append(tf.reshape(centerness[l], [-1]))
  box_cls_flatten = tf.concat(box_cls_flatten, 0)
  box_regression_flatten = tf.concat(box_regression_flatten, 0)
  centerness_flatten = tf.concat(centerness_flatten, 0)
  cls_ind_flatten = tf.concat(cls_ind_flatten, 0)
  reg_targets_flatten = tf.concat(reg_targets_flatten, 0)

  pos_inds = tf.where_v2(cls_ind_flatten > 0)
  # cls loss
  onehot_cls_target = tf.equal(
    tf.range(1, num_classes + 1, dtype=tf.int32)[tf.newaxis, :],
    tf.cast(cls_ind_flatten[:, tf.newaxis], tf.int32)
  )
  onehot_cls_target = tf.cast(onehot_cls_target, tf.float32)
  cls_loss = focal_loss_sigmoid(
    box_cls_flatten,
    onehot_cls_target,
    alpha=params['alpha'],
    gamma=params['gamma']
  ) / tf.cast((tf.shape(pos_inds)[0] + batch_size), tf.float32)  # add batch_size to avoid dividing by a zero

  box_regression_flatten = tf.gather_nd(box_regression_flatten, pos_inds)
  reg_targets_flatten = tf.gather_nd(reg_targets_flatten, pos_inds)
  centerness_flatten = tf.gather_nd(centerness_flatten, pos_inds)

  # centerness loss
  centerness_targets = compute_centerness_targets(reg_targets_flatten)
  centerness_loss = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=centerness_flatten,
    labels=centerness_targets
  )
  centerness_loss = tf.reduce_mean(centerness_loss)

  # regression loss
  reg_loss = iou_loss(
    box_regression_flatten,
    reg_targets_flatten,
    centerness_targets
  )
  return cls_loss, centerness_loss, reg_loss