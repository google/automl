from absl import logging
import tensorflow as tf

def vectorized_iou(d1, d2):
    x1, y1, w1, h1 = tf.split(d1[:, 1:5], 4, axis=1)
    x2, y2, w2, h2 = tf.split(d2[:, 1:5], 4, axis=1)
    
    x11 = x1
    y11 = y1
    x21 = x2
    y21 = y2

    x12 = x1 + w1
    y12 = y1 + h1
    x22 = x2 + w2
    y22 = y2 + h2

    xA = tf.maximum(x11, x21)
    yA = tf.maximum(y11, y21)
    xB = tf.minimum(x12, x22)
    yB = tf.minimum(y12, y22)

    interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


def find_matching_cluster(clusters, box):
  if len(clusters) == 0:
      return -1
  tiled_boxes = tf.tile(tf.expand_dims(box, axis=0), [len(clusters), 1])

  ious = vectorized_iou(tf.stack(clusters), tiled_boxes)
  ious = tf.concat([tf.constant([0.55]), tf.reshape(ious, [len(clusters)])], axis=0)
  best_index = tf.argmax(ious)

  return best_index - 1


def average_detections(detections):
  detections = tf.stack(detections)
  return [
      detections[0][0],
      tf.math.reduce_mean(detections[:, 1]),
      tf.math.reduce_mean(detections[:, 2]),
      tf.math.reduce_mean(detections[:, 3]),
      tf.math.reduce_mean(detections[:, 4]),
      tf.math.reduce_mean(detections[:, 5]),
      detections[0][6],
  ]


def ensemble_boxes(params, detections):
  # [id, x, y, w, h, score, class]

  all_clusters = []

  # cluster the detections
  for cid in range(params['num_classes']):
    indices = tf.where(tf.equal(detections[:, 6], cid))
    if indices.shape[0] == 0:
        continue
    class_detections = tf.gather_nd(detections, indices)

    clusters = []
    cluster_averages = []
    for d in class_detections:
        cluster_index = find_matching_cluster(cluster_averages, d)
        if cluster_index == -1:
            clusters.append([d])
            cluster_averages.append(d)
        else:
            clusters[cluster_index].append(d)
            cluster_averages[cluster_index] = average_detections(
                clusters[cluster_index])

    all_clusters.extend(cluster_averages)

  all_clusters.sort(reverse=True, key=lambda d: d[5])
  return tf.stack(all_clusters)
