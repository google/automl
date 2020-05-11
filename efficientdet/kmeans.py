import numpy as np
import tensorflow as tf
from absl import flags, app, logging

flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string('validation_file_pattern', None,
                    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string('test_file_pattern', None,
                    'Glob for test tfrecords (e.g., COCO test2017 set)')
flags.DEFINE_integer('cluster_number', 9, 'Cluster number')

FLAGS = flags.FLAGS


class Kmeans:

  def parse_tfrecord(self, example_proto):
    feature_description = {
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    xmins = features['image/object/bbox/xmin'].values
    xmaxs = features['image/object/bbox/xmax'].values
    ymins = features['image/object/bbox/ymin'].values
    ymaxs = features['image/object/bbox/ymax'].values
    return xmins, xmaxs, ymins, ymaxs

  def __init__(self, cluster_number, glob_path):
    self.cluster_number = cluster_number
    self.glob_path = glob_path

  def iou(self, boxes, clusters):  # 1 box -> k clusters
    n = boxes.shape[0]
    k = self.cluster_number

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result

  def avg_iou(self, boxes, clusters):
    accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
    return accuracy

  def kmeans(self, boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed()
    clusters = boxes[np.random.choice(box_number, k,
                                      replace=False)]  # init k clusters
    while True:
      distances = 1 - self.iou(boxes, clusters)
      current_nearest = np.argmin(distances, axis=1)
      if (last_nearest == current_nearest).all():
        break  # clusters won't change
      for cluster in range(k):
        clusters[cluster] = dist(  # update clusters
            boxes[current_nearest == cluster], axis=0)

      last_nearest = current_nearest

    return clusters

  def get_boxes(self):
    dataset = tf.data.Dataset.list_files(self.glob_path).interleave(
        tf.data.TFRecordDataset).map(self.parse_tfrecord)
    result = []
    for xmins, xmaxs, ymins, ymaxs in dataset.as_numpy_iterator():
      width = xmaxs - xmins
      height = ymaxs - ymins
      wh = np.transpose(np.stack([width, height], 0))
      result.append(wh)
    return np.concatenate(result, 0)

  def txt2clusters(self):
    all_boxes = self.get_boxes()
    result = self.kmeans(all_boxes, k=self.cluster_number)
    result = result[np.lexsort(result.T[0, None])]
    logging.info("K anchors ratio:\n {}".format(result / np.mean(result)))
    logging.info("Accuracy: {:.2f}%".format(
        self.avg_iou(all_boxes, result) * 100))


def run_kmeans(_):
  cluster_number = FLAGS.cluster_number
  globs = [
      FLAGS.training_file_pattern, FLAGS.validation_file_pattern,
      FLAGS.test_file_pattern
  ]

  def _is_not_None(item):
    return item is not None

  globs = list(filter(_is_not_None, globs))
  kmeans = Kmeans(cluster_number, globs)
  kmeans.txt2clusters()


if __name__ == '__main__':
  app.run(run_kmeans)
