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
"""Simple GraphEditor example."""

import numpy as np
import tensorflow as tf
import third_party.graph_edit as ge


def main(_):
  """Create and modify graph."""
  # create a graph
  g = tf.Graph()
  with g.as_default():
    a = tf.constant(1.0, shape=[2, 3], name="a")
    b = tf.constant(2.0, shape=[2, 3], name="b")
    c = tf.add(
        tf.placeholder(dtype=np.float32),
        tf.placeholder(dtype=np.float32),
        name="c")

  # modify the graph
  ge.swap_inputs(c.op, [a, b])

  # print the graph def
  print(g.as_graph_def())

  # and print the value of c
  with tf.Session(graph=g) as sess:
    res = sess.run(c)
    print(res)


if __name__ == "__main__":
  tf.app.run()
