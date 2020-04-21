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
# ======================================
"""Tests for hparams_config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import hparams_config


class HparamsConfigTest(tf.test.TestCase):

  def test_config_override(self):
    c = hparams_config.Config({'a': 1, 'b': 2})
    self.assertEqual(c.as_dict(), {'a': 1, 'b': 2})

    c.update({'a': 10})
    self.assertEqual(c.as_dict(), {'a': 10, 'b': 2})

    c.b = 20
    self.assertEqual(c.as_dict(), {'a': 10, 'b': 20})

    c.override(',')   # override with empty string has no effect.
    self.assertEqual(c.as_dict(), {'a': 10, 'b': 20})

    c.override('a=true,b=ss')
    self.assertEqual(c.as_dict(), {'a': True, 'b': 'ss'})

    c.override('a=100,,,b=2.3,')  # extra ',' is fine.
    self.assertEqual(c.as_dict(), {'a': 100, 'b': 2.3})

    # overrride string must be in the format of xx=yy.
    with self.assertRaises(ValueError):
      c.override('a=true,invalid_string')


if __name__ == '__main__':
  tf.test.main()
