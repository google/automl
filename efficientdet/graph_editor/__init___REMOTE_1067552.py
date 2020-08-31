<<<<<<< HEAD
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
=======
# Copyright 2020 Google Research. All Rights Reserved.
>>>>>>> bea39c1ee4c70d71018e6679da1d71c60ec8fafb
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
"""TensorFlow Graph Editor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import
from graph_editor.edit import *
from graph_editor.reroute import *
from graph_editor.select import *
from graph_editor.subgraph import *
from graph_editor.transform import *
from graph_editor.util import *
# pylint: enable=wildcard-import

# some useful aliases
# pylint: disable=g-bad-import-order
from graph_editor import subgraph as _subgraph
from graph_editor import util as _util
# pylint: enable=g-bad-import-order
ph = _util.make_placeholder_from_dtype_and_shape
sgv = _subgraph.make_view
sgv_scope = _subgraph.make_view_from_scope

del absolute_import
del division
del print_function
