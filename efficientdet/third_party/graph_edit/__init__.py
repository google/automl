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
"""TensorFlow Graph Editor."""

# pylint: disable=wildcard-import
from third_party.graph_edit.edit import *
from third_party.graph_edit.reroute import *
from third_party.graph_edit.select import *
from third_party.graph_edit.subgraph import *
from third_party.graph_edit.transform import *
from third_party.graph_edit.util import *
# pylint: enable=wildcard-import

# some useful aliases
# pylint: disable=g-bad-import-order
from third_party.graph_edit import subgraph as _subgraph
from third_party.graph_edit import util as _util
# pylint: enable=g-bad-import-order
ph = _util.make_placeholder_from_dtype_and_shape
sgv = _subgraph.make_view
sgv_scope = _subgraph.make_view_from_scope
