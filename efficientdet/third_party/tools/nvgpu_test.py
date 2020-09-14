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
"""Tests for nvgpu."""
import pprint
import unittest
from third_party.tools import nvgpu


class TestNvgpu(unittest.TestCase):
  """Tests for nvgpu."""

  def test_gpu_info(self):
    """Test gpu info."""
    nvgpu_info_d = nvgpu.gpu_info()
    if nvgpu_info_d:
      pp = pprint.PrettyPrinter(indent=2)
      pp.pprint(nvgpu_info_d)
      self.assertIn('cuda_version', nvgpu_info_d.keys())

  def test_gpu_memory_info(self):
    """Test gpu info."""
    nvgpu_info_d = nvgpu.gpu_info()
    if nvgpu_info_d:
      print(nvgpu_info_d['gpu']['fb_memory_usage']['total'])
      print(nvgpu_info_d['gpu']['fb_memory_usage']['used'])
      print(nvgpu_info_d['gpu']['utilization']['memory_util'])


if __name__ == '__main__':
  unittest.main()
