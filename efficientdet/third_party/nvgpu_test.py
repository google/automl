"""Tests for nvgpu."""
import shutil
import pprint
import unittest
from third_party import nvgpu


class TestNvgpu(unittest.TestCase):
  """Tests for nvgpu."""

  def test_gpu_info(self):
    """Test gpu info."""
    if not shutil.which("nvidia-smi"):
      self.skipTest("Nvidia card not available or driver/OS not supported.")
    nvgpu_info_d = nvgpu.gpu_info()
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(nvgpu_info_d)
    self.assertTrue("cuda_version" in nvgpu_info_d.keys())

  def test_gpu_memory_info(self):
    """Test gpu info."""
    if not shutil.which("nvidia-smi"):
      self.skipTest("Nvidia card not available or driver/OS not supported.")
    nvgpu_info_d = nvgpu.gpu_info()
    print(nvgpu_info_d['gpu']['fb_memory_usage']['total'])
    print(nvgpu_info_d['gpu']['fb_memory_usage']['used'])
    print(nvgpu_info_d['gpu']['utilization']['memory_util'])


if __name__ == "__main__":
  unittest.main()
