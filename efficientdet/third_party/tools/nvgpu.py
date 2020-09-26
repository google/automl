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
"""Python interface for nvidia-smi."""
import subprocess
from xml.etree import cElementTree as ElementTree


class XmlListConfig(list):
  """Convert XML to python list."""

  def __init__(self, aList):
    super().__init__()
    for element in aList:
      if element:
        # treat like dict
        if len(element) == 1 or element[0].tag != element[1].tag:
          self.append(XmlDictConfig(element))
        # treat like list
        elif element[0].tag == element[1].tag:
          self.append(XmlListConfig(element))
      elif element.text:
        text = element.text.strip()
        if text:
          self.append(text)


class XmlDictConfig(dict):
  """Convert XML to python dict.

  Example usage:

  >>> tree = ElementTree.parse('your_file.xml')
  >>> root = tree.getroot()
  >>> xmldict = XmlDictConfig(root)

  Or, if you want to use an XML string:

  >>> root = ElementTree.XML(xml_string)
  >>> xmldict = XmlDictConfig(root)

  And then use xmldict for what it is... a dict.
  """

  def __init__(self, parent_element):
    """Convert XML to dict."""
    super().__init__()
    if parent_element.items():
      self.update(dict(parent_element.items()))
    for element in parent_element:
      if element:
        # treat like dict - we assume that if the first two tags
        # in a series are different, then they are all different.
        if len(element) == 1 or element[0].tag != element[1].tag:
          a_dict = XmlDictConfig(element)
        # treat like list - we assume that if the first two tags
        # in a series are the same, then the rest are the same.
        else:
          # here, we put the list in dictionary; the key is the
          # tag name the list elements all share in common, and
          # the value is the list itself
          a_dict = {element[0].tag: XmlListConfig(element)}
        # if the tag has attributes, add those to the dict
        if element.items():
          a_dict.update(dict(element.items()))
        self.update({element.tag: a_dict})
      # this assumes that if you've got an attribute in a tag,
      # you won't be having any text. This may or may not be a
      # good idea -- time will tell. It works for the way we are
      # currently doing XML configuration files...
      elif element.items():
        self.update({element.tag: dict(element.items())})
      # finally, if there are no child tags and no attributes, extract
      # the text
      else:
        self.update({element.tag: element.text})


def gpu_info():
  """Provide information about GPUs."""
  try:
    query = subprocess.check_output(['nvidia-smi', '-q', '-x'])
    root = ElementTree.XML(query)
    return XmlDictConfig(root)
  except FileNotFoundError:
    return None


def gpu_memory_util_message():
  """Provide information about GPUs."""
  gpu_info_d = gpu_info()
  if gpu_info_d is not None:
    mem_used = gpu_info_d['gpu']['fb_memory_usage']['used']
    mem_total = gpu_info_d['gpu']['fb_memory_usage']['total']
    mem_util = commonsize(mem_used) / commonsize(mem_total)
    logstring = ('GPU memory used: {} = {:.1%} '.format(mem_used, mem_util) +
                 'of total GPU memory: {}'.format(mem_total))
    return logstring
  return None


def commonsize(input_size: str):
  """Convert memory information to a common size (MiB)."""
  const_sizes = {
      'B': 1,
      'KB': 1e3,
      'MB': 1e6,
      'GB': 1e9,
      'TB': 1e12,
      'PB': 1e15,
      'KiB': 1024,
      'MiB': 1048576,
      'GiB': 1073741824
  }
  input_size = input_size.split(' ')
  # convert all to MiB
  if input_size[1] != 'MiB':
    converted_size = float(input_size[0]) * (
        const_sizes[input_size[1]] / 1048576.0)
  else:
    converted_size = float(input_size[0])

  return converted_size
