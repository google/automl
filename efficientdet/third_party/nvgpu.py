"""Python interface for nvidia-smi."""
import subprocess

from xml.etree import cElementTree as ElementTree


class XmlListConfig(list):
  """Convert XML to python list."""

  def __init__(self, aList):
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
  query = subprocess.run(["nvidia-smi", "-q", "-x"], stdout=subprocess.PIPE)
  root = ElementTree.XML(query.stdout)
  gpu_info_d = XmlDictConfig(root)

  return gpu_info_d
