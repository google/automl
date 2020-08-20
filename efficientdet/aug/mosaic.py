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

from functools import partial
import tensorflow as tf
from typing import Tuple
from absl import logging

class Mosaic:
    """Mosaic.
        Mosaic Augmentation class.
        Notes:- 1. Mosaic sub images will not be preserving aspect ratio of original images passed.
                2. Tested on eager tensor for now, future will make compatible with static graphs too.
                3. This Implementation of mosaic augmentation is tested in tf2.x.
    """
    def __init__(
        self,
        out_size: Tuple = (680, 680),
        n_images: int = 4, #Currently four images are supported!
        _minimum_mosaic_image_dim: int = 25,
    ):
        """__init__.

        Args:
            out_size: output mosaic image size.
            n_images: number images to make mosaic
            _minimum_mosaic_image_dim: minimum percentage of out_size dimension should the mosaic be. i.e if out_size is (680,680) and _minimum_mosaic_image_dim is 25 , minimum mosaic sub images dimension will be 25 % of 680
        """
        # TODO #MED #use n_images to build mosaic.
        self._n_images = n_images
        self._out_size = out_size
        self._minimum_mosaic_image_dim = _minimum_mosaic_image_dim
        assert (
            _minimum_mosaic_image_dim > 0
        ), "Minimum Mosaic image dimension should be above 0"

    @property
    def n_images(self) -> int:
        """n_images.
            number of images tot mosaic property.
        """
        return self._n_images

    @property
    def out_size(self) -> int:
        """out_size.
            Output image size property.
        """
        return self._out_size

    def _mosaic_divide_points(self) -> Tuple:
        """_mosaic_divide_points.
            Returns:
                tuple of x and y which corresponds to mosaic divide points.
        """
        x_point = tf.random.uniform(
            shape=[1],
            minval=tf.cast(
                self.out_size[0] * (self._minimum_mosaic_image_dim / 100), tf.int32
            ),
            maxval=tf.cast(
                self.out_size[0] * ((100 - self._minimum_mosaic_image_dim) / 100),
                tf.int32,
            ),
            dtype=tf.int32,
        )
        y_point = tf.random.uniform(
            shape=[1],
            minval=tf.cast(
                self.out_size[1] * (self._minimum_mosaic_image_dim / 100), tf.int32
            ),
            maxval=tf.cast(
                self.out_size[1] * ((100 - self._minimum_mosaic_image_dim) / 100),
                tf.int32,
            ),
            dtype=tf.int32,
        )
        return x_point, y_point

    @staticmethod
    def _scale_box(box, image, mosaic_image):
        """_scale_box.
            static bounding boxes scaling methods which scales the boxes with mosaic sub image.

        Args:
            box: mosaic image box.
            image: original image.
            mosaic_image: mosaic sub image.
        Returns:
            Scaled bounding boxes.

        """
        return [
            box[0] * mosaic_image.shape[1] / image.shape[1],
            box[1] * mosaic_image.shape[0] / image.shape[0],
            box[2] * mosaic_image.shape[1] / image.shape[1],
            box[-1] * mosaic_image.shape[0] / image.shape[0],
        ]

    @tf.function
    def _scale_images(self, images, mosaic_divide_points: Tuple) -> Tuple:
        """_mosaic.
            Scale Sub Images.

        Args:
            images: original single images to make mosaic.
            mosaic_divide_points: Points to build mosaic around on given output size.
        Returns:
            (tuple)
            Scaled Mosaic sub images.
        """
        x, y = mosaic_divide_points[0][0], mosaic_divide_points[1][0]
        mosaic_image_topleft = tf.image.resize(images[0], (x, y))
        mosaic_image_topright = tf.image.resize(images[1], (self.out_size[0] - x, y))
        mosaic_image_bottomleft = tf.image.resize(images[2], (x, self.out_size[1] - y))
        mosaic_image_bottomright = tf.image.resize(
            images[3], (self.out_size[0] - x, self.out_size[1] - y)
        )
        return (
            mosaic_image_topleft,
            mosaic_image_topright,
            mosaic_image_bottomleft,
            mosaic_image_bottomright,
        )

    def _mosaic(self, images, boxes, mosaic_divide_points):
        """_mosaic.
            Builds mosaic of provided images.

        Args:
            images: original single images to make mosaic.
            boxes: corresponding bounding boxes to images.
            mosaic_divide_points: Points to build mosaic around on given output size.
        Returns:
            (tuple)
            Mosaic Image, Mosaic Boxes merged.
        """
        (
            mosaic_image_topleft,
            mosaic_image_topright,
            mosaic_image_bottomleft,
            mosaic_image_bottomright,
        ) = self._scale_images(images, mosaic_divide_points)

        #####################################################
        # Scale Boxes for TOP LEFT image.
        # Note:- Below function is complex because of TF item assignment restriction.
        # Map_fn is replace with vectorized_map below for optimization purpose.
        mosaic_box_topleft = tf.transpose(
            tf.vectorized_map(
                partial(
                    self._scale_box, image=images[0], mosaic_image=mosaic_image_topleft
                ),
                boxes[0],
            )
        )

        # Scale and Pad Boxes for TOP RIGHT image.

        mosaic_box_topright = tf.vectorized_map(
            partial(
                self._scale_box, image=images[1], mosaic_image=mosaic_image_topright
            ),
            boxes[1],
        )
        _num_boxes = boxes[1].shape[0]
        idx_tp = tf.constant([[1], [3]])
        update_tp = [
            [mosaic_image_topleft.shape[0]] * _num_boxes,
            [mosaic_image_topleft.shape[0]] * _num_boxes,
        ]
        mosaic_box_topright = tf.transpose(
            tf.tensor_scatter_nd_add(mosaic_box_topright, idx_tp, update_tp)
        )

        # Scale and Pad Boxes for BOTTOM LEFT image.

        mosaic_box_bottomleft = tf.vectorized_map(
            partial(
                self._scale_box, image=images[2], mosaic_image=mosaic_image_bottomleft
            ),
            boxes[2],
        )
        _num_boxes = boxes[2].shape[0]
        idx_bl = tf.constant([[0], [2]])
        update_bl = [
            [mosaic_image_topleft.shape[1]] * _num_boxes,
            [mosaic_image_topleft.shape[1]] * _num_boxes,
        ]
        mosaic_box_bottomleft = tf.transpose(
            tf.tensor_scatter_nd_add(mosaic_box_bottomleft, idx_bl, update_bl)
        )

        # Scale and Pad Boxes for BOTTOM RIGHT image.

        mosaic_box_bottomright = tf.vectorized_map(
            partial(
                self._scale_box, image=images[3], mosaic_image=mosaic_image_bottomright
            ),
            boxes[3],
        )
        _num_boxes = boxes[3].shape[0]
        idx_br = tf.constant([[0], [2], [1], [3]])
        update_br = [
            [mosaic_image_topright.shape[1]] * _num_boxes,
            [mosaic_image_topright.shape[1]] * _num_boxes,
            [mosaic_image_bottomleft.shape[0]] * _num_boxes,
            [mosaic_image_bottomleft.shape[0]] * _num_boxes,
        ]
        mosaic_box_bottomright = tf.transpose(
            tf.tensor_scatter_nd_add(mosaic_box_bottomright, idx_br, update_br)
        )

        # Gather mosaic_sub_images and boxes.
        mosaic_images = [
            mosaic_image_topleft,
            mosaic_image_topright,
            mosaic_image_bottomleft,
            mosaic_image_bottomright,
        ]
        mosaic_boxes = [
            mosaic_box_topleft,
            mosaic_box_topright,
            mosaic_box_bottomleft,
            mosaic_box_bottomright,
        ]

        return mosaic_images, mosaic_boxes

    def __call__(self, images, boxes):

        if len(images) != 4:
            _err_msg = "Currently Exact 4 Images are supported by Mosaic Augmentation."
            logging.error(_err_msg)
            raise Exception(_err_msg)

        if len(images) != len(boxes):
            _err_msg = "Each Image should have atleast one bbox."
            logging.error(_err_msg)
            raise Exception(_err_msg)


        """Builds mosaic with given images , boxes"""
        x, y = self._mosaic_divide_points()
        _mosaic_sub_images, _mosaic_boxes = self._mosaic(
            images, boxes, mosaic_divide_points=(x, y)
        )

        _upper_stack = tf.concat([_mosaic_sub_images[0], _mosaic_sub_images[1]], axis=0)
        _lower_stack = tf.concat([_mosaic_sub_images[2], _mosaic_sub_images[3]], axis=0)
        _mosaic_image = tf.concat([_upper_stack, _lower_stack], axis=1)
        return _mosaic_image, _mosaic_boxes
