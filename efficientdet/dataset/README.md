This folder provide a tool to convert coco data to tfrecord.

Example usage:

    $ pip install protoc
    $ protoc string_int_label_map.proto --python_out=.
    $ python create_coco_tfrecord.py \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"

We divde them into three groups, and users can specific any one of these groups:

  - train set: train_image_dir + train_annotations_file
  - val set: val_image_dir + val_annotations_file
  - test set: test_image_dir + testdev_annotations_file
