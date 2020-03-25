This folder provide a tool to convert coco data to tfrecord.

Example usage:

    $ export PYTHONPATH=".:$PYTHONPATH"
    $ pip install protoc
    $ protoc dataset/string_int_label_map.proto --python_out=.
    $ python dataset/create_coco_tfrecord.py \
      --image_dir="${IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=32

    # Input files are tried in order of:
    # image_info_file > object_annotations_file > caption_annotations_file
