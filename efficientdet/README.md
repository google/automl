# VFPN 

This work is heavily based on EfficientDet. We appreciate and thank them for their work. This repository serves merely as a VFPN Demo based on EfficientDet.


[1] Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.
Arxiv link: https://arxiv.org/abs/1911.09070

# Architecture

<img src="image.png" />


## 2. Pretrained EfficientDet Checkpoints with VFPN

We have provided a list of checkpoints and results as follows:

|       Model    |  AP<sup>val</sup> | AP<sup>test</sup>    |  AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>S</sub>   |  AP<sub>M</sub>    |  AP<sub>L</sub>   | | #params | #FLOPs |
|----------     |------ |------ |------ | -------- | ------| ------| ------ |------ |------ |  :------: |
|     EfficientDet-D0 ([ckpt](https://drive.google.com/open?id=1gZwKHMiLKC87pft9HIrjDiXF8heVjRgr) | 34.44 | 34.8 | 53.3 | 36.9 | 12.9 | 39.4 | 52.6 |  | 4.5M | 2.98B  |
|     EfficientDet-D1 ([ckpt](https://drive.google.com/open?id=1Uv4cCERDwPdMj_oArX1AIL2Q0fShsjzU) | 39.74  | 40.1 | 59.4 | 43.0 | 18.8 | 44.7 | 56.6 | | 7.8M | 7.45B |


** <em>val</em> denotes validation results, <em>test-dev</em> denotes test-dev2017 results. AP<sup>val</sup> is for validation accuracy, all other AP results in the table are for COCO test-dev2017. All accuracy numbers are for single-model single-scale without ensemble or test-time augmentation. All checkpoints are trained with baseline preprocessing (no autoaugmentation).



## 3. Benchmark model latency.


There are two types of latency: network latency and end-to-end latency.

(1) To measure the network latency (from the fist conv to the last class/box
prediction output), use the following command:

    !python model_inspect.py --runmode=bm --model_name=efficientdet-d0

** add --hparams="precision=mixed-float16" if running on V100.


(2) To measure the end-to-end latency (from the input image to the final rendered
new image, including: image preprocessing, network, postprocessing and NMS),
use the following command:

    !rm  -rf /tmp/benchmark/
    !python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 \
      --ckpt_path=efficientdet-d0 --saved_model_dir=/tmp/benchmark/ \

    !python model_inspect.py --runmode=saved_model_benchmark \
      --saved_model_dir=/tmp/benchmark/efficientdet-d0_frozen.pb \
      --model_name=efficientdet-d0  --input_image=testdata/img1.jpg  \
      --output_image_dir=/tmp/  \


## 4. Inference for images.

    # Step0: download model 

    # Step 1: export saved model.
    !python model_inspect.py --runmode=saved_model \
      --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 \
      --hparams="image_size=1920x1280" \
      --saved_model_dir=/tmp/saved_model

    # Step 2: do inference with saved model.
    !python model_inspect.py --runmode=saved_model_infer \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --hparams="image_size=1920x1280"  \
      --saved_model_dir=/tmp/saved_model  \
      --input_image=img.png --output_image_dir=/tmp/
    # you can visualize the output /tmp/0.jpg



Here is an example of EfficientDet-D0 with VFPN visualization: 

<p align="center">
<img src="people-demo.jpg" width="800" />
</p>

## 5. Inference for videos.

You can run inference for a video and show the results online:

    # step 0: download the example video.
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/data/video480p.mov -O input.mov

    # step 1: export saved model.
    !python model_inspect.py --runmode=saved_model \
      --model_name=efficientdet-d0 --ckpt_path=efficientdet-d0 \
      --saved_model_dir=/tmp/savedmodel

    # step 2: inference video using saved_model_video.
    !python model_inspect.py --runmode=saved_model_video \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --saved_model_dir=/tmp/savedmodel --input_video=input.mov

    # alternative step 2: inference video and save the result.
    !python model_inspect.py --runmode=saved_model_video \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --saved_model_dir=/tmp/savedmodel --input_video=input.mov  \
      --output_video=output.mov

## 6. Eval on COCO 2017 val or test-dev.

    // Download coco data.
    !wget http://images.cocodataset.org/zips/val2017.zip
    !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    !unzip val2017.zip
    !unzip annotations_trainval2017.zip

    // convert coco data to tfrecord.
    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
        --image_dir=val2017 \
        --caption_annotations_file=annotations/captions_val2017.json \
        --output_file_prefix=tfrecord/val \
        --num_shards=32

    // Run eval.
    !python main.py --mode=eval  \
        --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
        --validation_file_pattern=tfrecord/val*  \
        --val_json_file=annotations/instances_val2017.json  \
        --use_tpu=False

You can also run eval on test-dev set with the following command:

    !wget http://images.cocodataset.org/zips/test2017.zip
    !unzip -q test2017.zip
    !wget http://images.cocodataset.org/annotations/image_info_test2017.zip
    !unzip image_info_test2017.zip

    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
          --image_dir=test2017 \
          --image_info_file=annotations/image_info_test-dev2017.json \
          --output_file_prefix=tfrecord/testdev \
          --num_shards=32

    # Eval on test-dev: testdev_dir must be set.
    # Also, test-dev has 20288 images rather than val 5000 images.
    !python main.py --mode=eval  \
        --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
        --validation_file_pattern=tfrecord/testdev*  \
        --testdev_dir='testdev_output' --eval_samples=20288 \
        --use_tpu=False
    # Now you can submit testdev_output/detections_test-dev2017_test_results.json to
    # coco server: https://competitions.codalab.org/competitions/20794#participate

## 7. Train on PASCAL VOC 2012 with backbone ImageNet ckpt.

    # Download and convert pascal data.
    !wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    !tar xf VOCtrainval_11-May-2012.tar
    !mkdir tfrecord
    !PYTHONPATH=".:$PYTHONPATH"  python dataset/create_pascal_tfrecord.py  \
        --data_dir=VOCdevkit --year=VOC2012  --output_path=tfrecord/pascal

    # Download backbone checkopints.
    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz
    !tar xf efficientnet-b0.tar.gz 

    !python main.py --mode=train_and_eval \
        --training_file_pattern=tfrecord/pascal*.tfrecord \
        --validation_file_pattern=tfrecord/pascal*.tfrecord \
        --val_json_file=tfrecord/json_pascal.json \
        --model_name=efficientdet-d0 \
        --model_dir=/tmp/efficientdet-d0-scratch  \
        --backbone_ckpt=efficientnet-b0  \
        --train_batch_size=8 \
        --eval_batch_size=8 --eval_samples=512 \
        --num_examples_per_epoch=5717 --num_epochs=1  \
        --hparams="num_classes=20,moving_average_decay=0" \
        --use_tpu=False

## 8. Finetune on PASCAL VOC 2012 with detector COCO ckpt.
Create a config file for the PASCAL VOC dataset called voc_config.yaml and put this in it.

      num_classes: 20
      moving_average_decay: 0

Download efficientdet coco checkpoint.

    !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz
    !tar xf efficientdet-d0.tar.gz

Finetune needs to use --ckpt rather than --backbone_ckpt.

    !python main.py --mode=train_and_eval \
        --training_file_pattern=tfrecord/pascal*.tfrecord \
        --validation_file_pattern=tfrecord/pascal*.tfrecord \
        --val_json_file=tfrecord/json_pascal.json \
        --model_name=efficientdet-d0 \
        --model_dir=/tmp/efficientdet-d0-finetune  \
        --ckpt=efficientdet-d0  \
        --train_batch_size=8 \
        --eval_batch_size=8 --eval_samples=1024 \
        --num_examples_per_epoch=5717 --num_epochs=1  \
        --hparams=voc_config.yaml \
        --use_tpu=False

If you want to do inference for custom data, you can run

    # Setting hparams-flag is needed sometimes.
    !python model_inspect.py --runmode=infer \
      --model_name=efficientdet-d0   --ckpt_path=efficientdet-d0 \
      --hparams=voc_config.yaml  \
      --input_image=img.png --output_image_dir=/tmp/
  
You should check more details of runmode which is written in caption-4.

## 9. Training EfficientDets on TPUs.

To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource.
   * A GCS bucket to store your training checkpoints (the "model directory").
   * Install latest TensorFlow for both GCE VM and Cloud.

Then train the model:

    !export PYTHONPATH="$PYTHONPATH:/path/to/models"
    !python main.py --tpu=TPU_NAME --training_file_pattern=DATA_DIR/*.tfrecord --model_dir=MODEL_DIR

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access.
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.


For more instructions about training on TPUs, please refer to the following tutorials:

  * EfficientNet tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet
  * RetinaNet tutorial: https://cloud.google.com/tpu/docs/tutorials/retinanet

NOTE: this is not an official Google product.
