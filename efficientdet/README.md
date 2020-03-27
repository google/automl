# EfficientDet

[1] Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.
    Arxiv link: https://arxiv.org/abs/1911.09070

Updates:

  - **Mar26: Fixed a few bugs and updated all checkpoints/results.**
  - **Mar24: Added tutorial with visualization and coco eval: [tutorial.ipynb](tutorial.ipynb)**
  - Mar 13: Released the initial code and models.

## 1. About EfficientDet Models

EfficientDets are a family of object detection models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and more efficient than previous models.


EfficientDets are developed based on the advanced backbone, a new BiFPN, and a new scaling technique:

<p align="center">
<img src="./g3doc/network.png" width="800" />
</p>

  * **Backbone**: we employ the more advanced [EfficientNets](https://arxiv.org/abs/1905.11946) as our backbone networks.
  * **BiFPN**: we propose a new bi-directional feature network, named BiFPN, to enable easy and fast feature fusion. In addition to the bi-directional topology, we also propose a new fast normalized fusion that enables better fusion with negligible latency cost.
  * **Scaling**: we propose to use a single compound scaling factor to govern the network depth, width, and resolution for all backbone, feature network, and prediction networks.

Based on the BiFPN topology and the fast feature fusion technique, we first develop a baseline detection model EfficientDet-D0, which has comparable accuracy as [YOLOv3](https://arxiv.org/abs/1804.02767). Then we scale up this baseline model using our compound scaling method to obtain a list of detection models EfficientDet-D1 to D6, with different trade-offs between accuracy and model complexity.

Our evaluation on COCO dataset show our EfficientDets outperform previous detectors by a large margin: In particular, our EfficientDet-D6 achieves state-of-the-art 50.9 mAP on COCO dataset, with 51.9M parameters and 229B FLOPs. Compared to previous best single-model AmoebaNet + NAS-FPN + AutoAugment ([ref](https://arxiv.org/abs/1906.11172)), our model achieves higher accuracy with 4x fewer parameters and 13x fewer FLOPs, and meanwhile runs 3x - 5x faster on GPU/CPU.


<table border="0">
<tr>
    <td>
    <img src="./g3doc/flops.png" width="100%" />
    </td>
    <td>
    <img src="./g3doc/params.png", width="100%" />
    </td>
</tr>
</table>

** For simplicity, we compare the whole detectors here. For more comparison on FPN/NAS-FPN/BiFPN, please see Table 4 of our [paper](https://arxiv.org/abs/1911.09070).



## 2. Pretrained EfficientDet Checkpoints

We have provided a list of EfficientNet checkpoints for EfficientNet checkpoints:.

  * With the baseline RetinaNet preprocessing, we have achieved state-of-the-art results.


|       Model    |  AP    |  AP<sub>50</sub> | AP<sub>75</sub> |AP<sub>S</sub>   |  AP<sub>M</sub>    |  AP<sub>L</sub>   | | #params | #FLOPs |
|----------     |------ |------ | -------- | ------| ------| ------ |------ |------ |  :------: |
|     EfficientDet-D0 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0-coco.txt))    | 33.5 | 51.6 | 35.2 | 52.6 | 38.7 | 12.4 |  | 3.9M | 2.54B  |
|     EfficientDet-D1 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d1.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d1-coco.txt))    | 39.1 | 58.1 | 41.8 | 56.9 | 44.7 | 18.6 | | 6.6M | 6.10B |
|     EfficientDet-D2 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d2.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d2-coco.txt))    | 42.5 | 61.7 | 45.3 | 59.0 | 48.0 | 23.7 | | 8.1M | 11.0B |
|     EfficientDet-D3 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d3.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d3-coco.txt))    | 45.9 | 65.0 | 49.0 | 61.7 | 50.2 | 28.0 | | 12.0M | 24.9B |
|     EfficientDet-D4 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d4.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d4-coco.txt))    |  49.0 | 68.5 | 52.9 | 64.0 | 53.7 | 33.4 |  | 20.7M | 55.2B |
|     EfficientDet-D5 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d5.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d5-coco.txt))    |  50.5 | 70.0 | 54.4 | 64.6 | 54.9 | 34.3 |  | 33.7M | 135.4B |
|     EfficientDet-D6 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d6.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d6-coco.txt))    |  51.3 | 70.5 | 55.4 | 65.2 | 55.6 | 35.1 | | 51.9M  |  225.6B  |

  ** All checkpoints are trained without autoaugmentation.

## 3. Run inference.

    $ export MODEL=efficientdet-d0
    $ export CKPT_PATH=efficientdet-d0
    $ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/${MODEL}.tar.gz
    $ tar xf ${MODEL}.tar.gz
    $ python model_inspect.py --runmode=infer --model_name=$MODEL --ckpt_path=$CKPT_PATH --input_image=testdata/img1.jpg --output_image_dir=/tmp
    # you can visualize the output /tmp/0.jpg

Here is an example of EfficientDet-D0 visualization: more on [tutorial](tutorial.ipynb)

<p align="center">
<img src="https://user-images.githubusercontent.com/6027221/77340634-d16dc300-6cea-11ea-822c-63853f457329.jpg" width="800" />
</p>

## 4. Eval on COCO 2017 val.

    // Download coco data.
    $ wget http://images.cocodataset.org/zips/val2017.zip
    $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    $ unzip val2017.zip
    $ unzip annotations_trainval2017.zip

    // convert coco data to tfrecord.
    $ mkdir tfrecrod
    $ PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py \
        --image_dir=val2017 \
        --caption_annotations_file=annotations/captions_val2017.json \
        --output_file_prefix=tfrecord/val \
        --num_shards=32

    // Run eval.
    $ python main.py --mode=eval  \
        --model_name=${MODEL}  --model_dir=${CKPT_PATH}  \
        --validation_file_pattern=tfrecord/val*  \
        --val_json_file=annotations/instances_val2017.json  \
        --hparams="use_bfloat16=false" --use_tpu=False

## 5. Training EfficientDets on single GPU.

    $ python main.py --training_file_pattern=/coco_tfrecord/train* \
        --model_name=$MODEL \
        --model_dir=/tmp/$MODEL \
        --hparams="use_bfloat16=false" --use_tpu=False


## 6. Training EfficientDets on TPUs.

To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource.
   * A GCS bucket to store your training checkpoints (the "model directory").
   * Install TensorFlow version >= 1.13 for both GCE VM and Cloud.

Then train the model:

    $ export PYTHONPATH="$PYTHONPATH:/path/to/models"
    $ python main.py --tpu=TPU_NAME --training_file_pattern=DATA_DIR/*.tfrecord --model_dir=MODEL_DIR

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access.
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.


For more instructions about training on TPUs, please refer to the following tutorials:

  * EfficientNet tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet
  * RetinaNet tutorial: https://cloud.google.com/tpu/docs/tutorials/retinanet

NOTE: this is not an official Google product.
