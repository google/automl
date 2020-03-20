# EfficientDet

[1] Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020.
    Arxiv link: https://arxiv.org/abs/1911.09070


## 1. About EfficientDet Models

EfficientDets are a family of object detection models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and more efficient than previous models.


EfficientDets are developed based on the advanced backbone, a new BiFPN, and a new scaling technique:

<p align="center">
<img src="./g3doc/network.png" width="80%" />
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



## 2. Using Pretrained EfficientDet Checkpoints

We have provided a list of EfficientNet checkpoints for EfficientNet checkpoints:.

  * With the baseline RetinaNet preprocessing, we have achieved state-of-the-art results.


|       Model    |  mAP    |  |mAP<sub>S</sub>   |  mAP<sub>M</sub>    |  mAP<sub>L</sub>   | | #params | #FLOPs |
|----------     |------ |------ | -------- | ------| ------| ------ |------ |  :------: |
|     EfficientDet-D0 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0-coco.txt))    | 32.9 |   |  12.9   |  38.2 |  51.2 | | 3.9M | 2.54B  |
|     EfficientDet-D1 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d1.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d1-coco.txt))    | 38.9 |   |  19.5   |  41.1 |  56.5  | | 6.6M | 6.10B |
|     EfficientDet-D2 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d2.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d2-coco.txt))    | 42.2 |   |  24.0   |  46.3 |  59.4 | | 8.1M | 11.0B |
|     EfficientDet-D3 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d3.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d3-coco.txt))    | 45.5 |   |  29.1   |  49.2 |  61.3 | | 12.0M | 24.9B |
|     EfficientDet-D4 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d4.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d4-coco.txt))    |  48.0 |   |  31.9   |  52.0 | 63.1 | | 20.7M | 55.2B |
|     EfficientDet-D5 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d5.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d5-coco.txt))    |  49.8 |   |  33.6   |  53.7 | 64.2 | | 33.7M | 135.4B |
|     EfficientDet-D6 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d6.tar.gz), [result](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d6-coco.txt))    |  50.9 |   |  36.2   |  54.9 | 64.8 |  | 51.9M  |  225.6B  |

  ** All checkpoints are trained without autoaugmentation.


A quick way to load these checkpoints is to run:

    $ export MODEL=efficientdet-d0
    $ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/${MODEL}.tar.gz
    $ tar xf ${MODEL}.tar.gz
    $ python model_inspect.py --input_image_size=512 --runmode=ckpt --model_name=$MODEL --ckpt_path=$MODEL

TODO: add a colab for more examples.


## 3. Training EfficientDets on TPUs.


To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource.
   * A GCS bucket to store your training checkpoints (the "model directory").
   * Install TensorFlow version >= 1.13 for both GCE VM and Cloud.

Then train the model:

    $ export PYTHONPATH="$PYTHONPATH:/path/to/models"
    $ python main.py --tpu=TPU_NAME --data_dir=DATA_DIR --model_dir=MODEL_DIR

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access.
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.


For more instructions, please refer to the following tutorials:

  * EfficientNet tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet
  * RetinaNet tutorial: https://cloud.google.com/tpu/docs/tutorials/retinanet

