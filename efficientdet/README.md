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
|     EfficientDet-D0 ([ckpt](https://drive.google.com/open?id=1gZwKHMiLKC87pft9HIrjDiXF8heVjRgr)) | 34.44 | 34.8 | 53.3 | 36.9 | 12.9 | 39.4 | 52.6 |  | 4.5M | 2.98B  |
|     EfficientDet-D1 ([ckpt](https://drive.google.com/open?id=1Uv4cCERDwPdMj_oArX1AIL2Q0fShsjzU)) | 39.74  | 40.1 | 59.4 | 43.0 | 18.8 | 44.7 | 56.6 | | 7.8M | 7.45B |


** <em>val</em> denotes validation results, <em>test-dev</em> denotes test-dev2017 results. AP<sup>val</sup> is for validation accuracy, all other AP results in the table are for COCO test-dev2017. All accuracy numbers are for single-model single-scale without ensemble or test-time augmentation. All checkpoints are trained with baseline preprocessing (no autoaugmentation).


## 3. DEMO

<img src="people-demo.jpg" />
