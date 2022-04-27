# Domain Incremental Learning for Object Detection


This is the PyTorch implementation of our paper: <br>
**Domain Incremental Learning for Object Detection**<br>

<p align="center">
<img src="teaser/DILLB.png" width="85%">
</p>

# Installation

## Prerequisites

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.

## Install PyTorch in Conda env

```shell
# create conda env
conda create -n detectron2 python=3.6
# activate the enviorment
conda activate detectron2
# install PyTorch >=1.5 with GPU
conda install pytorch torchvision -c pytorch
```

## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Dataset download

1. Download COCO dataset

```shell
# download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

2. Organize the dataset as following:

```shell
DILLB/
└── datasets/
    └──bdd100k/
        └── images/
            ├── train
            └── val
        └── annotations/
            ├── bdd100k_labels_images_det_coco_train.json
            └── bdd100k_labels_images_det_coco_val.json
    └──Cityscapes_new/
        └── JPEGImages/
            ├── train_s
            └── test_s
        └── Annotations/
            ├── instances_train_s.json
            └── instances_test_s.json
    └── two_stage_datasets/
        └── index9/
            ├── coco12
            └── exdark
```
