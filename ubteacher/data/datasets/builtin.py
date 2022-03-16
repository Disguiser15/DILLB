# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging

logger = logging.getLogger(__name__)

_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}

_SPLITS_COCO5_FORMAT = {}
_SPLITS_COCO5_FORMAT["coco5"] = {
    "coco_2017_train_for_5cls": (
        "coco5/images/train2017",
        "coco5/annotations/instances_train2017.json",
    ),
    "coco_2017_val_for_5cls": (
        "coco5/images/val2017",
        "coco5/annotations/instances_val2017.json",
    ),
}

_SPLITS_VOC5_FORMAT = {}
_SPLITS_VOC5_FORMAT["voc5"] = {
    "voc_2007_train_for_5cls": (
        "VOC2007_5/JPEGImages/train",
        "VOC2007_5/Annotations/instances_train.json",
    ),
    "voc_2007_test_for_5cls": (
        "VOC2007_5/JPEGImages/test",
        "VOC2007_5/Annotations/instances_test.json",
    ),
    "voc_2007_val_for_5cls": (
        "VOC2007_5/JPEGImages/val",
        "VOC2007_5/Annotations/instances_val.json",
    ),
    "voc_2007_trainval_for_5cls": (
        "VOC2007_5/JPEGImages/trainval",
        "VOC2007_5/Annotations/instances_trainval.json",
    ),
    "voc_2012_train_for_5cls": (
        "VOC2012_5/JPEGImages/train",
        "VOC2012_5/Annotations/instances_train.json",
    ),
    "voc_2012_val_for_5cls": (
        "VOC2012_5/JPEGImages/val",
        "VOC2012_5/Annotations/instances_val.json",
    ),
    "voc_2012_trainval_for_5cls": (
        "VOC2012_5/JPEGImages/trainval",
        "VOC2012_5/Annotations/instances_trainval.json",
    ),
}

_SPLITS_COCO20_FORMAT = {}
_SPLITS_COCO20_FORMAT["coco20"] = {
    "coco_train_for_20cls": (
        "index1/coco20/images/train2017",
        "index1/coco20/annotations/instances_train2017_new.json",
    ), #index1
    "coco_val_for_20cls": (
        "index1/coco20/images/val2017",
        "index1/coco20/annotations/instances_val2017_new.json",
    ),
    "cocoa_train_1step": (
        "index1/step1/coco_a/images",
        "index1/step1/coco_a/annotations/coco_a.json",
    ), #1step
    "cocoa_val_1step": (
        "index1/step1/coco_a/val/images",
        "index1/step1/coco_a/val/annotations/coco_a_val.json",
    ),
    "voc20%_1step": (
        "index1/step1/semi/voc20%/JPEGImages",
        "index1/step1/semi/voc20%/Annotations/instances_train.json",
    ),
    "cocoa_label_1step": (
        "index1/step1/semi/cocoa_label/images",
        "index1/step1/semi/cocoa_label/annotations/cocoa_active_label.json",
    ),
    "cocoa_unlabel_1step": (
        "index1/step1/semi/cocoa_unlabel/images",
        "index1/step1/semi/cocoa_unlabel/annotations/cocoa_active_unlabel.json",
    ),
    "voc_cocoa_train_2step": (
        "index1/step2/voc+coco_a/images",
        "index1/step2/voc+coco_a/annotations/voc+coco_a.json",
    ), #2step
    "voc_cocoa_20%_2step": (
        "index1/step2/semi/voc+cocoa_20%/images",
        "index1/step2/semi/voc+cocoa_20%/annotations/voc+coco_a_20%.json",
    ),
    "voc_cocoa_test_2step": (
        "index1/step2/voc+coco_a/test/images",
        "index1/step2/voc+coco_a/test/annotations/voc+coco_a-test.json",
    ),
    "cocob_train_2step": (
        "index1/step2/coco_b/images",
        "index1/step2/coco_b/annotations/coco_b.json",
    ),
    "cocob_label_2step": (
        "index1/step2/semi/cocob_label/images",
        "index1/step2/semi/cocob_label/annotations/cocob_active_label.json",
    ),
    "cocob_unlabel_2step": (
        "index1/step2/semi/cocob_unlabel/images",
        "index1/step2/semi/cocob_unlabel/annotations/cocob_active_unlabel.json",
    ),
    "cocob_val_2step": (
        "index1/step2/coco_b/val/images",
        "index1/step2/coco_b/val/annotations/coco_b_val.json",
    ),
    "voc_cocoa_cocob_train_3step": (
        "index1/step3/voc+coco_a+coco_b/images",
        "index1/step3/voc+coco_a+coco_b/annotations/voc+coco_a+coco_b.json",
    ), #3step
    "voc_cocoa_cocob_20%_3step": (
        "index1/step3/semi/voc+cocoa+cocob_20%/images",
        "index1/step3/semi/voc+cocoa+cocob_20%/annotations/voc+coco_a+coco_b_20%.json",
    ),
    "voc_cocoa_cocob_test_3step": (
        "index1/step3/voc+coco_a+coco_b/test/images",
        "index1/step3/voc+coco_a+coco_b/test/annotations/voc+coco_a+coco_b-test.json",
    ),
    "cococ_train_3step": (
        "index1/step3/coco_c/images",
        "index1/step3/coco_c/annotations/coco_c.json",
    ),
    "cococ_label_3step": (
        "index1/step3/semi/cococ_label/images",
        "index1/step3/semi/cococ_label/annotations/cococ_active_label.json",
    ),
    "cococ_unlabel_3step": (
        "index1/step3/semi/cococ_unlabel/images",
        "index1/step3/semi/cococ_unlabel/annotations/cococ_active_unlabel.json",
    ),
    "cococ_val_3step": (
        "index1/step3/coco_c/val/images",
        "index1/step3/coco_c/val/annotations/coco_c_val.json",
    ),
    "voc20%_seed0": (
        "index1/semi/voc20%/seed_0/JPEGImages",
        "index1/semi/voc20%/seed_0/Annotations/instances_train.json",
    ),
    "voc20%_seed5": (
        "index1/semi/voc20%/seed_5/JPEGImages",
        "index1/semi/voc20%/seed_5/Annotations/instances_train.json",
    ),
    "voc20%_seed9": (
        "index1/semi/voc20%/seed_9/JPEGImages",
        "index1/semi/voc20%/seed_9/Annotations/instances_train.json",
    ),
    "coco20_label": (
        "index1/semi/coco20_label/images",
        "index1/semi/coco20_label/annotations/coco20_active_label.json",
    ),
    "coco20_unlabel": (
        "index1/semi/coco20_unlabel/images",
        "index1/semi/coco20_unlabel/annotations/coco20_unlabel.json",
    ),
    "productline1_train": (
        "two_stage_datasets/index2/productline1/train/images",
        "two_stage_datasets/index2/productline1/train/annotations/instances_default.json",
    ), #index2
    "productline1_train_20_percent": (
        "index2/semi/productline1_20%/images",
        "index2/semi/productline1_20%/annotations/instances_default_20%.json",
    ),
    "productline1_test": (
        "two_stage_datasets/index2/productline1/test/images",
        "two_stage_datasets/index2/productline1/test/annotations/instances_default.json",
    ),
    "productline2_train": (
        "two_stage_datasets/index2/productline2/train/images",
        "two_stage_datasets/index2/productline2/train/annotations/instances_default_3cls.json",
    ),
    "productline2_label": (
        "index2/semi/productline2_label/images",
        "index2/semi/productline2_label/annotations/productline2_active_label.json",
    ),
    "productline2_test": (
        "two_stage_datasets/index2/productline2/test/images",
        "two_stage_datasets/index2/productline2/test/annotations/instances_default_3cls.json",
    ),
    "soda10m_train": (
        "two_stage_datasets/index3/soda_10m/train",
        "two_stage_datasets/index3/soda_10m/annotations/instance_train.json",
    ), #index3
    "soda_train_20_percent": (
        "index3/semi/soda_20%/images",
        "index3/semi/soda_20%/annotations/instance_train_20%.json",
    ),
    "soda10m_val": (
        "two_stage_datasets/index3/soda_10m/val",
        "two_stage_datasets/index3/soda_10m/annotations/instance_val.json",
    ),
    "trafficrecorder_train": (
        "two_stage_datasets/index3/trafficrecoder/train/images",
        "two_stage_datasets/index3/trafficrecoder/train/annotations/instances_default.json",
    ),
    "trafficrecorder_label": (
        "index3/semi/trafficrecorder_label/images",
        "index3/semi/trafficrecorder_label/annotations/trafficrecorder_active_label.json",
    ),
    "trafficrecorder_test": (
        "two_stage_datasets/index3/trafficrecoder/test/images",
        "two_stage_datasets/index3/trafficrecoder/test/annotations/instances_default.json",
    ),
    "cocoa_train": (
        "two_stage_datasets/index4/coco20_a/images/train",
        "two_stage_datasets/index4/coco20_a/annotations/instances_train.json",
    ), #index4
    "cocoa_train_20_percent": (
        "index4/semi/cocoa_20%/images",
        "index4/semi/cocoa_20%/annotations/instance_train_20%.json",
    ),
    "cocoa_val": (
        "two_stage_datasets/index4/coco20_a/images/val",
        "two_stage_datasets/index4/coco20_a/annotations/instances_val.json",
    ),
    "cocob_train": (
        "two_stage_datasets/index4/coco20_b/images/train",
        "two_stage_datasets/index4/coco20_b/annotations/instances_train.json",
    ),
    "cocob_label": (
        "index4/semi/cocob_label/images",
        "index4/semi/cocob_label/annotations/cocob_active_label.json",
    ),
    "cocob_unlabel": (
        "index4/semi/cocob_unlabel/images",
        "index4/semi/cocob_unlabel/annotations/cocob_active_unlabel.json",
    ),
    "cocob_val": (
        "two_stage_datasets/index4/coco20_b/images/val",
        "two_stage_datasets/index4/coco20_b/annotations/instances_val.json",
    ),
    "tacoa_train": (
        "two_stage_datasets/index5/taco_a/images/train",
        "two_stage_datasets/index5/taco_a/annotations/instances_train.json",
    ), #index5
    "tacoa_train_20_percent": (
        "index5/semi/tacoa_20%/images",
        "index5/semi/tacoa_20%/annotations/instance_train_20%.json",
    ),
    "tacoa_test": (
        "two_stage_datasets/index5/taco_a/images/test",
        "two_stage_datasets/index5/taco_a/annotations/instances_test.json",
    ),
    "tacob_train": (
        "two_stage_datasets/index5/taco_b/images/train",
        "two_stage_datasets/index5/taco_b/annotations/instances_train.json",
    ),
    "tacob_label": (
        "index5/semi/tacob_label/images",
        "index5/semi/tacob_label/annotations/tacob_active_label.json",
    ),
    "tacob_test": (
        "two_stage_datasets/index5/taco_b/images/test",
        "two_stage_datasets/index5/taco_b/annotations/instances_test.json",
    ),
    "voc2007_trainval_index6": (
        "two_stage_datasets/index6/voc_2007/JPEGImages",
        "two_stage_datasets/index6/voc_2007/Annotations/instances_train.json",
    ), #index6
    "voc2007_trainval_index6_20_percent": (
        "index6/semi/voc07_20%/images",
        "index6/semi/voc07_20%/annotations/instance_train_20%.json",
    ),
    "voc2007_test_index6": (
        "two_stage_datasets/index6/voc_2007/test/JPEGImages",
        "two_stage_datasets/index6/voc_2007/test/Annotations/instances_train.json",
    ),
    "voc2012_train_index6": (
        "two_stage_datasets/index6/voc_2012/JPEGImages",
        "two_stage_datasets/index6/voc_2012/Annotations/instances_train.json",
    ),
    "voc2012_index6_label": (
        "index6/semi/voc12_label/images",
        "index6/semi/voc12_label/annotations/voc12_active_label.json",
    ),
    "voc2012_val_index6": (
        "two_stage_datasets/index6/voc_2012/val/JPEGImages",
        "two_stage_datasets/index6/voc_2012/val/Annotations/instances_train.json",
    ),
    "voc5_train_index7": (
        "two_stage_datasets/index7/voc07_12_5/JPEGImages/trainval",
        "two_stage_datasets/index7/voc07_12_5/Annotations/instances_trainval.json",
    ), #index7
    "voc5_index7_20_percent": (
        "index7/semi/voc5_20%/images",
        "index7/semi/voc5_20%/annotations/instance_trainval_20%.json",
    ),
    "voc5_test_index7": (
        "two_stage_datasets/index7/voc07_12_5/test/JPEGImages",
        "two_stage_datasets/index7/voc07_12_5/test/Annotations/instances_test.json",
    ),
    "coco5_train_index7": (
        "two_stage_datasets/index7/coco5/images/train2017",
        "two_stage_datasets/index7/coco5/annotations/instances_train2017_new.json",
    ),
    "coco5_index7_label": (
        "index7/semi/coco5_label/images",
        "index7/semi/coco5_label/annotations/coco5_active_label.json",
    ),
    "coco5_index7_unlabel": (
        "index7/semi/coco5_unlabel/images",
        "index7/semi/coco5_unlabel/annotations/coco5_active_unlabel.json",
    ),
    "coco5_val_index7": (
        "two_stage_datasets/index7/coco5/images/val2017",
        "two_stage_datasets/index7/coco5/annotations/instances_val2017_new.json",
    ),
    "voc5_train_index8": (
        "two_stage_datasets/index8/voc_07_12_2/JPEGImages/trainval",
        "two_stage_datasets/index8/voc_07_12_2/Annotations/instances_trainval.json",
    ), #index8
    "voc5_index8_20_percent": (
        "index8/semi/voc5_20%/images",
        "index8/semi/voc5_20%/annotations/instances_trainval_20%.json",
    ),
    "voc5_test_index8": (
        "two_stage_datasets/index8/voc_07_12_2/test/JPEGImages",
        "two_stage_datasets/index8/voc_07_12_2/test/Annotations/instances_test.json",
    ),
    "object365_train_index8": (
        "two_stage_datasets/index8/objects365_5/Images/train",
        "two_stage_datasets/index8/objects365_5/Annotations/instances_train.json",
    ),
    "object365_index8_label": (
        "index8/semi/objects365_label/images",
        "index8/semi/objects365_label/annotations/object365_active_label.json",
    ),
    "object365_index8_unlabel": (
        "index8/semi/objects365_unlabel/images",
        "index8/semi/objects365_unlabel/annotations/object365_active_unlabel.json",
    ),
    "object365_val_index8": (
        "two_stage_datasets/index8/objects365_5/Images/val",
        "two_stage_datasets/index8/objects365_5/Annotations/instances_val.json",
    ),
    "exdark12_train": (
        "two_stage_datasets/index9/exdark/images/train",
        "two_stage_datasets/index9/exdark/annotations/exdark_train.json",
    ), #index9
    "exdark12_20_percent": (
        "index9/semi/exdark_20%/images",
        "index9/semi/exdark_20%/annotations/exdark_train_20%.json",
    ),
    "exdark12_label": (
        "index9/semi/exdark12_label/images",
        "index9/semi/exdark12_label/annotations/exdark12_active_label.json",
    ),
    "exdark12_val": (
        "two_stage_datasets/index9/exdark/images/val",
        "two_stage_datasets/index9/exdark/annotations/exdark_val.json",
    ),
    "coco12_train": (
        "two_stage_datasets/index9/coco12/images/train2017",
        "two_stage_datasets/index9/coco12/annotations/instances_train2017.json",
    ),
    "coco12_20_percent": (
        "index9/semi/coco12_20%/images",
        "index9/semi/coco12_20%/annotations/instances_train2017_20%.json",
    ),
    "coco12_label": (
        "index9/semi/coco12_label/images",
        "index9/semi/coco12_label/annotations/coco12_active_label.json",
    ),
    "coco12_unlabel": (
        "index9/semi/coco12_unlabel/images",
        "index9/semi/coco12_unlabel/annotations/coco12_active_unlabel.json",
    ),
    "coco12_val": (
        "two_stage_datasets/index9/coco12/images/val2017",
        "two_stage_datasets/index9/coco12/annotations/instances_val2017.json",
    ),
    "exdark11_train": (
        "two_stage_datasets/index10/exdark/images/train",
        "two_stage_datasets/index10/exdark/annotations/exdark_train.json",
    ), #index10
    "exdark11_20_percent": (
        "index10/semi/exdark_20%/images",
        "index10/semi/exdark_20%/annotations/exdark_train_20%.json",
    ),
    "exdark11_val": (
        "two_stage_datasets/index10/exdark/images/val",
        "two_stage_datasets/index10/exdark/annotations/exdark_val.json",
    ),
    "voc11_train": (
        "two_stage_datasets/index10/voc_07_12_11/JPEGImages/trainval",
        "two_stage_datasets/index10/voc_07_12_11/Annotations/instances_trainval.json",
    ),
    "voc11_label": (
        "index10/semi/voc11_label/images",
        "index10/semi/voc11_label/annotations/voc11_active_label.json",
    ),
    "voc11_unlabel": (
        "index10/semi/voc11_unlabel/images",
        "index10/semi/voc11_unlabel/annotations/voc11_active_unlabel.json",
    ),
    "voc11_test": (
        "two_stage_datasets/index10/voc_07_12_11/test/JPEGImages",
        "two_stage_datasets/index10/voc_07_12_11/test/Annotations/instances_test.json",
    ),
}

_SPLITS_SODA_FORMAT = {}
_SPLITS_SODA_FORMAT["soda"] = {
    "soda_train": (
        "SODA/SSLAD-2D/labeled/train",
        "SODA/SSLAD-2D/labeled/annotations/instance_train.json",
    ),
    "soda_val": (
        "SODA/SSLAD-2D/labeled/val",
        "SODA/SSLAD-2D/labeled/annotations/instance_val.json",
    ),
}

_SPLITS_OBJECTS365_5_FORMAT = {}
_SPLITS_OBJECTS365_5_FORMAT["objects365_5"] = {
    "objects365_5_train": (
        "objects365_5/Images/train",
        "objects365_5/Annotations/instances_train.json",
    ),
    "objects365_5_val": (
        "objects365_5/Images/val",
        "objects365_5/Annotations/instances_val.json",
    ),
}

_SPLITS_OBJECTS365_5_NEW_FORMAT = {}
_SPLITS_OBJECTS365_5_NEW_FORMAT["objects365_5_new"] = {
    "objects365_5_new_train": (
        "objects365_5_new/Images/train",
        "objects365_5_new/Annotations/instances_train.json",
    ),
    "objects365_5_new_val": (
        "objects365_5_new/Images/val",
        "objects365_5_new/Annotations/instances_val.json",
    ),
}


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts

def register_coco5(root):
    for _, splits_per_dataset in _SPLITS_COCO5_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_voc5(root):
    for _, splits_per_dataset in _SPLITS_VOC5_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_coco20(root):
    for _, splits_per_dataset in _SPLITS_COCO20_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_soda(root):
    for _, splits_per_dataset in _SPLITS_SODA_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_objects365_5(root):
    for _, splits_per_dataset in _SPLITS_OBJECTS365_5_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def register_objects365_5_new(root):
    for _, splits_per_dataset in _SPLITS_OBJECTS365_5_NEW_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                {},
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_unlabel(_root)
register_coco5(_root)
register_voc5(_root)
register_coco20(_root)
register_soda(_root)
register_objects365_5(_root)
register_objects365_5_new(_root)
