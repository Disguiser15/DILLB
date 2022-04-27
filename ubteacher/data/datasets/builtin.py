# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets import register_coco_instances
from ubteacher.data.datasets.coco import register_coco_instances
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging

logger = logging.getLogger(__name__)

_SPLITS_COCO20_FORMAT = {}
_SPLITS_COCO20_FORMAT["coco20"] = {
    "bdd_train": (
        "bdd100k/images/train",
        "bdd100k/annotations/bdd100k_labels_images_det_coco_train.json",
    ), #bdd->cityscapes
    "bdd_train_1%": (
        "index11/semi/bdd1%/images",
        "index11/semi/bdd1%/annotations/bdd_1%.json",
    ),
    "bdd_train_1%_source": (
        "index11/semi/bdd1%/images",
        "index11/semi/bdd1%/annotations/bdd_1%_source.json",
    ),
    "bdd_val": (
        "bdd100k/images/val",
        "bdd100k/annotations/bdd100k_labels_images_det_coco_val.json",
    ),
    "bdd_val_source": (
        "bdd100k/images/val",
        "bdd100k/annotations/bdd100k_labels_images_det_coco_val_source.json",
    ),
    "cityscape_train": (
        "Cityscapes_new/JPEGImages/train_s",
        "Cityscapes_new/Annotations/instances_train_s.json",
    ),
    "cityscape_train_1%": (
        "index11/semi/cityscapes1%/images",
        "index11/semi/cityscapes1%/annotations/cityscapes_1%.json",
    ),
    "cityscape_train_1%_source": (
        "index11/semi/cityscapes1%/images",
        "index11/semi/cityscapes1%/annotations/cityscapes_1%_source.json",
    ),
    "cityscape_train_5%": (
        "index11/semi/cityscapes5%/images",
        "index11/semi/cityscapes5%/annotations/cityscapes_5%.json",
    ),
    "cityscape_train_5%_target": (
        "index11/semi/cityscapes5%/images",
        "index11/semi/cityscapes5%/annotations/cityscapes_5%_target.json",
    ),
    "cityscape_train_10%": (
        "index11/semi/cityscapes10%/images",
        "index11/semi/cityscapes10%/annotations/cityscapes_10%.json",
    ),
    "cityscape_train_10%_target": (
        "index11/semi/cityscapes10%/images",
        "index11/semi/cityscapes10%/annotations/cityscapes_10%_target.json",
    ),
    "cityscape_train_20%": (
        "index11/semi/cityscapes20%/images",
        "index11/semi/cityscapes20%/annotations/cityscapes_20%.json",
    ),
    "cityscape_train_20%_target": (
        "index11/semi/cityscapes20%/images",
        "index11/semi/cityscapes20%/annotations/cityscapes_20%_target.json",
    ),
    "cityscape_test": (
        "Cityscapes_new/JPEGImages/test_s",
        "Cityscapes_new/Annotations/instances_test_s.json",
    ),
    "cityscape_test_target": (
        "Cityscapes_new/JPEGImages/test_s",
        "Cityscapes_new/Annotations/instances_test_s_target.json",
    ),
    "foggycityscape_train": (
        "Cityscapes_new/JPEGImages/train_t",
        "Cityscapes_new/Annotations/instances_train_t.json",
    ),
    "foggycityscape_train_10%": (
        "index11/semi/foggycityscapes10%/images",
        "index11/semi/foggycityscapes10%/annotations/foggycityscapes_10%.json",
    ),
    "foggycityscape_train_10%_target": (
        "index11/semi/foggycityscapes10%/images",
        "index11/semi/foggycityscapes10%/annotations/foggycityscapes_10%_target.json",
    ),
    "foggycityscape_test": (
        "Cityscapes_new/JPEGImages/test_t",
        "Cityscapes_new/Annotations/instances_test_t.json",
    ),
    "exdark12_train": (
        "two_stage_datasets/index9/exdark/images/train",
        "two_stage_datasets/index9/exdark/annotations/exdark_train.json",
    ), #coco->exdark
    "exdark12_train_1%": (
        "index9/semi/exdark1%/images",
        "index9/semi/exdark1%/annotations/exdark_1%.json",
    ),
    "exdark12_train_1%_source": (
        "index9/semi/exdark1%/images",
        "index9/semi/exdark1%/annotations/exdark_1%_source.json",
    ),
    "exdark12_train_5%": (
        "index9/semi/exdark5%/images",
        "index9/semi/exdark5%/annotations/exdark_5%.json",
    ),
    "exdark12_train_5%_target": (
        "index9/semi/exdark5%/images",
        "index9/semi/exdark5%/annotations/exdark_5%_target.json",
    ),
    "exdark12_train_10%": (
        "index9/semi/exdark10%/images",
        "index9/semi/exdark10%/annotations/exdark_10%.json",
    ),
    "exdark12_train_10%_target": (
        "index9/semi/exdark10%/images",
        "index9/semi/exdark10%/annotations/exdark_10%_target.json",
    ),
    "exdark12_train_20%": (
        "index9/semi/exdark20%/images",
        "index9/semi/exdark20%/annotations/exdark_20%.json",
    ),
    "exdark12_train_20%_target": (
        "index9/semi/exdark20%/images",
        "index9/semi/exdark20%/annotations/exdark_20%_target.json",
    ),
    "exdark12_val": (
        "two_stage_datasets/index9/exdark/images/val",
        "two_stage_datasets/index9/exdark/annotations/exdark_val.json",
    ),
    "exdark12_val_target": (
        "two_stage_datasets/index9/exdark/images/val",
        "two_stage_datasets/index9/exdark/annotations/exdark_val_target.json",
    ),
    "coco12_train": (
        "two_stage_datasets/index9/coco12/images/train2017",
        "two_stage_datasets/index9/coco12/annotations/instances_train2017.json",
    ),
    "coco12_train_1%": (
        "index9/semi/coco1%/images",
        "index9/semi/coco1%/annotations/coco_1%.json",
    ),
    "coco12_train_1%_source": (
        "index9/semi/coco1%/images",
        "index9/semi/coco1%/annotations/coco_1%_source.json",
    ),
    "coco12_train_5%": (
        "index9/semi/coco5%/images",
        "index9/semi/coco5%/annotations/coco_5%.json",
    ),
    "coco12_train_5%_source": (
        "index9/semi/coco5%/images",
        "index9/semi/coco5%/annotations/coco_5%_source.json",
    ),
    "coco12_val": (
        "two_stage_datasets/index9/coco12/images/val2017",
        "two_stage_datasets/index9/coco12/annotations/instances_val2017.json",
    ),
    "coco12_val_source": (
        "two_stage_datasets/index9/coco12/images/val2017",
        "two_stage_datasets/index9/coco12/annotations/instances_val2017_source.json",
    ),
}


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

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco20(_root)