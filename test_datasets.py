from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from ubteacher import add_ubteacher_config
import json
from detectron2.data.build import (
    get_detection_dataset_dicts,
)
import ubteacher.data.datasets.builtin
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import random

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TEST,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    # dataset_dicts2 = get_detection_dataset_dicts(
    #     cfg.DATASETS.TEST,
    #     filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
    #     min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
    #     if cfg.MODEL.KEYPOINT_ON
    #     else 0,
    #     proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
    #     if cfg.MODEL.LOAD_PROPOSALS
    #     else None,
    # )
    # table = {}
    # for d in dataset_dicts2:
    #     table[d['file_name'].split('/')[-1]]=d
    for d in random.sample(dataset_dicts, 20):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN))
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite('temp/'+d['file_name'].split('/')[-1],out.get_image()[:, :, ::-1])
        # img = cv2.imread('datasets/coco/val2017/'+d['file_name'].split('/')[-1])
        # visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get('coco_2017_val'))
        # out = visualizer.draw_dataset_dict(table[d['file_name'].split('/')[-1]])
        # cv2.imwrite('temp/gt_'+d['file_name'].split('/')[-1],out.get_image()[:, :, ::-1])

if __name__ == '__main__':
    parser = default_argument_parser()
    args = parser.parse_args()
    args.config_file = 'configs/voc5/baseline/voc5_sup100_run1.yaml'
    args.config_file = 'configs/coco20/baseline/coco20_sup100_run1.yaml'
    args.config_file = 'configs/soda/baseline/soda_sup100_run1.yaml'
    args.config_file = 'configs/objects365_5/baseline/objects365_5_sup100_run1.yaml'
    main(args)