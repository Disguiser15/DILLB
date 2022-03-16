#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel


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
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    else:
        Trainer = BaselineTrainer

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer in ["ubteacher"]:
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # args.config_file = 'configs/voc5/baseline/voc5_sup100_run1.yaml'
    # # # args.config_file = 'configs/voc/pick/voc07+voc12_sup25_run1.yaml'
    # #args.config_file = 'configs/coco_supervision/baseline/faster_rcnn_R_50_FPN_sup100_run1.yaml'
    # # args.dist_url='tcp://127.0.0.1:1237'
    # args.num_gpus = 1
    # args.eval_only = True
    # #args.opts = ['SOLVER.IMG_PER_BATCH_LABEL', '8', 'SOLVER.IMG_PER_BATCH_UNLABEL', '8',  'MODEL.WEIGHTS', 'output/voc07+voc12/pick/voc07+voc12_sup25_pick_box_number20+random5_8/model_best.pth']
    # args.opts = ['SOLVER.IMG_PER_BATCH_LABEL', '16', 'SOLVER.IMG_PER_BATCH_UNLABEL', '16',  'MODEL.WEIGHTS', 'output/pick/imagenet_sup20_pick_all_indicator10+random10_16/model_best.pth']
    # # args.opts = ['OUTPUT_DIR',"output/test/imagenet_sup10_pick10_16"]
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
