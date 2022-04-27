# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from ..layer.da_heads import DomainAdaptationModule

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results
    
    def inference_NMS(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            ), proposals[0]._fields['proposal_boxes'].tensor
        else:
            return results

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN_DAHEADS(GeneralizedRCNN):
    def __init__(self, cfg):   
        super().__init__(cfg)
        self.cfg = cfg.clone()
        self.da_heads = DomainAdaptationModule(self.cfg)

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":

            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses, da_ins_feature, proposals = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )
            if 'is_source' in batched_inputs[0]:
                da_ins_labels = torch.zeros(len(batched_inputs) * self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, dtype=torch.float32)
                masks = []
                for i in range(len(batched_inputs)):
                    if batched_inputs[i]['is_source']==1:
                        da_ins_labels[i*self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE : (i+1)*self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE] = 1
            proposals_boxlist = [i._fields['proposal_boxes'] for i in proposals]
            features_list = [i for i in features.values()]
            targets = [i['instances']._fields['gt_boxes'] for i in batched_inputs]
            # da_losses = self.da_heads(proposals, features, da_ins_feature, da_ins_labels, da_proposals, targets, batched_inputs)
            da_losses = self.da_heads(proposals_boxlist, features_list, da_ins_feature, da_ins_labels, proposals_boxlist, targets, batched_inputs)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses, [], [], None
        elif branch == "unsup_data_weak":
            # img_grl_fea = [self.grl_img(fea) for fea in list(features.values())]
            # img_grl_consist_fea = [self.grl_img_consist(fea) for fea in list(features.values())]
            # da_img_features = self.imghead(img_grl_fea)
            # da_img_consist_features = self.imghead(img_grl_consist_fea)
            # da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions, da_ins_feature, proposals = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )


            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses, da_ins_feature, proposals = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _, da_ins_feature, proposals = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results
    
    def inference_NMS(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _, da_ins_feature, proposals = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            ), proposals[0]._fields['proposal_boxes'].tensor
        else:
            return results