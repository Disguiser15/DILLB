import json
import os
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data import detection_utils as utils
from detectron2.layers.nms import batched_nms

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.config import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel




@torch.no_grad()
def uncertainty_entropy(p):
    # p.size() = num_instances of a image, num_classes
    p = F.softmax(p, dim=1)
    p = - torch.log2(p) * p
    entropy_instances = torch.sum(p, dim=1)
    # set uncertainty of image eqs the mean uncertainty of instances
    entropy_image = torch.mean(entropy_instances)
    return entropy_image


data_hook = {}
def box_predictor_hooker(m, i, o):
    data_hook['scores_hooked'] = o[0].clone().detach()
    data_hook['boxes_hooked'] = o[1].clone().detach()


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
    assert cfg.SEMISUPNET.Trainer == "ubteacher", "Trainer Name must be ubteacher."
    Trainer = UBTeacherTrainer
    assert args.eval_only is True, "Inference should be eval only."
    inference(Trainer, cfg)



def inference(Trainer, cfg):
    print('Loading the Model named: ', cfg.MODEL.WEIGHTS)
    model = Trainer.build_model(cfg)
    model_teacher = Trainer.build_model(cfg)
    ensem_ts_model = EnsembleTSModel(model_teacher, model)

    DetectionCheckpointer(
        ensem_ts_model, save_dir=cfg.OUTPUT_DIR
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)


    ensem_ts_model.modelTeacher.roi_heads.box_predictor.register_forward_hook(box_predictor_hooker)
    ensem_ts_model.modelTeacher.eval()
    ensem_ts_model.modelTeacher.training = False
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    dic={}
    from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
    for j,item in enumerate(dataset_dicts):
        file_name = item['file_name']
        print(j,file_name)
        image = utils.read_image(file_name, format='BGR')
        image = torch.from_numpy(image.copy()).permute(2,0,1)
        res, proposal_boxes = ensem_ts_model.modelTeacher.inference_NMS([{'image':image}])

        # score
        scores = data_hook['scores_hooked'].to(torch.device("cpu"))
        boxes_devidation = data_hook['boxes_hooked'].to(torch.device("cpu"))
        proposal_boxes = proposal_boxes.to(torch.device("cpu"))
        scores_softmax = F.softmax(scores, dim=1)
        # for each bbox, grasp the most possible category
        scores_best = list()
        for i in range(scores_softmax.size(0)):
            category = torch.argmax(scores_softmax[i]).item()
            scores_best.append(category)
        
        scores_best_numpy = np.array(scores_best)
        idxs = torch.from_numpy(scores_best_numpy)

        scores_remove_background = torch.zeros(1, 1)
        bbox_devidation_remove_background = torch.zeros(1, 4)
        count = 0
        for i in range(len(scores_best)-1, -1, -1):
            if scores_best[i] == scores.size(1)-1:
                scores = torch.cat((scores[ :i], scores[i+1: ]), dim=0) # delete background category's scores distribution
                scores_softmax = torch.cat((scores_softmax[ :i], scores_softmax[i+1: ]), dim=0) # delete background category's scores distribution
                proposal_boxes = torch.cat((proposal_boxes[ :i], proposal_boxes[i+1: ]), dim=0) # delete background category's bbox
                boxes_devidation = torch.cat((boxes_devidation[ :i], boxes_devidation[i+1: ]), dim=0) # delete background category's bbox devidation
                idxs = torch.cat((idxs[ :i], idxs[i+1: ]), dim=0) # delete background category's idxs
            else:
                # grasp the most possible category's score and bbox devidation
                if count == 0:
                    bbox_devidation_remove_background = boxes_devidation[i][scores_best[i]*4:(scores_best[i]+1)*4].unsqueeze(0)
                    scores_remove_background = scores_softmax[i][scores_best[i]].unsqueeze(0)
                    count = count + 1
                else:
                    bbox_devidation_remove_background = torch.cat((boxes_devidation[i][scores_best[i]*4:(scores_best[i]+1)*4].unsqueeze(0), bbox_devidation_remove_background), dim=0)
                    scores_remove_background = torch.cat((scores_softmax[i][scores_best[i]].unsqueeze(0), scores_remove_background), dim=0)
                    count = count + 1
        
        boxes = proposal_boxes + bbox_devidation_remove_background # bounding box coordinates
        
        keep = batched_nms(boxes, scores_remove_background, idxs, iou_threshold=0.5)

        # select the score distribution after NMS operation
        if keep.size(0)!=0:
            scores_nms = torch.zeros(1, scores.size(1))
            count = 0
            for i in range(keep.size(0)):
                if count == 0:
                    scores_nms = scores[(keep[i]).item()].unsqueeze(0)
                    count = count + 1
                else:
                    scores_nms = torch.cat((scores_nms, scores[(keep[i]).item()].unsqueeze(0)), dim=0)
        else:
            scores_nms = scores
                        
        entropy = uncertainty_entropy(scores_nms)

        image_shape = (item['height'], item['width'])
        gt = Instances(image_shape)
        gt_bboxes = []
        gt_classes = []
        for annota in item['annotations']:
            x,y,w,h=annota['bbox']
            bbox=[x,y,x+w,y+h]
            gt_bboxes.append(bbox)
            gt_classes.append(annota['category_id'])
        gt_bboxes=Boxes(torch.tensor(gt_bboxes))
        gt_classes=torch.tensor(gt_classes)
        gt.pred_boxes=gt_bboxes
        gt.pred_classes=gt_classes
        gt.scores=torch.ones_like(gt_classes)
        match_quality_matrix = pairwise_iou(
            res[0]['instances'].pred_boxes, gt.pred_boxes.to(res[0]['instances'].pred_boxes.device)
        )
        if (len(res[0]['instances']))>0:
            iou_scores,idx = match_quality_matrix.max(dim=1)
            # iou_scores = iou_scores.squeeze()
            # idx = idx.squeeze()
            # print(iou_scores.shape)
            matched_classes = gt_classes[idx]
            matched_boxes = gt_bboxes.tensor[idx].squeeze()
        dic[file_name]=[]
        for i in range(len(res[0]['instances'])):
            box_info = {'confidence score':np.float(res[0]['instances'].scores.detach().numpy()[i]),
                        'iou with gt':np.float(iou_scores.detach().numpy()[i]),
                        'pred class':np.int(res[0]['instances'].pred_classes.detach().numpy()[i]),
                        'gt class':np.int(matched_classes.detach().numpy()[i]),
                        'pred box':res[0]['instances'].pred_boxes.tensor[i].detach().numpy().tolist(),
                        'gt box':matched_boxes[i].numpy().tolist(),
                        'entropy': entropy.detach().clone().item()
                        }
            dic[file_name].append(box_info)

        del res
        del gt

    with open(FILE_PATH, 'w') as f:
        f.write(json.dumps(dic))


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument("--static-file",type=str,default='temp/temp.json')
    parser.add_argument("--model-weights",type=str,default='output/model_best.pth')
    args = parser.parse_args()
    FILE_PATH = args.static_file #Json file of the intermediate process which you will use in tools/ScoreFuction.py
    #args.config_file = 'configs/voc/voc07+voc12_sup15_run1.yaml' #the config file you used to train this inference model
    args.eval_only = True
    args.resume = True
    # args.num_gpus = 1
    # you should config MODEL.WEIGHTS and keep other hyperparameters default(Odd-numbered items are keys, even-numbered items are values)
    args.opts = ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.5,'MODEL.ROI_HEADS.NMS_THRESH_TEST', 0.5,
     'TEST.DETECTIONS_PER_IMAGE', 20, 'INPUT.FORMAT', 'RGB','MODEL.WEIGHTS',args.model_weights,'MODEL.DEVICE','cpu']
    print("Command Line Args:", args)
    main(args)