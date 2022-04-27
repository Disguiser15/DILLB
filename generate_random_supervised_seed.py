from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from ubteacher import add_ubteacher_config
from ubteacher.data.datasets import builtin
import json
from detectron2.data.build import (
    get_detection_dataset_dicts,
)

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

def generate(cfg):
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
    print('len of ditaset:',len(dataset_dicts))
    try:
        with open(cfg.DATALOADER.RANDOM_DATA_SEED_PATH,'r') as f:
                dic=json.load(f)
    except:
        dic={}

    dic[str(cfg.DATALOADER.SUP_PERCENT)] = {}
    seeds = [int(i) for i in args.random_seeds.split(',')]
    for i in range(10):
        arr = generate_supervised_seed(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            seeds[i]
        )
        print(len(arr))
        dic[str(cfg.DATALOADER.SUP_PERCENT)][str(i)] = arr
    with open(cfg.DATALOADER.RANDOM_DATA_SEED_PATH,'w') as f:
        f.write(json.dumps(dic))


def main(args):
    cfg = setup(args)
    generate(cfg)

def generate_supervised_seed(
    dataset_dicts, SupPercent, seed
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    arr = range(num_all)
    import random
    random.seed(seed)
    return random.sample(arr,num_label)

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--random-seeds",type=str,default="0,1,2,3,4,5,6,7,8,9") #需要设置10次随机数种子，以','分割
    args = parser.parse_args()
    args.config_file = 'configs/dillb/bdd1_city10.yaml'
    # args.config_file = 'configs/soda/soda_sup26_run1.yaml'
    main(args)
