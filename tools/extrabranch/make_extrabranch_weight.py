import torch
import collections

def sup_to_semi():
    old=torch.load(args.input_dir)
    for i in old['model']:
        print(i)

    new=collections.OrderedDict()
    new['model']={}

    for k,v in old['model'].items():
        # if k!='pixel_mean' and k !='pixel_std':
        if 'roi_heads' not in k and 'proposal_generator' not in k:
            new['model'][k]=v

    torch.save(new,args.save_dir)
    for i in new['model']:
        print(i)

if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    import argparse
    parser = argparse.ArgumentParser(description='supervised checkpoint to semi-supervised checkpoint')
    parser.add_argument("--input-dir",type=str,default='output/baseline/bdd/model_best.pth')
    parser.add_argument("--save-dir",type=str,default='output/baseline/bdd/model_best_without_rpn_roi.pth')
    args = parser.parse_args()
    sup_to_semi()