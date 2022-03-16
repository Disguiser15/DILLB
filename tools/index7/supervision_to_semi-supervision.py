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
        new['model']['modelStudent.'+k]=v
        new['model']['modelTeacher.'+k]=v

    torch.save(new,args.save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='supervised checkpoint to semi-supervised checkpoint')
    parser.add_argument("--input-dir",type=str,default='output/index7/baseline/voc5_sup100/model_best.pth')
    parser.add_argument("--save-dir",type=str,default='output/index7/baseline/voc5_sup100/one_stage.pth')
    args = parser.parse_args()
    sup_to_semi()