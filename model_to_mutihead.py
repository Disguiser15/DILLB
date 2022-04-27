import torch
import collections

def sup_to_semi():
    old=torch.load(args.input_dir)
    for i in old['model']:
        print(i)

    new=collections.OrderedDict()
    new['model']={}

    for k,v in old['model'].items():
        if 'box_head' in k:
            new['model']['modelStudent.' + k] = v
            new['model']['modelTeacher.' + k] = v
            new['model']['modelStudent.' + k.replace('box_head', 'box_head_target')] = v
            new['model']['modelTeacher.' + k.replace('box_head', 'box_head_target')] = v
        else:
            new['model']['modelStudent.'+k]=v
            new['model']['modelTeacher.'+k]=v
    torch.save(new,args.save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='supervised checkpoint to semi-supervised checkpoint')
    parser.add_argument("--input-dir",type=str,default='./model_best.pth')
    parser.add_argument("--save-dir",type=str,default='./model_mutihead.pth')
    args = parser.parse_args()
    sup_to_semi()