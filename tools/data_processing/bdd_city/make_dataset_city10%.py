## 加载json文件
import json
import random
import math
import numpy as np
import re
from tqdm import tqdm
import os

def pick_dataset():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(args.input_dir,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    print("The number of original annotations:", len(json_data['annotations']))
    print("The number of original pictures:", len(json_data['images']))
    json_data_new = {}
    json_data_new_annotations=[]
    json_data_new_images = json_data['images']
    random.seed(0)
    sample_num = math.floor(args.pick_percent*len(json_data_new_images))
    json_data_new_images = random.sample(json_data_new_images, sample_num)

    json_data_new['images'] = json_data_new_images
    imgidlist = list()
    for i in range(len(json_data_new['images'])):
        imgidlist.append(json_data_new['images'][i]['id'])
    json_data_new['categories'] = json_data['categories']
    for i in json_data['annotations']:
        if i['image_id'] in imgidlist:
            json_data_new_annotations.append(i)

    json_data_new['annotations'] = json_data_new_annotations
    print("The number of annotations processed:", len(json_data_new['annotations']))
    print("The number of pictures processed:", len(json_data_new['images']))

    json.dump(json_data_new, open(args.save_dir+'cityscapes_10%.json', 'w'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='pick dataset of specific percentage')
    parser.add_argument("--input-dir",type=str,default='./datasets/Cityscapes_new/Annotations/instances_train_s.json')
    parser.add_argument("--save-dir",type=str,default='./datasets/index11/semi/cityscapes10%/annotations/')
    parser.add_argument("--pick-percent",type=float,default=0.1)
    args = parser.parse_args()
    pick_dataset()