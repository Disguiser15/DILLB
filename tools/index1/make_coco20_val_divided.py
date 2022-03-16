## 加载json文件
import json
import random
import math
import numpy as np
import re
from tqdm import tqdm

with open('./datasets/index1/coco20/annotations/instances_val2017_new.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
print("原标注数:", len(json_data['annotations']))
print("原图片数", len(json_data['images']))
random.seed(0)
random.shuffle(json_data['images'])
json_data_new1 = dict()
json_data_new1['images'] = list()
imgidlist1 = list()
json_data_new2 = dict()
json_data_new2['images'] = list()
imgidlist2 = list()
json_data_new3 = dict()
json_data_new3['images'] = list()
imgidlist3 = list()
for i in tqdm(range(math.floor(len(json_data['images'])/3))):
    json_data_new1['images'].append(json_data['images'][i])
    imgidlist1.append(json_data['images'][i]['id'])
print(len(json_data_new1['images']))
for i in tqdm(range(math.floor(len(json_data['images'])/3), math.floor(len(json_data['images'])*2/3))):
    json_data_new2['images'].append(json_data['images'][i])
    imgidlist2.append(json_data['images'][i]['id'])
print(len(json_data_new2['images']))
for i in tqdm(range(math.floor(len(json_data['images'])*2/3), len(json_data['images']))):
    json_data_new3['images'].append(json_data['images'][i])
    imgidlist3.append(json_data['images'][i]['id'])
print(len(json_data_new3['images']))
anno_base1 = list()
anno_base2 = list()
anno_base3 = list()
for i in tqdm(range(len(json_data['annotations']))):
    if json_data['annotations'][i]['image_id'] in imgidlist1:
        anno_base1.append(json_data['annotations'][i])
    if json_data['annotations'][i]['image_id'] in imgidlist2:
        anno_base2.append(json_data['annotations'][i])
    if json_data['annotations'][i]['image_id'] in imgidlist3:
        anno_base3.append(json_data['annotations'][i])
json_data_new1['annotations'] = anno_base1
json_data_new2['annotations'] = anno_base2
json_data_new3['annotations'] = anno_base3
print(len(json_data_new1['annotations']))
print(len(json_data_new2['annotations']))
print(len(json_data_new3['annotations']))
json_data_new1['categories'] = json_data['categories']
json_data_new2['categories'] = json_data['categories']
json_data_new3['categories'] = json_data['categories']
json.dump(json_data_new1, open('./datasets/index1/step1/coco_a/val/annotations/coco_a_val.json', 'w'))
json.dump(json_data_new2, open('./datasets/index1/step2/coco_b/val/annotations/coco_b_val.json', 'w'))
json.dump(json_data_new3, open('./datasets/index1/step3/coco_c/val/annotations/coco_c_val.json', 'w'))