## 加载json文件
import json
import random
import math
import numpy as np
import re
from tqdm import tqdm

with open('./datasets/two_stage_datasets/index9/exdark/annotations/exdark_train.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
print("原标注数:", len(json_data['annotations']))
print("原图片数", len(json_data['images']))
json_data_new = {}
json_data_new_images=[]
json_data_new_annotations=[]
for i in tqdm(json_data['images']):
    for j in json_data['annotations']:
        if i['id'] == j['image_id']:
            json_data_new_images.append(i)
            break
print("有效图片数", len(json_data_new_images))
random.seed(0)
sample_num = math.floor(0.2*len(json_data_new_images))
json_data_new_images = random.sample(json_data_new_images, sample_num)

json_data_new['images'] = json_data_new_images
json_data_new['categories'] = json_data['categories']
for i in json_data['annotations']:
    for j in json_data_new['images']:
        if i['image_id'] == j['id']:
            json_data_new_annotations.append(i)
            break

json_data_new['annotations'] = json_data_new_annotations
print("处理后标注数:", len(json_data_new['annotations']))
print("处理后图片数", len(json_data_new['images']))

json.dump(json_data_new, open('./datasets/index9/semi/exdark_20%/annotations/exdark_train_20%.json', 'w'))