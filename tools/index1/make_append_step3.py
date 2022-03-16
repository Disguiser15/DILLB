## 加载json文件
import json
import random
import math
import numpy as np
import re
import os
import shutil
from tqdm import tqdm

with open('./datasets/index1/step2/coco_b/annotations/coco_b.json','r',encoding='utf8')as fp:
    json_data1 = json.load(fp)
print("原标注数:", len(json_data1['annotations']))
print("原图片数", len(json_data1['images']))

with open('./datasets/index1/step2/voc+coco_a/annotations/voc+coco_a.json','r',encoding='utf8')as fp:
    json_data2 = json.load(fp)
print("原标注数:", len(json_data2['annotations']))
print("原图片数", len(json_data2['images']))

json_data_new1 = dict()
json_data_new1['images'] = json_data1['images'] + json_data2['images']
json_data_new1['annotations'] = json_data1['annotations'] + json_data2['annotations']
json_data_new1['categories'] = json_data2['categories']
print("合并后标注数", len(json_data_new1['annotations']))
print("合并后图片数", len(json_data_new1['images']))

with open('./datasets/index1/step2/coco_b/val/annotations/coco_b_val.json','r',encoding='utf8')as fp:
    json_data3 = json.load(fp)
print("原标注数:", len(json_data3['annotations']))
print("原图片数", len(json_data3['images']))

with open('./datasets/index1/step2/voc+coco_a/test/annotations/voc+coco_a-test.json','r',encoding='utf8')as fp:
    json_data4 = json.load(fp)
print("原标注数:", len(json_data4['annotations']))
print("原图片数", len(json_data4['images']))

json_data_new2 = dict()
json_data_new2['images'] = json_data3['images'] + json_data4['images']
json_data_new2['annotations'] = json_data3['annotations'] + json_data4['annotations']
json_data_new2['categories'] = json_data4['categories']
print("合并后标注数", len(json_data_new2['annotations']))
print("合并后图片数", len(json_data_new2['images']))
json.dump(json_data_new1, open('./datasets/index1/step3/voc+coco_a+coco_b/annotations/voc+coco_a+coco_b.json', 'w'))
json.dump(json_data_new2, open('./datasets/index1/step3/voc+coco_a+coco_b/test/annotations/voc+coco_a+coco_b-test.json', 'w'))

json_file = './datasets/index1/step2/voc+coco_a/annotations/voc+coco_a.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step2/voc+coco_a/images'
ana_img_save_path = './datasets/index1/step3/voc+coco_a+coco_b/images/'

for img in tqdm(data['images']):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    # ana_img_name = head + ".jpg"
    # old_img_path = img_path + '/' + ana_img_name
    # new_img_path = ana_img_save_path + '/' + ana_img_name
    old_img_path = img_path + '/' + filename
    new_img_path = ana_img_save_path + '/' + filename
    shutil.copy(old_img_path, new_img_path)

json_file = './datasets/index1/step2/coco_b/annotations/coco_b.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step2/coco_b/images'
ana_img_save_path = './datasets/index1/step3/voc+coco_a+coco_b/images/'

for img in tqdm(data['images']):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    # ana_img_name = head + ".jpg"
    # old_img_path = img_path + '/' + ana_img_name
    # new_img_path = ana_img_save_path + '/' + ana_img_name
    old_img_path = img_path + '/' + filename
    new_img_path = ana_img_save_path + '/' + filename
    shutil.copy(old_img_path, new_img_path)

json_file = './datasets/index1/step2/voc+coco_a/test/annotations/voc+coco_a-test.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step2/voc+coco_a/test/images'
ana_img_save_path = './datasets/index1/step3/voc+coco_a+coco_b/test/images/'

for img in tqdm(data['images']):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    # ana_img_name = head + ".jpg"
    # old_img_path = img_path + '/' + ana_img_name
    # new_img_path = ana_img_save_path + '/' + ana_img_name
    old_img_path = img_path + '/' + filename
    new_img_path = ana_img_save_path + '/' + filename
    shutil.copy(old_img_path, new_img_path)

json_file = './datasets/index1/step2/coco_b/val/annotations/coco_b_val.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step2/coco_b/val/images'
ana_img_save_path = './datasets/index1/step3/voc+coco_a+coco_b/test/images/'

for img in tqdm(data['images']):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    # ana_img_name = head + ".jpg"
    # old_img_path = img_path + '/' + ana_img_name
    # new_img_path = ana_img_save_path + '/' + ana_img_name
    old_img_path = img_path + '/' + filename
    new_img_path = ana_img_save_path + '/' + filename
    shutil.copy(old_img_path, new_img_path)

