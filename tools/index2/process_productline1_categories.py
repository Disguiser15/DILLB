## 加载json文件
import json
import random
import math
import numpy as np
import re
from tqdm import tqdm

with open('./datasets/two_stage_datasets/index2/productline1/train/annotations/instances_default.json','r',encoding='utf8')as fp:
    json_data1 = json.load(fp)

with open('./datasets/two_stage_datasets/index2/productline1/test/annotations/instances_default.json','r',encoding='utf8')as fp:
    json_data2 = json.load(fp)

for i in range(len(json_data1['categories'])-1, -1, -1):
    if json_data1['categories'][i]['name'] == 'screw_4' or json_data1['categories'][i]['name'] == 'screw_5' or json_data1['categories'][i]['name'] == 'mem':
        del json_data1['categories'][i]
print(json_data1['categories'])

for i in range(len(json_data2['categories'])-1, -1, -1):
    if json_data2['categories'][i]['name'] == 'screw_4' or json_data2['categories'][i]['name'] == 'screw_5' or json_data2['categories'][i]['name'] == 'mem':
        del json_data2['categories'][i]
print(json_data2['categories'])

json.dump(json_data1, open('./datasets/two_stage_datasets/index2/productline1/train/annotations/instances_default.json', 'w'))
json.dump(json_data2, open('./datasets/two_stage_datasets/index2/productline1/test/annotations/instances_default.json', 'w'))
