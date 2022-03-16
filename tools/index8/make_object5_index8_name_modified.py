import json
import numpy as np
# 读取json文件内容,返回字典格式
with open('./datasets/two_stage_datasets/index8/objects365_5/Annotations/instances_train.json','r',encoding='utf8')as fp:
    json_data1 = json.load(fp)
# print(json_data['categories'])
for i in json_data1['categories']:
    if i['name']=='Cat':
        i['name']='cat'
    if i['name']=='Cow':
        i['name']='cow'
    if i['name']=='Dog':
        i['name']='dog'
    if i['name']=='Horse':
        i['name']='horse'
    if i['name']=='Sheep':
        i['name']='sheep'
# print(json_data['categories'])
json.dump(json_data1, open('./datasets/two_stage_datasets/index8/objects365_5/Annotations/instances_train.json', 'w'))

# 读取json文件内容,返回字典格式
with open('./datasets/two_stage_datasets/index8/objects365_5/Annotations/instances_val.json','r',encoding='utf8')as fp:
    json_data1 = json.load(fp)
# print(json_data['categories'])
for i in json_data1['categories']:
    if i['name']=='Cat':
        i['name']='cat'
    if i['name']=='Cow':
        i['name']='cow'
    if i['name']=='Dog':
        i['name']='dog'
    if i['name']=='Horse':
        i['name']='horse'
    if i['name']=='Sheep':
        i['name']='sheep'
# print(json_data['categories'])
json.dump(json_data1, open('./datasets/two_stage_datasets/index8/objects365_5/Annotations/instances_val.json', 'w'))