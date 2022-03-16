def coco_name_modified_train():
    # 读取json文件内容,返回字典格式
    with open(args.input_annotation1, 'r',encoding='utf8')as fp:
        json_data = json.load(fp)

    for i in json_data["categories"]:
        if i["name"]=="motorcycle":
            i["name"]="motorbike"
        if i["name"]=="airplane":
            i["name"]="aeroplane"
        if i["name"]=="couch":
            i["name"]="sofa"
        if i["name"]=="potted plant":
            i["name"]="pottedplant"
        if i["name"]=="dining table":
            i["name"]="diningtable"
        if i["name"]=="tv":
            i["name"]="tvmonitor"
    # print(json_data["categories"])
    json.dump(json_data, open(args.output_dir1, 'w'))

def coco_name_modified_val():
    # 读取json文件内容,返回字典格式
    with open(args.input_annotation2, 'r',encoding='utf8')as fp:
        json_data = json.load(fp)

    for i in json_data["categories"]:
        if i["name"]=="motorcycle":
            i["name"]="motorbike"
        if i["name"]=="airplane":
            i["name"]="aeroplane"
        if i["name"]=="couch":
            i["name"]="sofa"
        if i["name"]=="potted plant":
            i["name"]="pottedplant"
        if i["name"]=="dining table":
            i["name"]="diningtable"
        if i["name"]=="tv":
            i["name"]="tvmonitor"
    # print(json_data["categories"])
    json.dump(json_data, open(args.output_dir2, 'w'))

if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    import json
    import numpy as np
    import random
    import re
    parser = argparse.ArgumentParser(description='modify coco20 train and val dataset')
    parser.add_argument("--input-annotation1", type=str, default='datasets/index1/coco20/annotations/instances_train2017.json')
    parser.add_argument("--output-dir1", type=str, default='datasets/index1/coco20/annotations/instances_train2017_new.json')
    parser.add_argument("--input-annotation2", type=str, default='datasets/index1/coco20/annotations/instances_val2017.json')
    parser.add_argument("--output-dir2", type=str, default='datasets/index1/coco20/annotations/instances_val2017_new.json')
    args = parser.parse_args()
    coco_name_modified_train()
    coco_name_modified_val()