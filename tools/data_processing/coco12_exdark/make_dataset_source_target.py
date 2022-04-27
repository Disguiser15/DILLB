## 加载json文件
import json
import numpy as np
from tqdm import tqdm

def add_source_target_label():
    # 读取json文件内容,返回字典格式
    with open(args.input_dir_source,'r',encoding='utf8')as fp:
        json_data = json.load(fp)

    for i in tqdm(range(len(json_data['images']))):
        json_data['images'][i]['is_source'] = 1
    print(json_data['images'][0])
    json.dump(json_data, open(args.save_dir_source, 'w'))

    # 读取json文件内容,返回字典格式
    with open(args.input_dir_target,'r',encoding='utf8')as fp:
        json_data = json.load(fp)

    for i in tqdm(range(len(json_data['images']))):
        json_data['images'][i]['is_source'] = 0
    print(json_data['images'][0])
    json.dump(json_data, open(args.save_dir_target, 'w'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='add source and target label')
    parser.add_argument("--input-dir-source",type=str,default='./datasets/index9/semi/coco1%/annotations/coco_1%.json')
    parser.add_argument("--save-dir-source",type=str,default='./datasets/index9/semi/coco1%/annotations/coco_1%_source.json')
    parser.add_argument("--input-dir-target",type=str,default='./datasets/index9/semi/exdark5%/annotations/exdark_5%.json')
    parser.add_argument("--save-dir-target",type=str,default='./datasets/index9/semi/exdark5%/annotations/exdark_5%_target.json')
    args = parser.parse_args()
    add_source_target_label()