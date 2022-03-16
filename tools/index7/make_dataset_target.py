# pick-active-semi
def generate_active_label_data():
    with open(args.input_annotation, 'r') as f:
        data = json.load(f)
        print("提取完成")

    dic_cls2imgidlist = {}
    dic_cls2imgidset = {}
    for k in tqdm(range(len(data['annotations']))):
        if data['annotations'][k]['category_id'] not in dic_cls2imgidlist.keys():
            dic_cls2imgidlist[data['annotations'][k]['category_id']] = list()
            dic_cls2imgidlist[data['annotations'][k]['category_id']].append(data['annotations'][k]['image_id'])
        else:
            dic_cls2imgidlist[data['annotations'][k]['category_id']].append(data['annotations'][k]['image_id'])
    for i in dic_cls2imgidlist.keys():
        dic_cls2imgidset[i] = set(dic_cls2imgidlist[i])
    clsidlist = sorted(list(dic_cls2imgidset.keys()))

    dic_id2annlist = {}
    for k in tqdm(range(len(data['annotations']))):
        if data['annotations'][k]['image_id'] not in dic_id2annlist.keys():
            dic_id2annlist[data['annotations'][k]['image_id']] = list()
            dic_id2annlist[data['annotations'][k]['image_id']].append(data['annotations'][k])
        else:
            dic_id2annlist[data['annotations'][k]['image_id']].append(data['annotations'][k])
    dic_id2classset = {}
    for key,value in tqdm(dic_id2annlist.items()):
        classidList = list()
        for i in value:
            clsId = clsidlist.index(i['category_id'])
            if clsId not in classidList:
                classidList.append(clsId)
        dic_id2classset[key] = sorted(classidList)

    dic_filename2imgidset = {}
    for k in tqdm(range(len(data['images']))):
        if data['images'][k]['file_name'] not in dic_filename2imgidset.keys():
            dic_filename2imgidset[data['images'][k]['file_name']] = data['images'][k]['id']

    ## 从评分json生成排好序的list
    # 读取json文件内容,返回字典格式
    with open(args.active_annotation, 'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    json_data_sorted = sorted(json_data.items(), key=lambda item: item[1], reverse=True)
    
    s1 = args.remove_string
    json_data_sorted_id = []
    for i in range(len(json_data_sorted)):
        string1 = json_data_sorted[i][0].replace(s1, '')
        json_data_sorted_id.append(string1)
    
    imgfilenamelist = json_data_sorted_id
    imgidlist = list()
    for i in tqdm(range(len(imgfilenamelist))):
        imgidlist.append(dic_filename2imgidset[imgfilenamelist[i]])

    newimgidlist = list()
    for i in tqdm(range(min(args.classes_number * 99, len(imgidlist)))):
        newimgidlist.append(int(imgidlist[i]))

    newdata = dict()
    newdata['categories'] = data['categories']
    # newdata['licenses'] = data['licenses']
    imgpool = list()
    for i in tqdm(range(len(data['images']))):
        if data['images'][i]['id'] in newimgidlist:
            imgpool.append(data['images'][i])
    newdata['images'] = imgpool
    anno_base5 = list()
    for i in tqdm(range(len(data['annotations']))):
        if data['annotations'][i]['image_id'] in newimgidlist:
            anno_base5.append(data['annotations'][i])
    newdata['annotations'] = anno_base5
    print('主动样本挑选有标签图片数：', len(newdata['images']))
    print('主动样本挑选有标签标注数：', len(newdata['annotations']))
    print("end-------------------------")
    with open(args.output_dir1, 'w') as f_obj3:
        json.dump(newdata, f_obj3)

def generate_unlabel_data():
    # 生成无标签 随机挑选
    with open(args.input_annotation, 'r') as f:
        data = json.load(f)
        print("提取完成")
    with open(args.output_dir1, 'r') as f:
        data2 = json.load(f)
        print("提取完成")

    print("start-------------------------")
    imgidlist2 = list()
    for i in tqdm(range(len(data2['images']))):
        imgidlist2.append(data2['images'][i]['id'])
    imgidlist2 = set(imgidlist2)
    imgidlist = list()
    num = 0
    for i in tqdm(range(len(data['images']))):
        imgid = data['images'][i]['id']
        if imgid not in imgidlist2:
            imgidlist.append(imgid)
        else:
            num = num + 1
    random.seed(args.data_seed)
    temp = len(imgidlist)
    temp2 = args.classes_number * 990
    samplenum = min(temp, temp2)
    imglistnew = random.sample(imgidlist, samplenum)
    newdata = dict()
    newdata['categories'] = data['categories']

    imglistnew = set(imglistnew)
    imgpool = list()
    for i in tqdm(range(len(data['images']))):
        if data['images'][i]['id'] in imglistnew:
            imgpool.append(data['images'][i])
    newdata['images'] = imgpool

    anno_base5 = list()
    for i in tqdm(range(len(data['annotations']))):
        if data['annotations'][i]['image_id'] in imglistnew:
            anno_base5.append(data['annotations'][i])
    newdata['annotations'] = anno_base5
    print('无标签图片数：', len(newdata['images']))
    print('无标签标注数：', len(newdata['annotations']))
    print("end-------------------------")
    with open(args.output_dir2, 'w') as f_obj3:
        json.dump(newdata, f_obj3)

if __name__=='__main__':
    import argparse
    from tqdm import tqdm
    import json
    import numpy as np
    import random
    import re
    parser = argparse.ArgumentParser(description='make target dataset')
    parser.add_argument("--input-annotation", type=str, default='datasets/two_stage_datasets/index7/coco5/annotations/instances_train2017_new.json')
    parser.add_argument("--active-annotation", type=str, default='results/coco5_index7.json')
    parser.add_argument("--output-dir1", type=str, default='datasets/index7/semi/coco5_label/annotations/coco5_active_label.json')
    parser.add_argument("--output-dir2", type=str, default='datasets/index7/semi/coco5_unlabel/annotations/coco5_active_unlabel.json')
    parser.add_argument("--remove-string", type=str, default='datasets/two_stage_datasets/index7/coco5/images/train2017/')
    parser.add_argument("--classes-number", type=str, default=5)
    parser.add_argument("--data-seed", type=str, default=1)
    args = parser.parse_args()
    generate_active_label_data()
    generate_unlabel_data()