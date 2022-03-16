import json
import argparse
import os

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET


'''
目录格式如下：
$COCO_PATH
----|annotations
----|train2017
----|val2017
----|test2017
'''
 
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
 
tailstr = '''\
</annotation>
'''
classes = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
    8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter',
    15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
 
def generate_xml():
    # 需要设置的路径
    dataDir = args.input_dir 
    savepath = args.save_dir 
    img_dir = savepath+'images/'
    anno_dir = savepath+'annotations/'
    datasets_list = ['train2017', 'val2017']

    #这里写要提取类的名字
    classes_names = [i for i in args.classes_names.split(',')]
    #包含所有类别的原coco数据集路径
    for dataset in datasets_list:
        #./COCO/annotations/instances_train2017.json
        annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)
    
        #使用COCO API用来初始化注释数据
        coco = COCO(annFile)
    
        #获取COCO数据集中的所有类别
        classes = id2name(coco)
        print(classes)

        classes_ids = coco.getCatIds(catNms=classes_names)
        print(classes_ids)
        for cls in classes_names:
            #获取该类的id
            cls_id=coco.getCatIds(catNms=[cls])
            img_ids=coco.getImgIds(catIds=cls_id)
            print(cls,len(img_ids))
            # imgIds=img_ids[0:10]
            for imgId in tqdm(img_ids):
                img = coco.loadImgs(imgId)[0]
                filename = img['file_name']
                # print(filename)
                objs=showimg(dataDir ,coco, dataset, img, classes, classes_names, classes_ids,show=False)
                print(objs)
                save_annotations_and_imgs(dataDir,anno_dir, img_dir, coco, dataset, filename, objs)

 
# 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
def mkr(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 可以创建多级目录

def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes
 
def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)
 
 
def save_annotations_and_imgs(dataDir,anno_dir,img_dir,coco,dataset,filename,objs):
    #将图片转为xml，例:COCO_train2017_000000196610.jpg-->COCO_train2017_000000196610.xml
    dst_anno_dir = os.path.join(anno_dir, dataset)
    mkr(dst_anno_dir)
    anno_path=dst_anno_dir + '/' + filename[:-3]+'xml'
    img_path=dataDir+dataset+'/'+filename
    print("img_path: ", img_path)
    dst_img_dir = os.path.join(img_dir, dataset)
    mkr(dst_img_dir)
    dst_imgpath=dst_img_dir+ '/' + filename
    print("dst_imgpath: ", dst_imgpath)
    img=cv2.imread(img_path)
    #if (img.shape[2] == 1):
    #    print(filename + " not a RGB image")
     #   return
    shutil.copy(img_path, dst_imgpath)
 
    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)
 
 
def showimg(dataDir,coco,dataset,img,classes,classes_names,cls_id,show=True):
    I=Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
    #通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        if int(ann['iscrowd']) !=0:
            continue
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()
 
    return objs


coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
category_set = dict()
category_name2id = dict()
image_set = set()

image_id = 20180000000
annotation_id = 0
def xml2json():
    xml_dir = args.save_dir+'/annotations'
    json_dir = args.save_dir+'/annotations'
    dataset_lists = ['train2017', 'val2017']
    # dataset_lists = ['val2017']

    for i,name in enumerate(args.classes_names.split(',')):
        category_name2id[name] = i+1

    for dataset in dataset_lists:
        global coco
        global category_set
        global image_set
        coco = dict()
        coco['images'] = []
        coco['type'] = 'instances'
        coco['annotations'] = []
        coco['categories'] = []
        category_set = dict()
        image_set = set()

        xml_path = os.path.join(xml_dir, dataset)
        json_file = json_dir + '/instances_{}.json'.format(dataset)
        parseXmlFiles(xml_path)
        json.dump(coco, open(json_file, 'w'))


def addCatItem(name):
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item['id'] = category_name2id[name]
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_name2id[name]
    return category_item['id']

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        size = dict()
        current_image_id = None
        current_category_id = None

        xml_file = os.path.join(xml_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))\
        
        for subelem in root.find('size'):
            size[subelem.tag] = int(subelem.text)

        file_name = root.find('filename').text
        current_image_id = addImgItem(file_name, size)
        print('add image with {} and {}'.format(file_name, size))

        for obj in root.findall('object'):
            bndbox = dict()
            object_name = obj.find('name').text
            if object_name not in category_set:
                current_category_id = addCatItem(object_name)
            else:
                current_category_id = category_set[object_name]
            for x in obj.find('bndbox'):
                bndbox[x.tag] = int(x.text)
            bbox = []
            #x
            bbox.append(bndbox['xmin'])
            #y
            bbox.append(bndbox['ymin'])
            #w
            bbox.append(bndbox['xmax'] - bndbox['xmin'])
            #h
            bbox.append(bndbox['ymax'] - bndbox['ymin'])
            print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
            addAnnoItem(object_name, current_image_id, current_category_id, bbox )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='make coco20 dataset')
    parser.add_argument("--input-dir",type=str,default='datasets/COCO2017/')
    parser.add_argument("--save-dir",type=str,default='datasets/index1/coco20/')
    parser.add_argument("--classes_names",type=str,default='airplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,dining table,dog,horse,motorcycle,person,potted plant,sheep,couch,train,tv')
    args = parser.parse_args()
    generate_xml()
    xml2json()


        


