import json
import argparse
import os
import math
import random

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from xml.dom import minidom

def generate_xml():
    # 需要设置的路径
    dataDir1 = args.input_dir1
    savepath = args.save_dir 
    img_dir = savepath+'JPEGImages/'
    anno_dir = savepath+'Annotations/'
    mkr(img_dir)
    mkr(anno_dir)
    with open(os.path.join(dataDir1,'ImageSets/Main/test.txt')) as f:
        train_07trainval_original = set()
        for line in f.readlines():
            train_07trainval_original.add(line.strip())

    train_07trainval = train_07trainval_original

    for xml_file in os.listdir(os.path.join(dataDir1,'Annotations/')):
        if not xml_file.endswith('.xml'):
            continue

        annFile = os.path.join(dataDir1,'Annotations/')+xml_file
        print('--------parsing: '+annFile)
        tree = ET.parse(annFile)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))


        if root.find('object') is not None:
            if xml_file.split('.')[0] in train_07trainval:
                print('save {} to {}'.format(annFile,os.path.join(anno_dir,xml_file)))
                shutil.copy(os.path.join(dataDir1,'Annotations/'+xml_file.split('.')[0]+'.xml'), os.path.join(anno_dir,xml_file.split('.')[0]+'.xml'))
                shutil.copy(os.path.join(dataDir1,'JPEGImages/'+xml_file.split('.')[0]+'.jpg'), os.path.join(img_dir,xml_file.split('.')[0]+'.jpg'))
                        
        print('--------end parsed: '+annFile)


def saveXML(root, filename,indent="", newl="", encoding="utf-8"):
    rawText = ET.tostring(root)
    dom = minidom.parseString(rawText)
    with open(filename, 'w') as f:
        dom.writexml(f, "", indent, newl, encoding)   

 
# 检查目录是否存在，如果存在，先删除再创建，否则，直接创建
def mkr(path):
    if not os.path.exists(path):
        os.makedirs(path)  # 可以创建多级目录

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
category_set = dict()
category_name2id = dict()
image_set = set()

image_id = 20200000000
annotation_id = 30000
def xml2json():
    xml_dir = args.save_dir+'/Annotations'
    json_dir = args.save_dir+'/Annotations'
    
    for i,name in enumerate(args.classes_names.split(',')):
        category_name2id[name] = i+1

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

    xml_path = os.path.join(xml_dir)
    json_file = json_dir + '/instances_{}.json'.format('train')
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
    parser = argparse.ArgumentParser(description='make voc_2007test dataset')
    parser.add_argument("--input-dir1",type=str,default='datasets/VOC2007/')
    parser.add_argument("--save-dir",type=str,default='datasets/index1/step1/voc_07_12/test/')
    parser.add_argument("--classes_names",type=str,default='aeroplane,bicycle,bird,boat,bottle,bus,car,cat,chair,cow,diningtable,dog,horse,motorbike,person,pottedplant,sheep,sofa,train,tvmonitor')
    args = parser.parse_args()
    generate_xml()
    xml2json()


        


