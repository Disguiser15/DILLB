import os
import json
import shutil
from tqdm import tqdm

json_file = './datasets/index1/step3/semi/voc+cocoa+cocob_20%/annotations/voc+coco_a+coco_b_20%.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step3/voc+coco_a+coco_b/images'
ana_img_save_path = './datasets/index1/step3/semi/voc+cocoa+cocob_20%/images/'

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

json_file = './datasets/index1/step3/semi/cococ_label/annotations/cococ_active_label.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step3/coco_c/images'
ana_img_save_path = './datasets/index1/step3/semi/cococ_label/images/'

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

json_file = './datasets/index1/step3/semi/cococ_unlabel/annotations/cococ_active_unlabel.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step3/coco_c/images'
ana_img_save_path = './datasets/index1/step3/semi/cococ_unlabel/images/'

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