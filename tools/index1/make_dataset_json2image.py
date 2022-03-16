import os
import json
import shutil
from tqdm import tqdm

json_file = './datasets/index1/semi/coco20_label/annotations/coco20_active_label.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/coco20/images/train2017'
ana_img_save_path = './datasets/index1/semi/coco20_label/images/'

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

json_file = './datasets/index1/semi/coco20_unlabel/annotations/coco20_unlabel.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/coco20/images/train2017'
ana_img_save_path = './datasets/index1/semi/coco20_unlabel/images/'

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