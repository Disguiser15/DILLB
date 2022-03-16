import os
import json
import shutil
from tqdm import tqdm

json_file = './datasets/index8/semi/voc5_20%/annotations/instances_trainval_20%.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index8/voc_07_12_2/JPEGImages/trainval'
ana_img_save_path = './datasets/index8/semi/voc5_20%/images/'

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

json_file = './datasets/index8/semi/objects365_label/annotations/object365_active_label.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index8/objects365_5/Images/train'
ana_img_save_path = './datasets/index8/semi/objects365_label/images/'

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

json_file = './datasets/index8/semi/objects365_unlabel/annotations/object365_active_unlabel.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index8/objects365_5/Images/train'
ana_img_save_path = './datasets/index8/semi/objects365_unlabel/images/'

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