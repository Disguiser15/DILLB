import os
import json
import shutil
from tqdm import tqdm

json_file = './datasets/index6/semi/voc07_20%/annotations/instance_train_20%.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index6/voc_2007/JPEGImages'
ana_img_save_path = './datasets/index6/semi/voc07_20%/images/'

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

json_file = './datasets/index6/semi/voc12_label/annotations/voc12_active_label.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index6/voc_2012/JPEGImages'
ana_img_save_path = './datasets/index6/semi/voc12_label/images/'

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