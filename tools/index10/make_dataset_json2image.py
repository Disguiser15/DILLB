import os
import json
import shutil
from tqdm import tqdm

json_file = './datasets/index10/semi/exdark_20%/annotations/exdark_train_20%.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index10/exdark/images/train'
ana_img_save_path = './datasets/index10/semi/exdark_20%/images/'

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

json_file = './datasets/index10/semi/voc11_label/annotations/voc11_active_label.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index10/voc_07_12_11/JPEGImages/trainval'
ana_img_save_path = './datasets/index10/semi/voc11_label/images/'

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

json_file = './datasets/index10/semi/voc11_unlabel/annotations/voc11_active_unlabel.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/two_stage_datasets/index10/voc_07_12_11/JPEGImages/trainval'
ana_img_save_path = './datasets/index10/semi/voc11_unlabel/images/'

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