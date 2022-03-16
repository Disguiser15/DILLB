import os
import json
import shutil
from tqdm import tqdm

json_file = './datasets/index1/step2/semi/voc+cocoa_20%/annotations/voc+coco_a_20%.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step2/voc+coco_a/images'
ana_img_save_path = './datasets/index1/step2/semi/voc+cocoa_20%/images/'

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

json_file = './datasets/index1/step2/semi/cocob_label/annotations/cocob_active_label.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step2/coco_b/images'
ana_img_save_path = './datasets/index1/step2/semi/cocob_label/images/'

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

json_file = './datasets/index1/step2/semi/cocob_unlabel/annotations/cocob_active_unlabel.json'
data = json.load(open(json_file, 'r'))
img_path = './datasets/index1/step2/coco_b/images'
ana_img_save_path = './datasets/index1/step2/semi/cocob_unlabel/images/'

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