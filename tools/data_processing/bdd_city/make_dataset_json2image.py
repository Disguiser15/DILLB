import os
import json
import shutil
from tqdm import tqdm

def json2image():
    if not os.path.exists(args.ana_img_save_path_source):
        os.makedirs(args.ana_img_save_path_source)
    if not os.path.exists(args.ana_img_save_path_target):
        os.makedirs(args.ana_img_save_path_target)
    json_file = args.json_file_source
    data = json.load(open(json_file, 'r'))
    img_path = args.img_path_source
    ana_img_save_path = args.ana_img_save_path_source

    for img in tqdm(data['images']):
        filename = img["file_name"]
        old_img_path = img_path + '/' + filename
        new_img_path = ana_img_save_path + '/' + filename
        shutil.copy(old_img_path, new_img_path)

    json_file = args.json_file_target
    data = json.load(open(json_file, 'r'))
    img_path = args.img_path_target
    ana_img_save_path = args.ana_img_save_path_target

    for img in tqdm(data['images']):
        filename = img["file_name"]
        old_img_path = img_path + '/' + filename
        new_img_path = ana_img_save_path + '/' + filename
        shutil.copy(old_img_path, new_img_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='pick the corresponding images from json file')
    parser.add_argument("--json-file-source",type=str,default='./datasets/index11/semi/bdd1%/annotations/bdd_1%.json')
    parser.add_argument("--img-path-source",type=str,default='./datasets/bdd100k/images/train')
    parser.add_argument("--ana-img-save-path-source",type=str,default='./datasets/index11/semi/bdd1%/images')
    parser.add_argument("--json-file-target",type=str,default='./datasets/index11/semi/cityscapes10%/annotations/cityscapes_10%.json')
    parser.add_argument("--img-path-target",type=str,default='./datasets/Cityscapes_new/JPEGImages/train_s')
    parser.add_argument("--ana-img-save-path-target",type=str,default='./datasets/index11/semi/cityscapes10%/images')
    args = parser.parse_args()
    json2image()