#!/bin/sh
echo "Data_processing"
echo "Data_processing bdd1%"
python tools/data_processing/bdd_city/make_dataset_bdd1%.py
echo "Data_processing city10%"
python tools/data_processing/bdd_city/make_dataset_city10%.py
echo "Copy the corresponding images"
python tools/data_processing/bdd_city/make_dataset_json2image.py
echo "Distinguish source and target"
python tools/data_processing/bdd_city/make_dataset_source_target.py

python ./tools/data_processing/bdd_city/make_dataset_source_target.py \
	--input-dir-source datasets/bdd100k/annotations/bdd100k_labels_images_det_coco_val.json \
	--save-dir-source datasets/bdd100k/annotations/bdd100k_labels_images_det_coco_val_source.json \
  --input-dir-target datasets/Cityscapes_new/Annotations/instances_test_s.json \
  --save-dir-target datasets/Cityscapes_new/Annotations/instances_test_s_target.json
echo "end"