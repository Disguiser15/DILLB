#!/bin/sh
echo "Data_processing"
echo "Data_processing coco1%"
python tools/data_processing/coco12_exdark/make_dataset_coco1%.py
echo "Data_processing exdark5%"
python tools/data_processing/coco12_exdark/make_dataset_exdark5%.py
echo "Copy the corresponding images"
python tools/data_processing/coco12_exdark/make_dataset_json2image.py
echo "Distinguish source and target"
python tools/data_processing/coco12_exdark/make_dataset_source_target.py

python ./tools/data_processing/coco12_exdark/make_dataset_source_target.py \
	--input-dir-source datasets/two_stage_datasets/index9/coco12/annotations/instances_train2017.json \
	--save-dir-source datasets/two_stage_datasets/index9/coco12/annotations/instances_val2017_source.json \
  --input-dir-target datasets/two_stage_datasets/index9/exdark/annotations/exdark_val.json \
  --save-dir-target datasets/two_stage_datasets/index9/exdark/annotations/exdark_val_target.json
echo "end"