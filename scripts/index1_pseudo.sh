#!/bin/sh

mkdir -p output/pseudo_label/index1
python tools/index1/make_dataset_voc2007train_voc2012trainval.py
cp -r datasets/index1/step1/voc_07_12/Annotations/instances_train.json output/pseudo_label/index1/
mv output/pseudo_label/index1/instances_train.json output/pseudo_label/index1/index1_source.json
echo "index1 pseudo label"
#index1 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/index1_pseudo.yaml \
	MODEL.WEIGHTS output/index1/baseline/voc07train+voc12trainval_sup100/model_best.pth \
	OUTPUT_DIR output/index1/baseline/voc07train+voc12trainval_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/index1/coco20/annotations/instances_train2017_new.json \
  --json_path_target_pseudo output/index1/baseline/voc07train+voc12trainval_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index1/index1_target_pseudo.json
echo "end"