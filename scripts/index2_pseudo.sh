#!/bin/sh

mkdir -p output/pseudo_label/index2
cp -r datasets/two_stage_datasets/index2/productline1/train/annotations/instances_default.json output/pseudo_label/index2/
mv output/pseudo_label/index2/instances_default.json output/pseudo_label/index2/index2_source.json
echo "index2 pseudo label"
#index2 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index2/index2_pseudo.yaml \
	MODEL.WEIGHTS output/index2/baseline/productline1_sup100/model_best.pth \
	OUTPUT_DIR output/index2/baseline/productline1_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index2/productline2/train/annotations/instances_default_3cls.json \
  --json_path_target_pseudo output/index2/baseline/productline1_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index2/index2_target_pseudo.json
echo "end"