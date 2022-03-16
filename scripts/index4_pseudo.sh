#!/bin/sh

mkdir -p output/pseudo_label/index4
cp -r datasets/two_stage_datasets/index4/coco20_a/annotations/instances_train.json output/pseudo_label/index4/
mv output/pseudo_label/index4/instances_train.json output/pseudo_label/index4/index4_source.json
echo "index4 pseudo label"
#index4 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index4/index4_pseudo.yaml \
	MODEL.WEIGHTS output/index4/baseline/cocoa_sup100/model_best.pth \
	OUTPUT_DIR output/index4/baseline/cocoa_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index4/coco20_b/annotations/instances_train.json \
  --json_path_target_pseudo output/index4/baseline/cocoa_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index4/index4_target_pseudo.json
echo "end"