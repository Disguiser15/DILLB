#!/bin/sh

mkdir -p output/pseudo_label/index5
cp -r datasets/two_stage_datasets/index5/taco_a/annotations/instances_train.json output/pseudo_label/index5/
mv output/pseudo_label/index5/instances_train.json output/pseudo_label/index5/index5_source.json
echo "index5 pseudo label"
#index5 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index5/index5_pseudo.yaml \
	MODEL.WEIGHTS output/index5/baseline/tacoa_sup100/model_best.pth \
	OUTPUT_DIR output/index5/baseline/tacoa_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index5/taco_b/annotations/instances_train.json \
  --json_path_target_pseudo output/index5/baseline/tacoa_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index5/index5_target_pseudo.json
echo "end"