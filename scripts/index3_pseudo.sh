#!/bin/sh

mkdir -p output/pseudo_label/index3
cp -r datasets/two_stage_datasets/index3/soda_10m/annotations/instance_train.json output/pseudo_label/index3/
mv output/pseudo_label/index3/instance_train.json output/pseudo_label/index3/index3_source.json
echo "index3 pseudo label"
#index3 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index3/index3_pseudo.yaml \
	MODEL.WEIGHTS output/index3/baseline/soda_sup100/model_best.pth \
	OUTPUT_DIR output/index3/baseline/soda_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index3/trafficrecoder/train/annotations/instances_default.json \
  --json_path_target_pseudo output/index3/baseline/soda_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index3/index3_target_pseudo.json
echo "end"