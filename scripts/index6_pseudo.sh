#!/bin/sh

mkdir -p output/pseudo_label/index6
cp -r datasets/two_stage_datasets/index6/voc_2007/Annotations/instances_train.json output/pseudo_label/index6/
mv output/pseudo_label/index6/instances_train.json output/pseudo_label/index6/index6_source.json
echo "index6 pseudo label"
#index6 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index6/index6_pseudo.yaml \
	MODEL.WEIGHTS output/index6/baseline/voc07_sup100/model_best.pth \
	OUTPUT_DIR output/index6/baseline/voc07_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index6/voc_2012/Annotations/instances_train.json \
  --json_path_target_pseudo output/index6/baseline/voc07_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index6/index6_target_pseudo.json
echo "end"