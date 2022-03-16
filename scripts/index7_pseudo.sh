#!/bin/sh

mkdir -p output/pseudo_label/index7
cp -r datasets/two_stage_datasets/index7/voc07_12_5/Annotations/instances_trainval.json output/pseudo_label/index7/
mv output/pseudo_label/index7/instances_trainval.json output/pseudo_label/index7/index7_source.json
echo "index7 pseudo label"
#index7 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index7/index7_pseudo.yaml \
	MODEL.WEIGHTS output/index7/baseline/voc5_sup100/model_best.pth \
	OUTPUT_DIR output/index7/baseline/voc5_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index7/coco5/annotations/instances_train2017_new.json \
  --json_path_target_pseudo output/index7/baseline/voc5_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index7/index7_target_pseudo.json
echo "end"