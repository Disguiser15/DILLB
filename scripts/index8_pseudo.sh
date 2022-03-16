#!/bin/sh

mkdir -p output/pseudo_label/index8
cp -r datasets/two_stage_datasets/index8/voc_07_12_2/Annotations/instances_trainval.json output/pseudo_label/index8/
mv output/pseudo_label/index8/instances_trainval.json output/pseudo_label/index8/index8_source.json
echo "index8 pseudo label"
#index8 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index8/index8_psesudo.yaml \
	MODEL.WEIGHTS output/index8/baseline/voc5_sup100/model_best.pth \
	OUTPUT_DIR output/index8/baseline/voc5_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index8/objects365_5/Annotations/instances_train.json \
  --json_path_target_pseudo output/index8/baseline/voc5_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index8/index8_target_pseudo.json
echo "end"