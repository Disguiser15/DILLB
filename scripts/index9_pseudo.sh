#!/bin/sh

mkdir -p output/pseudo_label/index9
cp -r datasets/two_stage_datasets/index9/exdark/annotations/exdark_train.json output/pseudo_label/index9/
mv output/pseudo_label/index9/exdark_train.json output/pseudo_label/index9/index9_source.json
echo "index9 pseudo label"
#index9 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index9/index9_pseudo.yaml \
	MODEL.WEIGHTS output/index9/baseline/exdark_sup100/model_best.pth \
	OUTPUT_DIR output/index9/baseline/exdark_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index9/coco12/annotations/instances_train2017.json \
  --json_path_target_pseudo output/index9/baseline/exdark_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index9/index9_target_pseudo.json
echo "end"