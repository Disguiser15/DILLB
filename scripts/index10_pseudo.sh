#!/bin/sh

mkdir -p output/pseudo_label/index10
cp -r datasets/two_stage_datasets/index10/exdark/annotations/exdark_train.json output/pseudo_label/index10/
mv output/pseudo_label/index10/exdark_train.json output/pseudo_label/index10/index10_source.json
echo "index10 pseudo label"
#index10 pseudo label
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index10/index10_pseudo.yaml \
	MODEL.WEIGHTS output/index10/baseline/exdark_sup100/model_best.pth \
	OUTPUT_DIR output/index10/baseline/exdark_sup100/pseudo_label

python tools/process_pseudo_label.py \
  --json_path_target_real datasets/two_stage_datasets/index10/voc_07_12_11/Annotations/instances_trainval.json \
  --json_path_target_pseudo output/index10/baseline/exdark_sup100/pseudo_label/inference/coco_instances_results.json \
  --json_path_target_pseudo_new output/pseudo_label/index10/index10_target_pseudo.json
echo "end"