#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index8
python generate_random_supervised_seed_index8_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index8/voc5_sup100_run1_full.yaml \
	OUTPUT_DIR output/index8/baseline/voc5+object365_sup100_full
echo "baseline val voc5 and object365"
#voc5和object365全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index8/voc5_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index8/baseline/voc5+object365_sup100_full/model_best.pth
echo "end"