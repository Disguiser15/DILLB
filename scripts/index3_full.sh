#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index3
python generate_random_supervised_seed_index3_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index3/soda10M_sup100_run1_full.yaml \
	OUTPUT_DIR output/index3/baseline/soda+trafficrecorder_sup100_full
echo "baseline val soda and trafficrecorder"
#soda和trafficrecorder全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index3/soda10M_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index3/baseline/soda+trafficrecorder_sup100_full/model_best.pth
echo "end"