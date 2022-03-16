#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index7
python generate_random_supervised_seed_index7_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index7/voc5_sup100_run1_full.yaml \
	OUTPUT_DIR output/index7/baseline/voc5+coco5_sup100_full
echo "baseline val voc5 and coco5"
#voc5和coco5全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index7/voc5_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index7/baseline/voc5+coco5_sup100_full/model_best.pth
echo "end"