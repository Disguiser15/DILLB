#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index9
python generate_random_supervised_seed_index9_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index9/exdark_sup100_run1_full.yaml \
	OUTPUT_DIR output/index9/baseline/exdark+coco12_sup100_full
echo "baseline val exdark and coco12"
#exdark和coco12全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index9/exdark_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index9/baseline/exdark+coco12_sup100_full/model_best.pth
echo "end"