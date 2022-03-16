#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index6
python generate_random_supervised_seed_index6_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index6/voc07trainval_sup100_run1_full.yaml \
	OUTPUT_DIR output/index6/baseline/voc07+voc12_full
echo "baseline val voc07 and voc12"
#voc07和voc12全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index6/voc07trainval_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index6/baseline/voc07+voc12_full/model_best.pth
echo "end"