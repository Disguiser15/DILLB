#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index10
python generate_random_supervised_seed_index10_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index10/exdark_sup100_run1_full.yaml \
	OUTPUT_DIR output/index10/baseline/exdark+voc11_sup100_full
echo "baseline val exdark and voc11"
#exdark和voc11全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index10/exdark_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index10/baseline/exdark+voc11_sup100_full/model_best.pth
echo "end"