#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index2
python ./tools/index2/process_productline1_categories.py
python generate_random_supervised_seed_index2_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index2/productline1_sup100_run1_full.yaml \
	OUTPUT_DIR output/index2/baseline/productline1+productline2_sup100_full
echo "baseline val productline1 and productline2"
#productline1和productline2全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index2/productline1_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index2/baseline/productline1+productline2_sup100_full/model_best.pth
echo "end"