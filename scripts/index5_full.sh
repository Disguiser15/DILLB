#!/bin/sh

#echo "Data_processing"
##数据处理
mkdir -p dataseed/index5
python generate_random_supervised_seed_index5_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index5/taco_a_sup100_run1_full.yaml \
	OUTPUT_DIR output/index5/baseline/tacoa+tacob_sup100_full
echo "baseline val tacoa and tacob"
#tacoa和tacob全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index5/taco_a_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index5/baseline/tacoa+tacob_sup100_full/model_best.pth
echo "end"