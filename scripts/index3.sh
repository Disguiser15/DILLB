#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index3
mkdir -p temp/index3
mkdir -p datasets/index3/semi/soda_20%/annotations
mkdir -p datasets/index3/semi/soda_20%/images
mkdir -p datasets/index3/semi/trafficrecorder_label/annotations
mkdir -p datasets/index3/semi/trafficrecorder_label/images
python generate_random_supervised_seed_index3.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index3/soda10M_sup100_run1.yaml \
	OUTPUT_DIR output/index3/baseline/soda_sup100
echo "baseline val soda and trafficrecorder"
#soda和trafficrecorder基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index3/soda10M_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index3/baseline/soda_sup100/model_best.pth \
	OUTPUT_DIR output/index3/baseline/soda_sup100/record
echo "Data_processing"
#数据处理
python ./tools/index3/pick_soda_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index3/supervision_to_semi-supervision.py \
	--input-dir output/index3/baseline/soda_sup100/model_best.pth \
	--save-dir output/index3/baseline/soda_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index3/sodatrain.json \
    --model-weights output/index3/baseline/soda_sup100/one_stage.pth \
    --config configs/index3/soda10M_sup100_run1_inference.yaml
python ./tools/index3/ScoreFunction.py \
	--static-file temp/index3/sodatrain.json \
	--indicator-file results/sodatrain
echo "Generate soda_label and soda_unlabel"
#生成有标签和无标签soda
python ./tools/index3/make_dataset_target.py
python ./tools/index3/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index3/soda10M_semi.yaml \
	OUTPUT_DIR output/index3/semi/soda_incremental \
	MODEL.WEIGHTS output/index3/baseline/soda_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index3/soda10M_semi_val.yaml \
	MODEL.WEIGHTS output/index3/semi/soda_incremental/model_best.pth
echo "end"