#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index2
mkdir -p temp/index2
mkdir -p datasets/index2/semi/productline1_20%/annotations
mkdir -p datasets/index2/semi/productline1_20%/images
mkdir -p datasets/index2/semi/productline2_label/annotations
mkdir -p datasets/index2/semi/productline2_label/images
python ./tools/index2/process_productline1_categories.py
python generate_random_supervised_seed_index2.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index2/productline1_sup100_run1.yaml \
	OUTPUT_DIR output/index2/baseline/productline1_sup100
echo "baseline val productline1 and productline2"
#productline1和productline2基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index2/productline1_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index2/baseline/productline1_sup100/model_best.pth \
	OUTPUT_DIR output/index2/baseline/productline1_sup100/record
echo "Data_processing"
#数据处理
python ./tools/index2/pick_productline1_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index2/supervision_to_semi-supervision.py \
	--input-dir output/index2/baseline/productline1_sup100/model_best.pth \
	--save-dir output/index2/baseline/productline1_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index2/productline2train.json \
    --model-weights output/index2/baseline/productline1_sup100/one_stage.pth \
    --config configs/index2/productline1_sup100_run1_inference.yaml
python ./tools/index3/ScoreFunction.py \
	--static-file temp/index2/productline2train.json \
	--indicator-file results/productline2train
echo "Generate productline2_label and productline2_unlabel"
#生成有标签和无标签productline2
python ./tools/index2/make_dataset_target.py
python ./tools/index2/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index2/productline1_semi.yaml \
	OUTPUT_DIR output/index2/semi/productline1_incremental \
	MODEL.WEIGHTS output/index2/baseline/productline1_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index2/productline1_semi_val.yaml \
	MODEL.WEIGHTS output/index2/semi/productline1_incremental/model_best.pth
echo "end"