#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index6
mkdir -p temp/index6
mkdir -p datasets/index6/semi/voc07_20%/annotations
mkdir -p datasets/index6/semi/voc07_20%/images
mkdir -p datasets/index6/semi/voc12_label/annotations
mkdir -p datasets/index6/semi/voc12_label/images
python generate_random_supervised_seed_index6.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index6/voc07trainval_sup100_run1.yaml \
	OUTPUT_DIR output/index6/baseline/voc07_sup100
echo "baseline val voc07 and voc12"
#voc07和voc12基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index6/voc07trainval_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index6/baseline/voc07_sup100/model_best.pth \
	OUTPUT_DIR output/index6/baseline/voc07_sup100/record
echo "Data_processing"
#数据处理
python tools/index6/pick_voc07_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index6/supervision_to_semi-supervision.py \
	--input-dir output/index6/baseline/voc07_sup100/model_best.pth \
	--save-dir output/index6/baseline/voc07_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index6/voc12_index6.json \
    --model-weights output/index6/baseline/voc07_sup100/one_stage.pth \
    --config configs/index6/voc07trainval_sup100_run1_inference.yaml
python ./tools/index6/ScoreFunction.py \
	--static-file temp/index6/voc12_index6.json \
	--indicator-file results/voc12_index6
echo "Generate voc12_label and voc12_unlabel"
#生成有标签和无标签voc12
python ./tools/index6/make_dataset_target.py
python ./tools/index6/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index6/voc07trainval_semi.yaml \
	OUTPUT_DIR output/index6/semi/voc07_incremental \
	MODEL.WEIGHTS output/index6/baseline/voc07_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index6/voc07trainval_semi_val.yaml \
	MODEL.WEIGHTS output/index6/semi/voc07_incremental/model_best.pth
echo "end"