#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index9
mkdir -p temp/index9
mkdir -p datasets/index9/semi/coco12_20%/annotations
mkdir -p datasets/index9/semi/coco12_20%/images
mkdir -p datasets/index9/semi/exdark12_label/annotations
mkdir -p datasets/index9/semi/exdark12_label/images
python generate_random_supervised_seed_index9_reverse.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index9/coco12_sup100_run1.yaml \
	OUTPUT_DIR output/index9/baseline/coco12_sup100
echo "baseline val coco12 and exdark12"
#coco12和exdark12基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index9/coco12_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index9/baseline/coco12_sup100/model_best.pth \
	OUTPUT_DIR output/index9/baseline/coco12_sup100/record
echo "Data_processing"
#数据处理
python tools/index9/pick_coco12_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index9/supervision_to_semi-supervision.py \
	--input-dir output/index9/baseline/coco12_sup100/model_best.pth \
	--save-dir output/index9/baseline/coco12_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index9/exdark12_index9.json \
    --model-weights output/index9/baseline/coco12_sup100/one_stage.pth \
    --config configs/index9/coco12_sup100_run1_inference.yaml
python ./tools/index9/ScoreFunction.py \
	--static-file temp/index9/exdark12_index9.json \
	--indicator-file results/exdark12_index9
echo "Generate exdark12_label and exdark12_unlabel"
#生成有标签和无标签exdark12
python ./tools/index9/make_dataset_target_reverse.py
python ./tools/index9/make_dataset_json2image_reverse.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index9/coco12_semi.yaml \
	OUTPUT_DIR output/index9/semi/coco12_incremental \
	MODEL.WEIGHTS output/index9/baseline/coco12_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index9/coco12_semi_val.yaml \
	MODEL.WEIGHTS output/index9/semi/coco12_incremental/model_best.pth
echo "end"