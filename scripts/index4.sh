#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index4
mkdir -p temp/index4
mkdir -p datasets/index4/semi/cocoa_20%/annotations
mkdir -p datasets/index4/semi/cocoa_20%/images
mkdir -p datasets/index4/semi/cocob_label/annotations
mkdir -p datasets/index4/semi/cocob_label/images
mkdir -p datasets/index4/semi/cocob_unlabel/annotations
mkdir -p datasets/index4/semi/cocob_unlabel/images
python generate_random_supervised_seed_index4.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index4/coco_a_sup100_run1.yaml \
	OUTPUT_DIR output/index4/baseline/cocoa_sup100
echo "baseline val cocoa and cocob"
#coco_a和coco_b基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index4/coco_a_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index4/baseline/cocoa_sup100/model_best.pth \
	OUTPUT_DIR output/index4/baseline/cocoa_sup100/record
echo "Data_processing"
#数据处理
python tools/index4/pick_coco_a_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index4/supervision_to_semi-supervision.py \
	--input-dir output/index4/baseline/cocoa_sup100/model_best.pth \
	--save-dir output/index4/baseline/cocoa_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index4/cocobtrain.json \
    --model-weights output/index4/baseline/cocoa_sup100/one_stage.pth \
    --config configs/index4/coco_a_sup100_run1_inference.yaml
python ./tools/index4/ScoreFunction.py \
	--static-file temp/index4/cocobtrain.json \
	--indicator-file results/cocobtrain
echo "Generate cocob_label and cocob_unlabel"
#生成有标签和无标签cocob
python ./tools/index4/make_dataset_target.py
python ./tools/index4/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index4/coco_a_semi.yaml \
	OUTPUT_DIR output/index4/semi/cocoa_incremental \
	MODEL.WEIGHTS output/index4/baseline/cocoa_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index4/coco_a_semi_val.yaml \
	MODEL.WEIGHTS output/index4/semi/cocoa_incremental/model_best.pth
echo "end"
