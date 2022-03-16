#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index8
mkdir -p temp/index8
mkdir -p datasets/index8/semi/voc5_20%/annotations
mkdir -p datasets/index8/semi/voc5_20%/images
mkdir -p datasets/index8/semi/objects365_label/annotations
mkdir -p datasets/index8/semi/objects365_label/images
mkdir -p datasets/index8/semi/objects365_unlabel/annotations
mkdir -p datasets/index8/semi/objects365_unlabel/images
python generate_random_supervised_seed_index8.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index8/voc5_sup100_run1.yaml \
	OUTPUT_DIR output/index8/baseline/voc5_sup100
echo "baseline val voc5 and object365_5"
#voc5和object365_5基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index8/voc5_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index8/baseline/voc5_sup100/model_best.pth \
	OUTPUT_DIR output/index8/baseline/voc5_sup100/record
echo "Data_processing"
#数据处理
python tools/index8/pick_voc5_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index8/supervision_to_semi-supervision.py \
	--input-dir output/index8/baseline/voc5_sup100/model_best.pth \
	--save-dir output/index8/baseline/voc5_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference_cpu.py \
    --static-file temp/index8/object365_index8.json \
    --model-weights output/index8/baseline/voc5_sup100/one_stage.pth \
    --config configs/index8/voc5_sup100_run1_inference.yaml
python ./tools/index8/ScoreFunction.py \
	--static-file temp/index8/object365_index8.json \
	--indicator-file results/object365_index8
echo "Generate object365_label and object365_unlabel"
#生成有标签和无标签object365
python ./tools/index8/make_dataset_target.py
python ./tools/index8/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index8/voc5_semi.yaml \
	OUTPUT_DIR output/index8/semi/voc5_incremental \
	MODEL.WEIGHTS output/index8/baseline/voc5_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index8/voc5_semi_val.yaml \
	MODEL.WEIGHTS output/index8/semi/voc5_incremental/model_best.pth
echo "end"