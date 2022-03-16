#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index5
mkdir -p temp/index5
mkdir -p datasets/index5/semi/tacoa_20%/annotations
mkdir -p datasets/index5/semi/tacoa_20%/images
mkdir -p datasets/index5/semi/tacob_label/annotations
mkdir -p datasets/index5/semi/tacob_label/images
python generate_random_supervised_seed_index5.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index5/taco_a_sup100_run1.yaml \
	OUTPUT_DIR output/index5/baseline/tacoa_sup100
echo "baseline val tacoa and tacob"
#taco_a和taco_b基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index5/taco_a_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index5/baseline/tacoa_sup100/model_best.pth \
	OUTPUT_DIR output/index5/baseline/tacoa_sup100/record
echo "Data_processing"
#数据处理
python tools/index5/pick_taco_a_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index5/supervision_to_semi-supervision.py \
	--input-dir output/index5/baseline/tacoa_sup100/model_best.pth \
	--save-dir output/index5/baseline/tacoa_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index5/tacobtrain.json \
    --model-weights output/index5/baseline/tacoa_sup100/one_stage.pth \
    --config configs/index5/taco_a_sup100_run1_inference.yaml
python ./tools/index5/ScoreFunction.py \
	--static-file temp/index5/tacobtrain.json \
	--indicator-file results/tacobtrain
echo "Generate tacob_label and cocob_unlabel"
#生成有标签和无标签tacob
python ./tools/index5/make_dataset_target.py
python ./tools/index5/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index5/taco_a_semi.yaml \
	OUTPUT_DIR output/index5/semi/tacoa_incremental \
	MODEL.WEIGHTS output/index5/baseline/tacoa_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index5/taco_a_semi_val.yaml \
	MODEL.WEIGHTS output/index5/semi/tacoa_incremental/model_best.pth
echo "end"