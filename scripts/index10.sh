#!/bin/sh


#echo "Data_processing"
##数据处理
mkdir -p dataseed/index10
mkdir -p temp/index10
mkdir -p datasets/index10/semi/exdark_20%/annotations
mkdir -p datasets/index10/semi/exdark_20%/images
mkdir -p datasets/index10/semi/voc11_label/annotations
mkdir -p datasets/index10/semi/voc11_label/images
mkdir -p datasets/index10/semi/voc11_unlabel/annotations
mkdir -p datasets/index10/semi/voc11_unlabel/images
python generate_random_supervised_seed_index10.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index10/exdark_sup100_run1.yaml \
	OUTPUT_DIR output/index10/baseline/exdark_sup100
echo "baseline val exdark and voc11"
#exdark和voc11基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index10/exdark_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index10/baseline/exdark_sup100/model_best.pth \
	OUTPUT_DIR output/index10/baseline/exdark_sup100/record
echo "voc11 pseudo label"
##voc11伪标签
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index10/exdark_sup100_run1_pseudo.yaml \
	MODEL.WEIGHTS output/index10/baseline/exdark_sup100/model_best.pth \
	OUTPUT_DIR output/pseudo_label/index10
echo "Data_processing"
#数据处理
python tools/index10/pick_exdark_20%.py
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index10/supervision_to_semi-supervision.py \
	--input-dir output/index10/baseline/exdark_sup100/model_best.pth \
	--save-dir output/index10/baseline/exdark_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index10/voc11_index10.json \
    --model-weights output/index10/baseline/exdark_sup100/one_stage.pth \
    --config configs/index10/exdark_sup100_run1_inference.yaml
python ./tools/index10/ScoreFunction.py \
	--static-file temp/index10/voc11_index10.json \
	--indicator-file results/voc11_index10
echo "Generate voc11_label and voc11_unlabel"
#生成有标签和无标签voc11
python ./tools/index10/make_dataset_target.py
python ./tools/index10/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index10/exdark_semi.yaml \
	OUTPUT_DIR output/index10/semi/exdark_incremental \
	MODEL.WEIGHTS output/index10/baseline/exdark_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index10/exdark_semi_val.yaml \
	MODEL.WEIGHTS output/index10/semi/exdark_incremental/model_best.pth
echo "end"