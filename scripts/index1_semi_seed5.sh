#!/bin/sh


#python download_data.py --data ./data/VOC.yaml
#python download_data.py --data ./data/coco.yaml
echo "Data_processing"
#数据处理
python ./tools/index1/make_dataset_voc20%.py \
	--dataseed 5 \
	--save-dir datasets/index1/semi/voc20%/seed_5/
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index1/supervision_to_semi-supervision.py \
	--input-dir output/index1/baseline/voc07train+voc12trainval_sup100/model_best.pth \
	--save-dir output/index1/baseline/voc07train+voc12trainval_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index1/coco20train.json \
    --model-weights output/index1/baseline/voc07train+voc12trainval_sup100/one_stage.pth \
    --config configs/index1/voc07train+voc12trainval_sup100_inference.yaml
python ./tools/index1/ScoreFunction.py \
	--static-file temp/index1/coco20train.json \
	--indicator-file results/coco20train
echo "Generate coco_label and coco_unlabel"
#生成有标签和无标签coco
python ./tools/index1/make_dataset_target.py
python ./tools/index1/make_dataset_json2image.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index1/voc07train+voc12trainval_semi_seed5.yaml \
	OUTPUT_DIR output/index1/semi/voc07train+voc12trainval_incremental \
	MODEL.WEIGHTS output/index1/baseline/voc07train+voc12trainval_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc07train+voc12trainval_semi_val.yaml \
	MODEL.WEIGHTS output/index1/semi/voc07train+voc12trainval_incremental/model_best.pth
echo "end"