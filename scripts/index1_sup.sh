#!/bin/sh


#python download_data.py --data ./data/VOC.yaml
#python download_data.py --data ./data/coco.yaml
echo "Data_processing"
#数据处理
python tools/index1/make_dataset_coco20.py
python tools/index1/make_coco_name_modified.py
python generate_random_supervised_seed.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index1/voc07train+voc12trainval_sup100_run1.yaml \
	OUTPUT_DIR output/index1/baseline/voc07train+voc12trainval_sup100
echo "baseline val voc and coco"
#voc和coco基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc07train+voc12trainval_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index1/baseline/voc07train+voc12trainval_sup100/model_best.pth
echo "end"