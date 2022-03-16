#!/bin/sh


#python download_data.py --data ./data/VOC.yaml
#python download_data.py --data ./data/coco.yaml
echo "Data_processing"
#数据处理
python tools/index1/make_dataset_coco20.py
python tools/index1/make_coco_name_modified.py
python generate_random_supervised_seed_index1_full.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index1/voc+coco_sup100_run1_full.yaml \
	OUTPUT_DIR output/index1/baseline/voc+coco_sup100_full
echo "baseline val voc and coco"
#voc和coco全量基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc+coco_sup100_run1_full_val.yaml \
	MODEL.WEIGHTS output/index1/baseline/voc+coco_sup100_full/model_best.pth
echo "end"