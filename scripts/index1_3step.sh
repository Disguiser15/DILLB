#!/bin/sh


#python download_data.py --data ./data/VOC.yaml
#python download_data.py --data ./data/coco.yaml
echo "Data_processing"
#数据处理
python tools/index1/make_dataset_coco20.py
python tools/index1/make_coco_name_modified.py
mkdir -p datasets/index1/step1/coco_a/annotations
mkdir -p datasets/index1/step1/coco_a/images
mkdir -p datasets/index1/step2/coco_b/annotations
mkdir -p datasets/index1/step2/coco_b/images
mkdir -p datasets/index1/step3/coco_c/annotations
mkdir -p datasets/index1/step3/coco_c/images
python tools/index1/make_coco20_train_divided.py
python tools/index1/make_cocoabc_train_json2image.py
mkdir -p datasets/index1/step1/coco_a/val/annotations
mkdir -p datasets/index1/step1/coco_a/val/images
mkdir -p datasets/index1/step2/coco_b/val/annotations
mkdir -p datasets/index1/step2/coco_b/val/images
mkdir -p datasets/index1/step3/coco_c/val/annotations
mkdir -p datasets/index1/step3/coco_c/val/images
python tools/index1/make_coco20_val_divided.py
python tools/index1/make_cocoabc_val_json2image.py
python tools/index1/make_dataset_voc2007train_voc2012trainval.py
python tools/index1/make_dataset_voc2007test.py
mkdir -p datasets/index1/step2/voc+coco_a/annotations
mkdir -p datasets/index1/step2/voc+coco_a/images
mkdir -p datasets/index1/step2/voc+coco_a/test/annotations
mkdir -p datasets/index1/step2/voc+coco_a/test/images
python tools/index1/make_append_step2.py
mkdir -p datasets/index1/step3/voc+coco_a+coco_b/annotations
mkdir -p datasets/index1/step3/voc+coco_a+coco_b/images
mkdir -p datasets/index1/step3/voc+coco_a+coco_b/test/annotations
mkdir -p datasets/index1/step3/voc+coco_a+coco_b/test/images
python tools/index1/make_append_step3.py
mkdir -p datasets/index1/step1/semi/cocoa_label/annotations
mkdir -p datasets/index1/step1/semi/cocoa_label/images
mkdir -p datasets/index1/step1/semi/cocoa_unlabel/annotations
mkdir -p datasets/index1/step1/semi/cocoa_unlabel/images
mkdir -p datasets/index1/step2/semi/voc+cocoa_20%/annotations
mkdir -p datasets/index1/step2/semi/voc+cocoa_20%/images
mkdir -p datasets/index1/step2/semi/cocob_label/annotations
mkdir -p datasets/index1/step2/semi/cocob_label/images
mkdir -p datasets/index1/step2/semi/cocob_unlabel/annotations
mkdir -p datasets/index1/step2/semi/cocob_unlabel/images
mkdir -p datasets/index1/step3/semi/voc+cocoa+cocob_20%/images
mkdir -p datasets/index1/step3/semi/voc+cocoa+cocob_20%/annotations
mkdir -p datasets/index1/step3/semi/cococ_label/annotations
mkdir -p datasets/index1/step3/semi/cococ_label/images
mkdir -p datasets/index1/step3/semi/cococ_unlabel/annotations
mkdir -p datasets/index1/step3/semi/cococ_unlabel/images
python generate_random_supervised_seed_index1_3step.py
echo "Fully supervised baseline"
#全监督基线
python train_net.py \
	--num-gpus 4 \
	--config configs/index1/voc_sup100_run1.yaml \
	OUTPUT_DIR output/index1/baseline/voc_sup100
echo "baseline val voc and cocoa"
#voc和cocoa基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc_sup100_run1_val.yaml \
	MODEL.WEIGHTS output/index1/baseline/voc_sup100/model_best.pth \
	OUTPUT_DIR output/step1/baseline/record
echo "Data_processing"
#数据处理
python ./tools/index1/make_dataset_voc20%.py \
	--dataseed 0 \
	--save-dir datasets/index1/step1/semi/voc20%/
echo "Supervision to Semi-supervision weight"
#权重转化
python ./tools/index1/supervision_to_semi-supervision.py \
	--input-dir output/index1/baseline/voc_sup100/model_best.pth \
	--save-dir output/index1/baseline/voc_sup100/one_stage.pth
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index1/cocoatrain.json \
    --model-weights output/index1/baseline/voc_sup100/one_stage.pth \
    --config configs/index1/voc_sup100_run1_inference.yaml
python ./tools/index1/ScoreFunction.py \
	--static-file temp/index1/cocoatrain.json \
	--indicator-file results/cocoatrain_1step
echo "Generate cocoa_label and cocoa_unlabel"
#生成有标签和无标签cocoa
python ./tools/index1/make_dataset_target_1step.py
python ./tools/index1/make_dataset_json2image_1step.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa_semi.yaml \
	OUTPUT_DIR output/index1/semi/step1_incremental \
	MODEL.WEIGHTS output/index1/baseline/voc_sup100/one_stage.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa_semi_val.yaml \
	MODEL.WEIGHTS output/index1/semi/step1_incremental/model_best.pth \
	OUTPUT_DIR output/step1/semi/record
echo "baseline val voc_cocoa and cocob"
#voc_cocoa和cocob基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa+cocob_semi_val1.yaml \
	MODEL.WEIGHTS output/index1/semi/step1_incremental/model_best.pth \
	OUTPUT_DIR output/step2/baseline/record
echo "Data_processing"
#数据处理
python tools/index1/pick_step2_20%.py
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index1/cocobtrain.json \
    --model-weights output/index1/semi/step1_incremental/model_best.pth \
    --config configs/index1/cocob_step2_inference.yaml
python ./tools/index1/ScoreFunction.py \
	--static-file temp/index1/cocobtrain.json \
	--indicator-file results/cocobtrain_2step
echo "Generate cocob_label and cocob_unlabel"
#生成有标签和无标签cocob
python ./tools/index1/make_dataset_target_2step.py
python ./tools/index1/make_dataset_json2image_2step.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa+cocob_semi.yaml \
	OUTPUT_DIR output/index1/semi/step2_incremental \
	MODEL.WEIGHTS output/index1/semi/step1_incremental/model_best.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa+cocob_semi_val2.yaml \
	MODEL.WEIGHTS output/index1/semi/step2_incremental/model_best.pth \
	OUTPUT_DIR output/step2/semi/record
echo "baseline val voc_cocoa_cocob and cococ"
#voc_cocoa_cocob和cococ基线
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa+cocob+cococ_semi_val1.yaml \
	MODEL.WEIGHTS output/index1/semi/step2_incremental/model_best.pth \
	OUTPUT_DIR output/step3/baseline/record
echo "Data_processing"
#数据处理
python tools/index1/pick_step3_20%.py
echo "Active sample selection"
#主动样本挑选
python inference.py \
    --static-file temp/index1/cococtrain.json \
    --model-weights output/index1/semi/step2_incremental/model_best.pth \
    --config configs/index1/cococ_step3_inference.yaml
python ./tools/index1/ScoreFunction.py \
	--static-file temp/index1/cococtrain.json \
	--indicator-file results/cococtrain_3step
echo "Generate cococ_label and cococ_unlabel"
#生成有标签和无标签cococ
python ./tools/index1/make_dataset_target_3step.py
python ./tools/index1/make_dataset_json2image_3step.py
echo "semi-supervised"
##半监督
python train_net.py \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa+cocob+cococ_semi.yaml \
	OUTPUT_DIR output/index1/semi/step3_incremental \
	MODEL.WEIGHTS output/index1/semi/step2_incremental/model_best.pth
echo "semi-supervised-val"
##半监督评估
python train_net.py \
	--eval-only \
	--num-gpus 4 \
	--config configs/index1/voc+cocoa+cocob+cococ_semi_val2.yaml \
	MODEL.WEIGHTS output/index1/semi/step3_incremental/model_best.pth \
	OUTPUT_DIR output/step3/semi/record
echo "end"
