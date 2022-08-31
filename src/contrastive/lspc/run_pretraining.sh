#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --export=NONE
MODEL=$1
CHECKPOINT=$2
BATCH=$3
LR=$4
TEMP=$5
SIZE=$6
AUG=$7
COMBO=$8
PROB=0.10
CUDA_VISIBLE_DEVICES=4 python run_pretraining.py \
    --do_train \
    --train_file /ceph/dmittal/di-student/data/processed/wdc-lspc/contrastive/pre-train/computers/computers_train_$SIZE.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/raw/wdc-lspc/training-sets/computers_train_$SIZE.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /ceph/dmittal/di-student/reports/contrastive/computers-$SIZE-$AUG$PROB$BATCH-$LR-$TEMP-${MODEL##*/}/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=200 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG \