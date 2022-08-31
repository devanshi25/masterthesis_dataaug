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
AUG=$6
COMBO=$7
PROB=0.05
CUDA_VISIBLE_DEVICES=7 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /ceph/dmittal/di-student/reports/contrastive/abtbuy-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
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
	--augment=$AUG 
PROB=0.10
CUDA_VISIBLE_DEVICES=6 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /ceph/dmittal/di-student/reports/contrastive/abtbuy-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
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
	--augment=$AUG 
PROB=0.20
CUDA_VISIBLE_DEVICES=6 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /ceph/dmittal/di-student/reports/contrastive/abtbuy-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
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
	--augment=$AUG 
PROB=0.30
CUDA_VISIBLE_DEVICES=6 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /ceph/dmittal/di-student/reports/contrastive/abtbuy-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
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
	--augment=$AUG 
PROB=0.40
CUDA_VISIBLE_DEVICES=6 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /ceph/dmittal/di-student/reports/contrastive/abtbuy-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/ \
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
	--augment=$AUG 
