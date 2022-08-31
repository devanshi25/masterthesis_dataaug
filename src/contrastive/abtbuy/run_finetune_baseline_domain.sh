#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --export=NONE
MODEL=$1
LR=$2
DOMAIN_AUG=$3
D_AUG=$4
D_AUG_PROB=0.05
CUDA_VISIBLE_DEVICES=1 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=True \
	--d_aug_prob=$D_AUG_PROB \
	--domain_aug=$DOMAIN_AUG \
	--d_aug=$D_AUG \
	--output_dir /ceph/dmittal/di-student/reports/contrastive/Domainaug-abtbuy-clean-PROB-$D_AUG_PROB-AUG-$D_AUG-$LR-${MODEL##*/}/ \
	--temperature=0.07 \
	--per_device_train_batch_size=512 \
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
	--logging_strategy="epoch" 
D_AUG_PROB=0.10
CUDA_VISIBLE_DEVICES=1 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=True \
	--d_aug_prob=$D_AUG_PROB \
	--domain_aug=$DOMAIN_AUG \
	--d_aug=$D_AUG \
	--output_dir /ceph/dmittal/di-student/reports/contrastive/Domainaug-abtbuy-clean-PROB-$D_AUG_PROB-AUG-$D_AUG-$LR-${MODEL##*/}/ \
	--temperature=0.07 \
	--per_device_train_batch_size=512 \
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
	--logging_strategy="epoch" 
D_AUG_PROB=0.20
CUDA_VISIBLE_DEVICES=1 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=True \
	--d_aug_prob=$D_AUG_PROB \
	--domain_aug=$DOMAIN_AUG \
	--d_aug=$D_AUG \
	--output_dir /ceph/dmittal/di-student/reports/contrastive/Domainaug-abtbuy-clean-PROB-$D_AUG_PROB-AUG-$D_AUG-$LR-${MODEL##*/}/ \
	--temperature=0.07 \
	--per_device_train_batch_size=512 \
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
	--logging_strategy="epoch" 
D_AUG_PROB=0.30
CUDA_VISIBLE_DEVICES=1 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=True \
	--d_aug_prob=$D_AUG_PROB \
	--domain_aug=$DOMAIN_AUG \
	--d_aug=$D_AUG \
	--output_dir /ceph/dmittal/di-student/reports/contrastive/Domainaug-abtbuy-clean-PROB-$D_AUG_PROB-AUG-$D_AUG-$LR-${MODEL##*/}/ \
	--temperature=0.07 \
	--per_device_train_batch_size=512 \
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
	--logging_strategy="epoch" 
D_AUG_PROB=0.40
CUDA_VISIBLE_DEVICES=1 python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /ceph/dmittal/di-student/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer=$MODEL \
	--grad_checkpoint=True \
	--d_aug_prob=$D_AUG_PROB \
	--domain_aug=$DOMAIN_AUG \
	--d_aug=$D_AUG \
	--output_dir /ceph/dmittal/di-student/reports/contrastive/Domainaug-abtbuy-clean-PROB-$D_AUG_PROB-AUG-$D_AUG-$LR-${MODEL##*/}/ \
	--temperature=0.07 \
	--per_device_train_batch_size=512 \
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
	--logging_strategy="epoch" 