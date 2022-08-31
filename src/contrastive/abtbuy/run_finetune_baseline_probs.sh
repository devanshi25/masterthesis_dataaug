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
AUG=$5
COMBO=$6
PROB=0.05
CUDA_VISIBLE_DEVICES=5 python run_finetune_baseline.py \
    --do_train \
	--dataset_name=abt-buy \
    --train_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
	--combo=$COMBO \
    --output_dir /ceph/dmittal/di-student/reports/baseline/abtbuy-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-${MODEL##*/}/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG 
PROB=0.10
CUDA_VISIBLE_DEVICES=5 python run_finetune_baseline.py \
    --do_train \
 	--dataset_name=abt-buy \
    --train_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
    --combo=$COMBO \
	--output_dir /ceph/dmittal/di-student/reports/baseline/abtbuy-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-${MODEL##*/}/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG 
PROB=0.20
CUDA_VISIBLE_DEVICES=5 python run_finetune_baseline.py \
    --do_train \
	--dataset_name=abt-buy \
    --train_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
    --combo=$COMBO \
	--output_dir /ceph/dmittal/di-student/reports/baseline/abtbuy-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-${MODEL##*/}/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG 
PROB=0.30
CUDA_VISIBLE_DEVICES=5 python run_finetune_baseline.py \
    --do_train \
	--dataset_name=abt-buy \
    --train_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
    --combo=$COMBO \
	--output_dir /ceph/dmittal/di-student/reports/baseline/abtbuy-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-${MODEL##*/}/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG 
PROB=0.40
CUDA_VISIBLE_DEVICES=5 python run_finetune_baseline.py \
    --do_train \
	--dataset_name=abt-buy \
    --train_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
	--augment_prob=$PROB \
    --combo=$COMBO \
	--output_dir /ceph/dmittal/di-student/reports/baseline/abtbuy-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-${MODEL##*/}/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG 
#--do_param_opt \

