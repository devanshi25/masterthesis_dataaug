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
FROZEN=$6
AUG=$7
PREAUG=$8
COMBO=$9
PROB=0.05
CUDA_VISIBLE_DEVICES=3 python run_finetune_siamese.py \
	--model_pretrained_checkpoint /ceph/dmittal/di-student/reports/contrastive/abtbuy-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$LR-$TEMP-${MODEL##*/}/pytorch_model.bin \
    --do_train \
	--dataset_name=abt-buy \
	--frozen=$FROZEN \
    --train_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--validation_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-train.json.gz \
	--test_file /ceph/dmittal/di-student/data/interim/abt-buy/abt-buy-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer=$MODEL \
	--grad_checkpoint=$CHECKPOINT \
    --output_dir /ceph/dmittal/di-student/reports/contrastive-ft-siamese/Domainaug-abtbuy-clean-COMBO-$COMBO-PROB-$PROB-AUG-$AUG$BATCH-$PREAUG$LR-$TEMP-$FROZEN-${MODEL##*/}/ \
	--per_device_train_batch_size=64 \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG 
