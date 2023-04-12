#!/bin/bash

# In this example, we show how to train PCL on supervised NLI data.

DATE=$(date '+%Y%m%d%H%M')
SEED=1

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/sup/bert-base-more-augs \
    --num_train_epochs 3 \
    --per_device_train_batch_size 256 \
    --learning_rate 2e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed "$SEED" \
    --augs dp+de+rp+rv \
    --sup_or_unsup sup \
    --use_negative \
    "$@"
