#!/bin/bash

# Experiment: 4B_baseline
# Model: 4B (/home/models/qwen3-4b)
# Config: baseline (Beta: 0.1, Accum: 1, LR: 5e-07)

# Ensure we are in the RL directory (parent of experiments)
cd "$(dirname "$0")/.." || exit

echo "Starting training for 4B_baseline..."

python RL/mem_grpo_trainer.py \
    --model_name_or_path "/home/models/qwen3-4b" \
    --output_dir "./output/4B_baseline" \
    --data_path "./datas/train.jsonl" \
    --beta 0.1 \
    --gradient_accumulation_steps 1 \
    --lr 5e-07 \
    --save_steps 9999 \
    --max_generate_length 4096 \
    --epoch 2 \
    --wandb_name "4B_baseline"

echo "Training finished for 4B_baseline"
