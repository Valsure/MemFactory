#!/bin/bash

# Experiment: 4B_recommended
# Model: 4B (/home/models/qwen3-4b)
# Config: recommended (Beta: 0.01, Accum: 4, LR: 1e-06)

# Ensure we are in the RL directory (parent of experiments)
cd "$(dirname "$0")/.." || exit

echo "Starting training for 4B_recommended..."

python RL/mem_grpo_trainer.py \
    --model_name_or_path "/home/models/qwen3-4b" \
    --output_dir "./output/4B_recommended" \
    --data_path "./datas/train.jsonl" \
    --beta 0.01 \
    --gradient_accumulation_steps 4 \
    --lr 1e-06 \
    --save_steps 9999 \
    --max_generate_length 4096 \
    --epoch 2 \
    --wandb_name "4B_recommended"

echo "Training finished for 4B_recommended"
