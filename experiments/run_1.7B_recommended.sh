#!/bin/bash

# Experiment: 1.7B_recommended
# Model: 1.7B (/home/models/Qwen3-1.7B)
# Config: recommended (Beta: 0.01, Accum: 4, LR: 1e-06)

# Ensure we are in the RL directory (parent of experiments)
cd "$(dirname "$0")/.." || exit

echo "Starting training for 1.7B_recommended..."

python RL/mem_grpo_trainer.py \
    --model_name_or_path "/home/models/Qwen3-1.7B" \
    --output_dir "./output/1.7B_recommended" \
    --data_path "./datas/train.jsonl" \
    --beta 0.01 \
    --gradient_accumulation_steps 4 \
    --lr 1e-06 \
    --save_steps 9999 \
    --max_generate_length 4096 \
    --epoch 2 \
    --wandb_name "1.7B_recommended"

echo "Training finished for 1.7B_recommended"
