# WARNING: The model path below is a placeholder and may not exist.
#!/bin/bash

# Experiment: 3B_baseline
# Model: 3B (/home/models/Qwen2.5-3B)
# Config: baseline (Beta: 0.1, Accum: 1, LR: 5e-07)

# Ensure we are in the RL directory (parent of experiments)
cd "$(dirname "$0")/.." || exit

echo "Starting training for 3B_baseline..."

python RL/mem_grpo_trainer.py \
    --model_name_or_path "/home/models/Qwen2.5-3B" \
    --output_dir "./output/3B_baseline" \
    --data_path "./datas/train.jsonl" \
    --beta 0.05 \
    --gradient_accumulation_steps 4 \
    --lr 5e-07 \
    --save_steps 9999 \
    --max_generate_length 4096 \
    --epoch 2 \
    --wandb_name "3B_baseline"

echo "Training finished for 3B_baseline"
