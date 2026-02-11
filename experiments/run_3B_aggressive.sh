# WARNING: The model path below is a placeholder and may not exist.
#!/bin/bash

# Experiment: 3B_aggressive
# Model: 3B (/home/models/Qwen2.5-3B)
# Config: aggressive (Beta: 0.01, Accum: 8, LR: 2e-06)

# Ensure we are in the RL directory (parent of experiments)
cd "$(dirname "$0")/.." || exit

echo "Starting training for 3B_aggressive..."

python RL/mem_grpo_trainer.py \
    --model_name_or_path "/home/models/Qwen2.5-3B" \
    --output_dir "./output/3B_aggressive" \
    --data_path "./datas/train.jsonl" \
    --beta 0.01 \
    --gradient_accumulation_steps 8 \
    --lr 2e-06 \
    --save_steps 9999 \
    --max_generate_length 4096 \
    --epoch 2 \
    --wandb_name "3B_aggressive"

echo "Training finished for 3B_aggressive"
