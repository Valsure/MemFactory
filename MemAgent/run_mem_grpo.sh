#!/bin/bash
set -x

# ----------------------------------------------------------------------------
# Configuration for Memory GRPO Training
# ----------------------------------------------------------------------------

# 1. Paths
# The user specified data path relative to MemAgent directory
# Note: 目前看来 Qwen3 系列的训练效果好，Qwen2.5 则不好。
DATA_PATH="./data/eval_50.json" 
MODEL_NAME="Qwen2.5-3B-Instruct"
MODEL_PATH="/home/models/${MODEL_NAME}"
MODEL_TAG="${MODEL_NAME//\//_}"
# Output directory
OUTPUT_DIR="./output/mem_grpo_qwen3_4b"

# 2. Parallelism & Batch Size
# Reference script uses:
#   nodes=4, gpus_per_node=8 -> Total 32 GPUs.
#   train_batch_size=128 (Global Batch Size).
# User constraints:
#   bs=1 (Batch size per device/step).
#   N=8 (Number of GPUs available, likely on 1 node).
#
# To match the reference Global Batch Size of 128:
#   Effective Batch Size = (Batch Size Per Device) * (Num Devices) * (Gradient Accumulation Steps)
#   
# Since mem_grpo_trainer.py uses device_map="auto" (Model Parallelism) and runs as a SINGLE process:
#   Effective Devices = 1 (The model is split across 8 GPUs but acts as 1 logical device)
#   128 = 1 * 1 * Gradient_Accumulation_Steps
#   => Gradient_Accumulation_Steps = 128
#
# If you were running 8 independent processes (DDP), you would set this to 16.
# But the script doesn't support DDP out of the box.
GRAD_ACC_STEPS=24

# 3. Hyperparameters
# Matched to reference script where possible.
LR=5e-6                # Reference: actor_rollout_ref.actor.optim.lr=1e-6
BETA=0.001             # Reference: algorithm.kl_ctrl.kl_coef=0.001 (default in script is 0.1)
MAX_GEN_LEN=3072       # Reference: MAX_NEW_TOKEN=1024 (default in script is 4096)
SAVE_STEPS=1000         # Adjusted for frequent saving. Reference save_freq=10.

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------

# Ensure we are in the directory containing the script if running relatively
# cd "$(dirname "$0")"

echo "Starting training with Model: $MODEL_NAME ($MODEL_PATH), Data: $DATA_PATH"
echo "Gradient Accumulation Steps: $GRAD_ACC_STEPS to match Global BS=128"

python3 mem_grpo_trainer.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --beta "$BETA" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --max_generate_length "$MAX_GEN_LEN" \
    --save_steps "$SAVE_STEPS" \
    --wandb_name "mem_agent_${MODEL_TAG}_acc_steps${GRAD_ACC_STEPS}_lr${LR}"

# Note on hardcoded parameters in mem_grpo_trainer.py that cannot be changed via CLI:
# - epoch: Hardcoded to 2. Reference uses 30.
# - num_generations: Hardcoded to 4. Reference uses 16.
