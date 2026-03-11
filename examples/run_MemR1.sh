#!/bin/bash
set -x

# ----------------------------------------------------------------------------
# Configuration for MemR1 Training
# ----------------------------------------------------------------------------

# 1. Paths
# Assuming same model and output structure
DATA_PATH="./datas/train.jsonl" # Use the train data from reference code context
MODEL_NAME="Qwen2.5-3B-Instruct"
MODEL_PATH="/home/models/${MODEL_NAME}"
OUTPUT_DIR="./output/mem_r1_qwen25_3b"

# 2. Environment & Agent Configuration
# Select the Environment
ENV_TYPE="memory_bank" 
# Select the Agent Module
AGENT_TYPE="memory_r1_agent" 

# 3. Training Hyperparameters
LR=1e-6
BETA=0.01
MAX_GEN_LEN=2048
GRAD_ACC_STEPS=12
SAVE_STEPS=1000
BATCH_SIZE=1
NUM_GENS=8
EPOCHS=5

# Training Control
# Default: Train both Extractor and Updater
# Add --no_train_extraction or --no_train_update to args if needed

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------

cd "$(dirname "$0")/.." # Go to project root

echo "Starting MemR1 Training..."
echo "Model: $MODEL_NAME"
echo "Env: $ENV_TYPE | Agent: $AGENT_TYPE"

python3 examples/train_mem_grpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --beta "$BETA" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --max_generate_length "$MAX_GEN_LEN" \
    --max_prompt_length 3072 \
    --save_steps "$SAVE_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --num_generations "$NUM_GENS" \
    --env_type "$ENV_TYPE" \
    --agent_type "$AGENT_TYPE" \
    --wandb_name "mem_r1_${AGENT_TYPE}_${ENV_TYPE}_${MODEL_NAME}" \
    --train_extraction \
    --train_update \
    --epoch "$EPOCHS"