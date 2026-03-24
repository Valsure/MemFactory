#!/bin/bash
set -x

# ----------------------------------------------------------------------------
# Configuration for MemFactory Training
# ----------------------------------------------------------------------------

# 1. Paths
DATA_PATH="/home/guozl/project/guozl_project/MemAgentDatas/converted_hotpotqa_2000.json" 
MODEL_NAME="Qwen3-1.7B"
MODEL_PATH="/home/models/${MODEL_NAME}"
OUTPUT_DIR="./output/MemoryAgent1.7B"

# 2. Environment & Agent Configuration
# Select the Environment
ENV_TYPE="longcontext" # Options: longcontext, memory_bank
# Select the Agent Module
AGENT_TYPE="memagent"
# Select sub-modules (Not used for naive agent)
EXTRACTOR_TYPE="none"
UPDATER_TYPE="none"
RETRIEVER_TYPE="none"

# 3. Training Hyperparameters
LR=5e-6
BETA=0.001
NUM_GENS=16
MAX_PROMPT_LEN=6000
MAX_GEN_LEN=2500
GRAD_ACC_STEPS=24
SAVE_STEPS=250

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------

# Ensure we are in the project root or adjust paths accordingly
# Assuming script is run from project root or examples/ folder
cd "$(dirname "$0")/.." # Go to project root

echo "Starting MemFactory Training..."
echo "Model: $MODEL_NAME"
echo "Env: $ENV_TYPE | Agent: $AGENT_TYPE"

python3 examples/train_mem_grpo.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --lr "$LR" \
    --beta "$BETA" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --num_generations "$NUM_GENS" \
    --chunk_size 2500 \
    --max_chunk_number 6 \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --max_generate_length "$MAX_GEN_LEN" \
    --save_steps "$SAVE_STEPS" \
    --env_type "$ENV_TYPE" \
    --agent_type "$AGENT_TYPE" \
    --wandb_name "memfactory_${AGENT_TYPE}_${ENV_TYPE}_${MODEL_NAME}" \
    --epoch 2
