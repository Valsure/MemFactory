#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

export HF_TOKEN=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  # Read

# huggingface-cli whoami

hf download Qwen/Qwen3-1.7B-Instruct --repo-type model --local-dir /home/models/Qwen3-1.7B-Instruct --token $HF_TOKEN
