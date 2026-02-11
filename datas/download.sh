#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

export HF_TOKEN=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  # Read

# huggingface-cli whoami

hf download Qwen/Qwen2.5-3B --repo-type model --local-dir /home/models/Qwen/Qwen2.5-3B --token $HF_TOKEN
