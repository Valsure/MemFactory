#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

export HF_TOKEN=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  # Read

# huggingface-cli whoami

hf download Qwen/Qwen3-14B --repo-type model --local-dir /home/gq/model/Qwen/Qwen3-14B --token $HF_TOKEN
