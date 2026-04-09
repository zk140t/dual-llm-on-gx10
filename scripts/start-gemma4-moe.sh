#!/bin/bash
# Start Gemma 4 26B-A4B MoE abliterated on port 8001
# Uses custom vllm-gemma4 image (build with: docker build -t vllm-gemma4 .)
#
# Required env vars:
#   HF_TOKEN      - HuggingFace token
#   HF_CACHE      - path to HuggingFace cache dir
#   GEMMA_MODEL   - model ID or local snapshot path
#                   e.g. "WWTCyberLab/gemma-4-26B-A4B-it-abliterated"
#                   or local: "/root/.cache/huggingface/hub/models--WWTCyberLab--gemma-4-26B-A4B-it-abliterated/snapshots/<hash>"
#
# NOTE: If model snapshot is missing preprocessor_config.json, copy it from
#       any complete Gemma 4 model before starting (see README.md).

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${HF_CACHE:?HF_CACHE is required}"
: "${GEMMA_MODEL:?GEMMA_MODEL is required}"

docker run -d \
  --name gemma4 \
  --gpus all \
  --restart always \
  -p 8001:8000 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  --shm-size=1g \
  vllm-gemma4 \
  "$GEMMA_MODEL" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.45

echo "Gemma 4 MoE started on port 8001"
echo "Waiting for startup (~5-8 min from cache)..."
echo "Monitor: docker logs -f gemma4"
echo "Check: curl http://localhost:8001/v1/models"
