#!/bin/bash
# Start Gemma 4 31B dense on port 8001 (slower alternative to MoE)
# ~6 tok/s vs ~24 tok/s for MoE — use when reasoning depth matters more than speed
#
# Required env vars:
#   HF_TOKEN      - HuggingFace token
#   HF_CACHE      - path to HuggingFace cache dir
#   GEMMA_MODEL   - model ID or local snapshot path
#                   e.g. "google/gemma-4-31B-it" (censored)
#                   or abliterated local path

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
    --gpu-memory-utilization 0.60

echo "Gemma 4 31B dense started on port 8001"
echo "Note: uses 76GB VRAM. Adjust bielik gpu-memory-utilization if both running."
echo "Monitor: docker logs -f gemma4"
