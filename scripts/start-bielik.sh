#!/bin/bash
# Start Bielik 4.5B on port 8000
# Uses original vllm/vllm-openai:v0.17.1 image (no custom image needed)
#
# Required env vars:
#   HF_TOKEN   - HuggingFace token
#   HF_CACHE   - path to HuggingFace cache dir (e.g. /home/user/.cache/huggingface)

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${HF_CACHE:?HF_CACHE is required}"

docker run -d \
  --name bielik-vllm \
  --gpus all \
  --restart always \
  -p 8000:8000 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  --shm-size=64m \
  vllm/vllm-openai:v0.17.1 \
  speakleash/Bielik-4.5B-v3.0-Instruct \
    --dtype float16 \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.10 \
    --revision 4b1220a9d745bdd874c44347075ef25484ef322b

echo "Bielik started on port 8000"
echo "Check: curl http://localhost:8000/v1/models"
