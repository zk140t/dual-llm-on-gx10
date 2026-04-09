# Dual LLM on NVIDIA GB10 — vLLM Setup Guide

Running two LLMs simultaneously on a single NVIDIA GB10 (Grace Blackwell, 128GB unified memory) using Docker + vLLM. Includes Gemma 4 support workarounds, MoE architecture performance notes, and Telegram bot integration.

## Hardware

- **GPU:** NVIDIA GB10 (Grace Blackwell) — 128GB unified CPU+GPU memory
- **CUDA:** 13.0+
- **Disk:** 500GB+ recommended (models are large)

## What's running

| Container | Port | Model | VRAM | Speed |
|-----------|------|-------|------|-------|
| `bielik-vllm` | 8000 | speakleash/Bielik-4.5B-v3.0-Instruct | ~12GB | ~18 tok/s |
| `gemma4` | 8001 | Gemma 4 26B-A4B MoE abliterated | ~55GB | ~24 tok/s |

**Total GPU used: ~67GB / 128GB**

Both expose OpenAI-compatible APIs (`/v1/chat/completions`).

---

## Why MoE beats a smaller dense model in speed

The Gemma 4 26B-A4B uses **Mixture of Experts** architecture:
- 26B total parameters split across 128 expert sub-networks
- Per token: only top-8 experts activate = **~3.8B active params**
- Reads ~7GB of weights per token vs 9GB for Bielik 4.5B dense
- Result: faster generation despite 6× more total parameters

Real benchmark (same prompt, same hardware, all models loaded simultaneously):

| Model | Architecture | Active params | VRAM | tok/s |
|-------|-------------|-------------|------|-------|
| Bielik 4.5B dense | Dense | 4.5B | ~12GB | ~18 |
| Gemma 4 26B MoE | MoE | ~3.8B | ~55GB | ~24 |
| Gemma 4 31B NVFP4-turbo | Dense + FP4 quant | 31B | ~25GB | ~9.5 |

MoE is fastest despite being the largest total-parameter model. NVFP4 quantization reduces VRAM significantly but dense attention to all 31B params still loses to MoE's sparse activation.

---

## Problem: vLLM doesn't support Gemma 4 yet

`vllm/vllm-openai:latest` ships with transformers 4.57.6 which lacks `gemma4` architecture.

**Fix:** build a custom image that upgrades transformers from git.

### Dockerfile

See `Dockerfile` in this repo.

```bash
docker build -t vllm-gemma4 .
```

---

## Problem: MoE abliterated models missing processor config

Abliterated community models often strip multimodal processor files. vLLM fails with:
```
OSError: Can't load feature extractor... ensure presence of a 'preprocessor_config.json'
```

**Fix:** copy `processor_config.json` from any complete Gemma 4 repo into the model snapshot directory:

```bash
cp /path/to/complete-gemma4-model/processor_config.json \
   /path/to/abliterated-model/snapshot/preprocessor_config.json

cp /path/to/complete-gemma4-model/processor_config.json \
   /path/to/abliterated-model/snapshot/processor_config.json
```

---

## Setup

### 1. Set environment variables

```bash
export HF_TOKEN=your_huggingface_token_here
export HF_CACHE=/path/to/huggingface/cache   # e.g. /home/user/.cache/huggingface
```

### 2. Build custom vLLM image

```bash
docker build -t vllm-gemma4 .
```

### 3. Download models

```bash
# Download any model to HF cache
docker run --rm --entrypoint '' \
  -v $HF_CACHE:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  vllm-gemma4 \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('MODEL_ID_HERE')"
```

### 4. Start Bielik (port 8000)

```bash
./scripts/start-bielik.sh
```

### 5. Start Gemma 4 MoE (port 8001)

```bash
./scripts/start-gemma4-moe.sh
```

---

## Memory allocation math

`--gpu-memory-utilization` tells vLLM what fraction of **total** GPU memory to reserve (model weights + KV cache).

```
Total GPU: 128GB

Bielik:   0.10 × 128 = 12.8GB   (9GB weights + 3.8GB KV cache)
Gemma MoE: 0.45 × 128 = 57.6GB  (49GB weights + 8.6GB KV cache)
─────────────────────────────────
Total used: ~70GB  |  Free: ~58GB
```

Tune `--gpu-memory-utilization` based on concurrent user load.
More KV cache = more simultaneous requests, not faster generation.

---

## GPU memory utilization tuning guide

| Concurrent users (Bielik) | Recommended `--gpu-memory-utilization` |
|--------------------------|----------------------------------------|
| 1–3 | 0.10 (current) |
| 3–5 | 0.15 |
| 5–10 | 0.20 |

Adjust Gemma 4 MoE down proportionally to stay within 128GB total.

---

## Switching models at runtime

No hot-reload — requires stop/rm/run cycle:

```bash
docker stop gemma4 && docker rm gemma4
# run new docker run command with different model
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start-bielik.sh` | Start Bielik on port 8000 |
| `scripts/start-gemma4-moe.sh` | Start Gemma 4 MoE on port 8001 |
| `scripts/start-gemma4-31b-dense.sh` | Start Gemma 4 31B dense on port 8001 (slower) |
| `scripts/switch-model.sh` | Helper to stop/start gemma4 container |

---

## Verified model combinations

| Port 8001 model | VRAM | tok/s | Notes |
|-----------------|------|-------|-------|
| `WWTCyberLab/gemma-4-26B-A4B-it-abliterated` | ~55GB | ~24 | **Recommended** — fast MoE, uncensored |
| `paperscarecrow/Gemma-4-31B-it-abliterated` | ~76GB | ~6 | Slower, denser reasoning |
| `google/gemma-4-31B-it` | ~76GB | ~6 | Censored, reference version |
| `LilaRest/gemma-4-31B-it-NVFP4-turbo` | ~25GB | ~9.5 | ❌ Tested, rejected — MoE faster despite 2.2× more VRAM |

---

## Integration with Telegram bot (agent-tomek)

Both models expose OpenAI-compatible APIs. Any client using `openai` SDK works:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://YOUR_GX10_IP:8001/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="MODEL_NAME_OR_LOCAL_PATH",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

For Telegram bots: set LLM base URL and model name in config, increase request timeout to 180s for large models.

---

## Tested on

- NVIDIA GB10 (Grace Blackwell, 128GB unified memory)
- Ubuntu 24.04
- CUDA 13.0, Driver 580.126.09
- vLLM 0.19.0 (via custom image)
- Docker 27+
