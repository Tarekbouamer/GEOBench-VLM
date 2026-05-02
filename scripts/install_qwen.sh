#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/Out_weights"
MODEL_DIR="$WEIGHTS_DIR/Qwen2-VL-7B-Instruct"

# Install dependencies (per https://github.com/QwenLM/Qwen3-VL README)
pip install "transformers>=4.57.0"
pip install accelerate
pip install "qwen-vl-utils[decord]==0.0.14"   # [decord] for faster video loading
pip install torch torchvision pillow pandas tqdm

# Flash-Attention 2 for RTX 3090 (Ampere, CUDA >= 12.0)
# FA3/FA4 require Hopper (H100+) — not supported on RTX 3090
# pip install packaging psutil ninja   # prerequisites; ninja speeds build from 2h -> 3min
# MAX_JOBS=1 pip install flash-attn --no-build-isolation

# Download model weights
mkdir -p "$WEIGHTS_DIR"
if [ ! -d "$MODEL_DIR" ]; then
    hf download Qwen/Qwen2-VL-7B-Instruct --local-dir "$MODEL_DIR"
fi
