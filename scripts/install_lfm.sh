#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/Out_weights"
MODEL_DIR="$WEIGHTS_DIR/LFM2.5-VL-450M"

# Install dependencies (per https://huggingface.co/LiquidAI/LFM2.5-VL-450M)
pip install "transformers>=5.1.0"
pip install torch torchvision pillow pandas tqdm

# Flash-Attention 2 for RTX 3090 (Ampere, CUDA >= 12.0)
# FA3/FA4 require Hopper (H100+) — not supported on RTX 3090
# pip install packaging psutil ninja   # prerequisites; ninja speeds build from 2h -> 3min
# MAX_JOBS=1 pip install flash-attn --no-build-isolation

# Download model weights
mkdir -p "$WEIGHTS_DIR"
hf download LiquidAI/LFM2.5-VL-450M --local-dir "$MODEL_DIR"
