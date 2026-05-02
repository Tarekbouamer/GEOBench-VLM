#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_DIR="$REPO_ROOT/Out_weights"
MODEL_DIR="$WEIGHTS_DIR/llava-onevision-qwen2-7b-si"
LLAVA_NEXT_DIR="$REPO_ROOT/LLaVA-NeXT"
ENV_NAME="llavaone1"

# Create dedicated conda env — LLaVA-NeXT requires transformers<4.46
# (apply_chunking_to_forward was removed in 4.46+)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

# Clone and install LLaVA-NeXT (per https://github.com/LLaVA-VL/LLaVA-NeXT README)
if [ ! -d "$LLAVA_NEXT_DIR" ]; then
    git clone https://github.com/LLaVA-VL/LLaVA-NeXT "$LLAVA_NEXT_DIR"
fi
cd "$LLAVA_NEXT_DIR"
pip install --upgrade pip
pip install -e ".[train]"   # install with all extras (e.g. for video support)
# Pin transformers to last version before apply_chunking_to_forward was removed
pip install "transformers==4.45.2"

# Flash-Attention 2 for RTX 3090 (Ampere, CUDA >= 12.0)
# FA3/FA4 require Hopper (H100+) — not supported on RTX 3090
# pip install packaging psutil ninja   # prerequisites; ninja speeds build from 2h -> 3min
# MAX_JOBS=1 pip install flash-attn --no-build-isolation

# Download model weights
mkdir -p "$WEIGHTS_DIR"
hf download lmms-lab/llava-onevision-qwen2-7b-si --local-dir "$MODEL_DIR"
