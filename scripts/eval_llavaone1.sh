#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Default: /content/GEOBench-VLM on Colab, override via DATA_PATH env var
DATA_PATH="${DATA_PATH:-/content/GEOBench-VLM}"
PYTHON="$(conda info --base 2>/dev/null)/envs/llavaone1/bin/python"
# On Colab (no conda) fall back to current python
[ -f "$PYTHON" ] || PYTHON="$(which python)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO_ROOT/eval_geobenchvlm"
"$PYTHON" runmodel.py llavaone1 --data_path "$DATA_PATH"
