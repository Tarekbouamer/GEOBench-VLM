#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Default: /content/GEOBench-VLM on Colab, override via DATA_PATH env var
DATA_PATH="${DATA_PATH:-/content/GEOBench-VLM}"

cd "$REPO_ROOT/eval_geobenchvlm"
python runmodel.py lfm --data_path "$DATA_PATH"
