#!/usr/bin/env bash
set -e

[ -f .env ] && { set -o allexport; source .env; set +o allexport; }

uv run --group qwen geobench-inspect --model qwen \
    --data "$DATASET_DIR"
