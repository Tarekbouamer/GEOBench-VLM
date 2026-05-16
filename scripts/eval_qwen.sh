#!/usr/bin/env bash
set -e

[ -f .env ] && { set -o allexport; source .env; set +o allexport; }
RESULTS_DIR="${RESULTS_DIR:-results}"

uv run --group qwen geobench-single --model qwen \
    --data "$DATASET_DIR" \
    --results "$RESULTS_DIR" \
    --max-samples 500 \
    --batch-size 1
