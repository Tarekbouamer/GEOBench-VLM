#!/usr/bin/env bash
set -e

[ -f .env ] && { set -o allexport; source .env; set +o allexport; }
RESULTS_DIR="${RESULTS_DIR:-results}"

uv run --group llava geobench-single --model llava1pt6 \
    --data "$DATASET_DIR" \
    --results "$RESULTS_DIR" \
    --max-samples 500 \
    --batch-size 1
