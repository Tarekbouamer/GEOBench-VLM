#!/usr/bin/env bash
set -e

[ -f .env ] && { set -o allexport; source .env; set +o allexport; }

uv run --group lfm geobench-inspect --model lfm \
    --data "$DATASET_DIR"
