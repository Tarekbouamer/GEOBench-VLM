#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env from repo root if present
if [ -f "$REPO_ROOT/.env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +o allexport
fi

DATASET_DIR="/home/torres/datasets/GEO/"

cd "$REPO_ROOT"
uv run --group lfm geobench-inspect --model lfm --data "$DATASET_DIR" "$@"
