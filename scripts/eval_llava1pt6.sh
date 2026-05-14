#!/usr/bin/env bash
set -e

DATASET_DIR="/home/torres/datasets/GEO/"
RESULTS_DIR="/home/torres/projects/GEOBench-VLM/results"

uv run --group llava geobench-single --model llava1pt6 \
    --data "$DATASET_DIR" \
    --results "$RESULTS_DIR"
