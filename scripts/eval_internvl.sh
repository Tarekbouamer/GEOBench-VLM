#!/usr/bin/env bash
set -e

DATASET_DIR="/home/torres/datasets/GEO/"
RESULTS_DIR="/home/torres/projects/GEOBench-VLM/results"

uv run --group internvl geobench-single --model internvl --data "$DATASET_DIR" \
	--results "$RESULTS_DIR" \
	--max-samples 10 \
	--batch-size 1
