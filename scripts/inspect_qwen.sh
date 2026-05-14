#!/usr/bin/env bash

DATASET_DIR="/home/torres/datasets/GEO/"

uv run --group qwen geobench-inspect --model qwen --data "$DATASET_DIR"
