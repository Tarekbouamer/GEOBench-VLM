#!/usr/bin/env bash
set -e

[ -f .env ] && { set -o allexport; source .env; set +o allexport; }

mkdir -p "$DATASET_DIR"

echo "Downloading to: $DATASET_DIR"
uv run huggingface-cli download aialliance/GEOBench-VLM --repo-type dataset --local-dir "$DATASET_DIR"

for zip in "$DATASET_DIR"/*.zip; do
    [ -f "$zip" ] || continue
    [ -d "${zip%.zip}" ] && { echo "Skipping $(basename "${zip%.zip}") (exists)"; continue; }
    echo "Extracting $(basename "$zip")..."
    unzip -q "$zip" -d "$DATASET_DIR"
done

# delete zip files after extraction
rm -f "$DATASET_DIR"/*.zip

echo "Done. Dataset at: $DATASET_DIR"
