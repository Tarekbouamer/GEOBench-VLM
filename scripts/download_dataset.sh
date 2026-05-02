#!/usr/bin/env bash
set -e

# Default: /content/GEOBench-VLM on Colab, override via DATA_PATH env var
DATASET_DIR="${DATA_PATH:-/content/GEOBench-VLM}"

mkdir -p "$DATASET_DIR"

hf download aialliance/GEOBench-VLM --repo-type dataset --local-dir "$DATASET_DIR"

# Extract task zip archives (Single, Temporal, Captioning, Ref-Det, Ref-Seg)
echo "Extracting task archives..."
for zip in "$DATASET_DIR"/*.zip; do
    [ -f "$zip" ] || continue
    folder="${zip%.zip}"
    if [ ! -d "$folder" ]; then
        echo "  Extracting $(basename "$zip")..."
        t0=$(date +%s)
        unzip -q "$zip" -d "$DATASET_DIR"
        t1=$(date +%s)
        echo "    Done in $((t1 - t0))s."
    else
        echo "  $(basename "$folder")/ already extracted, skipping."
    fi
done
echo "Done."
