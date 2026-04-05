#!/bin/bash
# 30-layer d=128 homomorphic LM head test
# Assumes 1-layer test succeeded

set -e

PROMPT="${1:-"2+2="}"
LAYERS="${2:-30}"
BATCH_SIZE="${3:-256}"

echo "=== Starting 30-layer d=128 FHE inference ==="
echo "Prompt: $PROMPT"
echo "Layers: $LAYERS"
echo "Batch size: $BATCH_SIZE"
echo "Expected token: 33 (\"1\") or 36 (\"4\")"
echo "=== "

# Run with single thread to avoid memory pressure
RAYON_NUM_THREADS=1 cargo +nightly run --release \
    -p poulpy-chimera \
    --features enable-avx \
    --example test_smollm2_d128_30layer \
    "$PROMPT" "$LAYERS" \
    2>&1 | tee /tmp/fhe_d128_30layer_sequential.log

echo "=== Test complete ==="
echo "Check /tmp/fhe_d128_30layer_sequential.log for results"