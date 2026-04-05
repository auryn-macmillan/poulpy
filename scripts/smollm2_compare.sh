#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:-"What is 2+2? Answer with one token."}

echo "== Exact transformers reference =="
"/home/dev/repo/.venv-smollm/bin/python" "/home/dev/repo/scripts/smollm2_exact.py" "$PROMPT"
echo
echo "== CHIMERA exact-vs-refreshed plaintext =="
cargo +nightly run -p poulpy-chimera --example smollm2_decompose --release -- "$PROMPT"
