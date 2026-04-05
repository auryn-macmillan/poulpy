#!/bin/bash
# Quick check: Run cleartext LM head on existing hidden states
# This determines if noise amplification is the fundamental blocker

set -e

echo "=== Quick LM Head Verification ==="
echo "Running cleartext LM head on d=128, 1-layer hidden states"
echo "This should produce correct token if noise is the only issue"
echo ""

# Run the cleartext test with quick flag
timeout 120 cargo +nightly run --release -p poulpy-chimera --features enable-avx --example test_lm_head_quick 2>&1 | tee /tmp/quick_lm_head_check.log || echo "Timeout or error"

echo ""
echo "=== Results ==="
if grep -q "SUCCESS" /tmp/quick_lm_head_check.log; then
    echo "✅ Cleartext LM head works correctly"
    echo "   → Noise amplification is the fundamental blocker"
elif grep -q "FAIL" /tmp/quick_lm_head_check.log; then
    echo "❌ Cleartext LM head also fails"
    echo "   → Implementation bug (not just noise)"
else
    echo "⚠️  No clear result in log"
fi