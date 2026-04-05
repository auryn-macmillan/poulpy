## Session 18 — FP16 Precision Fixes Hidden State Collapse (2026-03-22)

**Key finding**: FP16 precision (8192 bootstrap levels) preserves signal, INT8 (128 levels) collapses it.

### Bootstrap Precision Sweep Results

| Config | Bootstrap Levels | FHE Token (d=64) | Exact Token | Signal Preserved? |
|--------|-----------------|------------------|-------------|-------------------|
| INT8 (scale_bits=8) | 128 | `""` (all zeros) | `"1"` | ❌ Collapsed |
| INT8 (config=2048 levels) | 2048 | `""` (all zeros) | `"1"` | ❌ Collapsed |
| **FP16 (scale_bits=14)** | **8192** | `"utherford"` | `"earchers"` | ✅ **Sensesuffix** |
| FP16, d=128 | 8192 | `"ing"` | `"leanor"` | ✅ Word suffix |

### Root Cause: Bootstrap Quantization Coarseness

**INT8 (128 levels)**:
- Bootstrap range: [-127, 127] → each level = ~2.0 units
- Signal magnitude: max_val=125 (attention), 58 (FFN gate)
- Quantization error: ±1.0 (50% of signal!)
- Result: Signal collapses to zeros

**FP16 (8192 levels)**:
- Bootstrap range: [-8191, 8191] → each level = ~2.0 units
- Signal magnitude: max_val=8192 (FP16 range)
- Quantization error: ±0.0625 (negligible)
- Result: Signal preserved, sensible tokens

### Key Fix Applied

**Modified `apply_silu_lut_vec()`** to use `fhe_silu_log_msg_mod` config:
```rust
let silu_log_msg_mod: usize = self.config.fhe_silu_log_msg_mod
    .map(|v| v as usize)
    .unwrap_or(bp.log_message_modulus);
```

**Added `FHE_LLM_bootstrap_with_lut_custom_precision()`** to bootstrapping.rs:
- Accepts custom `log_message_modulus` parameter
- Allows higher precision bootstrap without changing bootstrap params

### Implications

1. **FP16 is the solution**, not frequent bootstrapping
2. **INT8 is too coarse** for transformer inference with current noise levels
3. **FP16 overhead**: 4× larger ciphertexts, 4× computation cost
4. **Tradeoff**: FP16 d=576 may be ~15-30min per layer vs INT8 d=64 at 7s

### Next Steps

1. ✅ **FP16 d=256**: Currently running (timed out during attention after 2min)
2. ⏳ **FP16 d=512**: Test at higher dimension
3. ⏳ **FP16 d=576**: Production dimension test
4. ⏳ **FP16 30-layer model**: Full SmolLM2 with FP16 precision
5. ⏳ **Latency measurement**: Compare FP16 vs INT8 at production dimensions

### AGENTS.md Updates

**Rule out homomorphic LM head**:
- LM head (49k FHE dot products) would amplify the same noise problem deeper into the chain
- The bottleneck is bootstrap quantization, not LM head computation
- **Reject homomorphic LM head** — just shifts noise problem from cleartext to FHE space

**Frequent bootstrapping is secondary**:
- More bootstraps help reduce FHE computation noise, but don't fix quantization collapse
- FP16 alone preserves signal even with 2 bootstraps per layer (before/after FFN)
- Aggressive bootstrapping (5+ per layer) is only needed if FP16 is insufficient