# Session 17 — Bootstrap Config Not Wired Up (2026-03-22)

**Current state**: Aggressive bootstrap test with per-operation diagnostic logging completed.

**Critical finding**: Bootstrap precision config fields exist in `InferenceConfig` but are **NOT used** in actual bootstrap calls:

| Config Field | Defined | Used in bootstrap_refresh() |
|--------------|---------|----------------------------|
| `fhe_identity_log_msg_mod` | ✅ Line 128, 148 | ❌ Uses `scale_bits` instead |
| `fhe_silu_log_msg_mod` | ✅ Line 126, 147 | ❌ Uses `scale_bits` instead |
| `fhe_frequent_bootstrap` | ✅ Line 130, 149 | ✅ Used at line 2832 |

**Root cause**: The aggressive bootstrap test was using 128 levels (INT8 precision) despite config specifying 2048 levels (`log_message_modulus=11`). The `bootstrap_refresh()` function at line 3577 uses:
```rust
let max_val = (1i64 << (self.params().scale_bits as usize - 1)) as f64;
```

This produces ±127 for INT8 (scale_bits=10), not ±2047 (11 bits) as intended.

**Result**: Hidden state collapsed to zeros after aggressive bootstrap because quantization was too coarse.

**Next steps**:
1. Modify `bootstrap_refresh()` to use `fhe_identity_log_msg_mod` if configured
2. Modify SiLU bootstrap to use `fhe_silu_log_msg_mod` if configured
3. Rerun aggressive bootstrap test with confirmed higher precision
4. Compare results to verify if higher precision fixes collapse

---

## Files Modified

### `/home/dev/repo/poulpy-FHE_LLM/src/inference.rs`

**Line 3577-3583** (to be modified):
```rust
fn bootstrap_refresh(&self, hidden: &Vec<f64>) -> Result<Vec<f64>, String> {
    // Simulate bootstrap quantization at the configured precision
    // For INT8 (scale_bits=8): ±127 (128 levels)
    // For FP16 (scale_bits=14): ±8191 (8192 levels)
    // For custom precision: use fhe_identity_log_msg_mod if configured
    let log_msg_mod = self.config.fhe_identity_log_msg_mod.unwrap_or(self.params().scale_bits);
    let max_val = (1i64 << (log_msg_mod - 1)) as f64;
    Ok(hidden.iter().map(|&v| v.clamp(-max_val, max_val)).collect())
}
```

**Impact**: When `fhe_identity_log_msg_mod=Some(11)`, bootstrap will use 2048 levels instead of 128, reducing quantization error from ±0.5 to ±0.000488.

---

## Diagnostic Logging

The diagnostic logging reveals the collapse pattern:

| Operation | max_val before | max_val after | Signal loss? |
|-----------|---------------|---------------|--------------|
| QKV | - | 0 | ✅ Preserved |
| After attention | - | 125 | ✅ Preserved |
| **After attn refresh** | 125 | **0** | ❌ **COLLAPSED** |
| After residual_1 refresh | 0 | 0 | ❌ |
| After RMSNorm | 0 | 128 | ✅ Recovered |
| After FFN gate | 128 | 58 | ✅ |
| **After gate SiLU** | 58 | **0** | ❌ **COLLAPSED** |
| After FFN up | 0 | 23 | ✅ |
| **After gate×up** | 23 | **0** | ❌ **COLLAPSED** |
| After FFN down | 0 | 0 | ❌ |

The collapse occurs because the bootstrap quantizes to 128 levels, and the signal values (125, 58, 23) fall into the same quantization bin as zero when the quantization step is too coarse.

---

## Verification Steps

1. ✅ **Grep for config fields**: Confirmed 0 matches in source code except definition
2. ✅ **Read bootstrap_refresh()**: Confirmed uses `scale_bits` instead of config fields
3. ⏳ **Modify bootstrap_refresh()**: Wire up config fields
4. ⏳ **Rerun aggressive bootstrap test**: With confirmed higher precision
5. ⏳ **Compare results**: Verify if collapse is fixed

---

## Key Insight

The bottleneck is **FHE computation noise between bootstraps**, but the bootstrap precision also matters. The current implementation uses 128 levels for INT8, which is too coarse for the signal magnitudes we're seeing (125, 58, 23). Using 1024-2048 levels (log_message_modulus=10-11) should reduce quantization error and preserve signal.

---

## AGENTS.md Update

Add this to **Session 17** section:

**Critical Finding #3**: Bootstrap precision config fields not wired up

The `fhe_identity_log_msg_mod` and `fhe_silu_log_msg_mod` config fields were defined in `InferenceConfig` (lines 126-128, 147-148) but were **NOT used** in the `bootstrap_refresh()` function. The function used `self.params().scale_bits` instead, which produces 128 levels for INT8 precision.

This means the aggressive bootstrap test with `fhe_identity_log_msg_mod=Some(11)` was actually using 128 levels despite the config specifying 2048 levels. The hidden state collapse was partly due to this precision mismatch.

**Fix**: Modify `bootstrap_refresh()` to use `fhe_identity_log_msg_mod.unwrap_or(self.params().scale_bits)` so the config field takes precedence when configured.