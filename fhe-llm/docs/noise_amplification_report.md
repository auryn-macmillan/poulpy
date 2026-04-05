# Noise Amplification Analysis Report

## Summary

We investigated why the FHE inference pipeline fails to produce the correct token for the prompt "2+2=" despite the cleartext LM head being correct. The key findings are:

1. **LM Head Correctness**: Unit tests confirm the cleartext LM head computes exact integer logits and returns the correct argmax for all test cases. The LM head is not the source of errors.

2. **Plaintext Baseline Works**: The exact and refreshed plaintext paths (without homomorphic encryption) produce the correct token (ID 3592 "Two") for the prompt "2+2? Answer with one token" with zero error (L-inf=0.0, MAE=0.0).

3. **FHE Noise Amplification**: 
   - **Single-layer FP16 FHE** (d_model=64): Hidden state range `[-38, 44]`, L-inf ≈ 40. Logit error ≈ 800. Predicted token: 22477 ("Gaussian") – incorrect.
   - **Single-layer FP16 with Frequent Bootstraps** (8192 levels): Same hidden state range, same token error. No improvement.
   - **Multi-layer FP16 with Frequent Bootstraps** (d_model=128, 2 layers): The run timed out before completion, but previous attempts showed similar hidden state noise levels.

4. **Noise Threshold**: For correct token prediction, hidden-state L-inf must be < 0.015 (as derived from logit gap analysis). Current FHE hidden state noise is > 30, which is 2000× too large.

## Experiments Conducted

| Experiment | Configuration | Result |
|------------|---------------|--------|
| Plaintext baseline (exact/refreshed) | `d_model=576, n_layers=30` | Token 3592 ("Two"), zero error |
| FP16 single-layer, no frequent bootstraps | `d_model=64, n_layers=1` | Token 22477 ("Gaussian"), hidden L-inf ≈ 40 |
| FP16 single-layer, with frequent bootstraps | Same as above + `fhe_frequent_bootstrap=true` | Same token 22477, no improvement |
| FP16 multi-layer, with frequent bootstraps | `d_model=128, n_layers=2` | Run timed out; previous attempts show similar noise |

## Root Cause

The dominant source of error is **FHE computation noise between bootstrap boundaries** (QKV projection, attention scores, FFN projections). Even with 8192-level bootstrap (scale_bits=14), the noise accumulates to L-inf ≈ 40 after a single layer. Frequent bootstraps did not reduce this noise because the bootstrap quantization error (±0.5 at 8192 levels) is still too coarse for the signal magnitude.

## Recommendations

1. **Increase Bootstrap Precision**: The current implementation caps LUT size at 8192 levels (13 bits). Raising to 16384 levels (14 bits) may reduce quantisation error, but the LUT size limit caused a panic. We need to redesign the LUT construction to support larger sizes or use a different bootstrap strategy.

2. **Scale Down Model Dimensions**: The hidden state noise scales with model width. Reducing `d_model` from 128 to 64 halves the noise (theoretically). However, the signal also halves, so the signal-to-noise ratio may not improve. We need to test with a very small model (e.g., d_model=32) to see if the token becomes correct.

3. **Use FP16 Precision (8192 levels)**: Already in use. No further improvement without higher precision.

4. **Re-evaluate Frequency of Bootstraps**: The current frequent bootstrap is applied after every operation (attention, FFN). More frequent bootstraps (e.g., after every dot product) may help but increases overhead. We need to profile noise growth per operation to find the optimal bootstrap schedule.

5. **Consider FP64 Precision**: Not supported in the current `Precision` enum. Adding FP64 would require significant code changes and larger ciphertexts. Given the limited gain (4× precision vs 2× noise reduction), it may not be worth it.

6. **Alternative: Homomorphic LM Head?**: The homomorphic LM head was rejected because 49k FHE dot products would amplify the same noise problem deeper into the chain. This does not solve the fundamental noise amplification.

7. **Next Concrete Steps**:
   - Run a **single-layer experiment with d_model=32** (FP16, frequent bootstraps) to see if noise drops below the 0.015 threshold.
   - Profile the **per-operation noise growth** (QKV, attention, FFN) to identify which operation contributes most to the noise budget.
   - If noise remains too high, consider **reducing the model size** for the governance use case (the system may tolerate a smaller model with lower accuracy).

## Conclusion

The cleartext LM head is correct. The failure to recover the correct token in FHE is due to **excessive hidden-state noise** after the transformer body. Current bootstrap precision (8192 levels) and frequent bootstrap scheduling are insufficient to keep the noise below the required threshold. The next priority is to **reduce the model width** or **increase bootstrap precision via larger LUTs**. Without addressing the noise amplification, FHE inference for the target governance workload is not feasible.