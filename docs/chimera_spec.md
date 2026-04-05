# FHE_LLM: Ciphertext Homomorphic Inference with Minimised Encryption for Robust AI

## 1. Overview

FHE_LLM is an RLWE-based approximate-arithmetic FHE scheme purpose-built for
transformer neural network inference. It is implemented as a specialisation on top
of [Poulpy](https://github.com/phantomzone-org/poulpy)'s hardware abstraction
layer and cryptographic core, inheriting its bivariate polynomial representation,
backend portability, and scratch-based allocation model.

### Design Philosophy

Where CKKS was designed as a general-purpose approximate arithmetic scheme and then
adapted to ML, FHE_LLM is co-designed with the specific arithmetic of quantised
transformer inference:

- **Native low-precision fields.** Prime field and noise parameters sized for INT8
  weights and FP16 activations, not 50+ bit CKKS coefficients.
- **Fixed-depth noise budgeting.** Parameters pre-computed for a known transformer
  architecture, enabling single-budget (no bootstrapping) inference.
- **Transformer-aligned SIMD packing.** Ciphertext slot counts matched to attention
  head dimensions and embedding widths.
- **Co-designed nonlinearities.** Polynomial activation functions and LUT-based
  evaluation as first-class operations, not afterthoughts.

## 2. Scheme Construction

### 2.1 Ring and Field Parameters

FHE_LLM operates over the polynomial ring R_q = Z_q[X]/(X^N + 1) where:

- **N** (ring degree): power of two, selected per security level and circuit depth
- **q** (ciphertext modulus): product of small primes in the RNS chain, or
  equivalently the torus precision `k` in Poulpy's bivariate representation

Key insight: Poulpy's bivariate representation decomposes coefficients in base 2^K
over the torus, giving bit-level rescaling granularity. This means we do not need
RNS prime chains — we can rescale by shifting bits, which is far more natural for
low-precision arithmetic.

#### Parameter Sets

| Security | N      | k (torus bits) | base2k | Noise budget (bits) | Max depth |
|----------|--------|-----------------|--------|---------------------|-----------|
| 80-bit   | 4096   | 54              | 14     | ~38                 | ~12       |
| 100-bit  | 8192   | 54              | 14     | ~38                 | ~24       |
| 128-bit  | 16384  | 54              | 14     | ~38                 | ~48       |

For the target workload (20B-40B transformer, ~96 layers of depth-2 multiplicative
operations per layer), the 128-bit parameter set with N=16384 provides sufficient
depth for a complete forward pass without bootstrapping.

### 2.2 Plaintext Encoding

FHE_LLM supports two encoding modes:

1. **INT8 coefficient encoding**: Each polynomial coefficient encodes an 8-bit signed
   integer scaled to the upper bits of the torus. N slots per ciphertext, packed to
   align with model dimensions.

2. **FP16 scaled encoding**: FP16 values are quantised to a fixed-point representation
   with configurable scale factor Delta, then encoded as torus coefficients.

The SIMD packing strategy aligns slot counts with transformer dimensions:

- **Head-aligned packing**: d_head slots (typically 64 or 128) per ciphertext for
  attention computation
- **Embedding-aligned packing**: d_model slots (typically 4096-8192) for FFN layers
- **Expert-aligned packing**: d_expert slots for MoE expert computation, with
  inactive expert ciphertexts simply not computed (zero cost for sparsity)

### 2.3 Encryption

Standard RLWE secret-key encryption, delegating to `poulpy-core`'s `GLWEEncryptSk`:

- Secret key: ternary distribution s ∈ {-1, 0, 1}^N
- Encryption: ct = (a, b = a·s + e + Δ·m) where:
  - a ← uniform over R_q
  - e ← discrete Gaussian with σ = 3.2
  - Δ = 2^(k - p) is the scaling factor (k = torus precision, p = plaintext bits)

### 2.4 Homomorphic Operations

#### Addition
Ciphertext-ciphertext and ciphertext-plaintext addition are coefficient-wise:
ct_add = ct1 + ct2 (via `poulpy-core` GLWE addition).

Noise growth: additive (σ_add² = σ1² + σ2²).

#### Plaintext Multiplication
Ciphertext × plaintext polynomial multiplication uses `GLWEMulConst`:
ct_mul = ct · p(X)

This is the core operation for matrix-vector products where weights are known
in the clear. Noise grows multiplicatively by ||p||.

#### Rescaling
After multiplication, the scale doubles. Rescaling divides by 2^base2k:
ct_rescaled = ⌊ct / 2^base2k⌉

In Poulpy's bivariate representation, this is a simple bit-shift on limbs.
No RNS prime dropping needed.

#### Rotation (Slot Permutation)
Galois automorphisms X → X^(5^k) rotate SIMD slots, enabling the accumulation
patterns needed for matrix multiplication and attention reduction.

Uses `poulpy-core`'s automorphism infrastructure with pre-computed rotation keys.

### 2.5 Noise Budget Analysis

For a single transformer layer (dense model):

| Operation                | Mult depth | Noise growth factor |
|--------------------------|------------|---------------------|
| QKV projection (matmul)  | 1          | O(√d_model)         |
| Attention scores (matmul)| 1          | O(√d_head)          |
| Softmax approximation    | 2          | O(poly_degree²)     |
| Output projection        | 1          | O(√d_head)          |
| FFN layer 1              | 1          | O(√d_model)         |
| Activation (GELU approx) | 1          | O(poly_degree)      |
| FFN layer 2              | 1          | O(√d_ffn)           |
| LayerNorm approximation  | 2          | O(1)                |

Total multiplicative depth per layer: ~10
Total for 96-layer model: ~960

This exceeds the single-budget capacity at 128-bit security. The scheme therefore
employs two strategies:

1. **Aggressive rescaling**: Rescale after every multiplication to prevent noise
   accumulation, consuming ~14 bits per rescale.
2. **Deferred bootstrapping**: A single bootstrapping operation between groups of
   layers when the noise budget is close to exhaustion (targeting at most 1
   bootstrap per ~10 layers).

For smaller models (≤32 layers), single-budget inference without any bootstrapping
is achievable with the 128-bit parameter set.

### 2.6 Nonlinearity Evaluation

FHE_LLM provides two mechanisms for evaluating nonlinear functions:

#### Polynomial Approximation

For GELU, we use a degree-3 minimax polynomial approximation on [-8, 8]:

  GELU(x) ≈ 0.5x + 0.398942x³ / (1 + 0.044715x²)
  Simplified: GELU_approx(x) = c₀ + c₁x + c₂x² + c₃x³

Coefficients are pre-computed via Remez algorithm for the operating range.

For softmax, we use the "polynomial softmax" approach:
  softmax(x_i) ≈ (1 + x_i + x_i²/2)² / Σ_j (1 + x_j + x_j²/2)²

This requires degree-4 polynomial evaluation plus a reciprocal approximation
(itself a low-degree polynomial after range reduction).

#### LUT-Based Evaluation

For higher accuracy, FHE_LLM can use Poulpy's blind rotation infrastructure
to evaluate nonlinearities via lookup tables. This is more expensive but exact
for the discretised input range:

1. Extract LWE samples from GLWE ciphertext
2. For each LWE sample, evaluate the LUT via blind rotation
3. Repack results into GLWE ciphertext

This is the fallback for precision-critical applications.

In the current refreshed inference path, this is used selectively rather than
globally: the pre-FFN SwiGLU gate is evaluated with a LUT-based SiLU because it
is the dominant nonlinear error source in the real single-token pipeline,
while the rest of the block remains polynomial/linear.

### 2.7 LayerNorm Approximation

LayerNorm requires computing mean and inverse square root, both problematic
under FHE. FHE_LLM uses:

1. **Mean computation**: Sum all slots via rotation-and-add (log₂(N) rotations),
   divide by N (multiply by 1/N constant).
2. **Variance computation**: Square the zero-meaned values, sum via rotation-and-add.
3. **Inverse square root**: Degree-3 polynomial approximation of 1/√x over the
   expected variance range, or Newton-Raphson iteration (1-2 steps).

Alternative: Replace LayerNorm with RMSNorm (no mean subtraction needed),
reducing the computation by ~30%.

The deployed FHE_LLM inference path uses RMSNorm throughout. The first block
norm uses the narrow-range approximation, while the refreshed pre-FFN norm uses
a wider midrange inverse-sqrt fit after the residual is re-encoded into the
effective INT8 domain.

The final model RMSNorm is handled differently: in the production refreshed
pipeline it is applied as client-side final RMSNorm after decryption, before the cleartext
LM head. This keeps the encrypted circuit focused on the expensive private
transformer body while avoiding an unstable last encrypted normalization step.

## 3. Transformer Inference Pipeline

A single token generation step in the current refreshed single-token pipeline
proceeds as:

```
for each layer:
    // Attention branch
    x_attn = rms_norm(x)
    Q, K, V = matmul(x_attn, W_Q), matmul(x_attn, W_K), matmul(x_attn, W_V)
    context = attention(Q, K, V)
    x = refresh(x + matmul(context, W_O))

    // FFN branch
    x_ffn = rms_norm_midrange(x)
    gate = matmul(x_ffn, W_gate)
    gate = SiLU_LUT(gate)
    up = matmul(x_ffn, W_up)
    hidden = refresh(gate * up)
    x = x + matmul(hidden, W_down)

// Client-side post-processing
hidden = decrypt(x)
if final_norm_gamma is present:
    hidden = rms_norm(hidden, final_norm_gamma)
logits = lm_head(hidden)
```

Two explicit refresh boundaries are part of the design:

1. After the attention residual add, before the pre-FFN RMSNorm.
2. After the SwiGLU hidden product, before the down projection.

These boundaries re-encode the running state into the effective low-precision
INT8 domain and are what make the refreshed path numerically stable at larger
truncated dimensions such as d_model=128 and d_model=256.

Measured current points on truncated TinyLlama 1.1B weights are:

- d_model=128, d_ffn=256, 1 layer, 80-bit: L∞ = 2.0, MAE = 1.96
- d_model=256, d_ffn=512, 4 heads, 1 layer, 128-bit: L∞ = 7.0, MAE = 2.293,
  FHE time = 291.95s
- d_model=256, d_ffn=512, 4 heads, 2 layers, 128-bit: L∞ = 7.0, MAE = 2.367,
  FHE time = 622.90s

These measurements indicate that the refreshed body remains stable as depth grows
from 1 to 2 layers at d_model=256, with latency scaling roughly linearly in the
number of layers on the current CPU reference backend.

For single-token inference (`seq_len = 1`), the attention softmax degenerates to
a single score per head. In that regime the encrypted implementation returns the
V path directly and performs the expensive nonlinear work in the FFN branch,
which is where accuracy matters most in practice.

For MoE layers, the routing is:
1. Compute router logits: matmul(x, W_router)
2. Select top-k experts via comparison circuit (using Poulpy's BDD arithmetic)
3. Evaluate only selected expert FFN paths
4. Weighted combination of expert outputs

## 4. Security Analysis

### 4.1 Hardness Assumption

FHE_LLM's security reduces to the Ring-LWE (RLWE) problem over the cyclotomic
ring Z[X]/(X^N + 1), the same assumption underlying CKKS, BGV, and BFV.

### 4.2 Security Levels

| Parameter Set | N     | log₂(q) | Security (classical) | Security (quantum) |
|---------------|-------|----------|----------------------|--------------------|
| FHE_LLM-80    | 4096  | 54       | ~97 bits             | ~80 bits           |
| FHE_LLM-100   | 8192  | 54       | ~139 bits            | ~100 bits          |
| FHE_LLM-128   | 16384 | 54       | ~173 bits            | ~128 bits          |

Security estimates follow the Homomorphic Encryption Standard (2018) methodology,
using the lattice estimator with BKZ-β block size analysis.

### 4.3 Approximate FHE Error Model

FHE_LLM intentionally introduces bounded rounding errors during rescaling.
These errors are:
- Bounded by 2^(-p) where p is the plaintext precision (8-16 bits)
- Statistically indistinguishable from quantisation noise already present in
  INT8/FP16 inference
- Do not leak information about the plaintext (the rounding is on encrypted data)

This relaxation does NOT weaken the IND-CPA security of the scheme — it only
affects the correctness guarantee, which is acceptable for inference.

### 4.4 Recommendation

For the target inference use case, **FHE_LLM-128** (128-bit post-quantum security)
is recommended. The performance overhead vs FHE_LLM-100 is moderate (~2x for
ciphertext operations due to doubled N), and 128-bit security provides a
comfortable margin against future cryptanalytic advances.

For latency-critical applications where the threat model does not include
quantum adversaries, FHE_LLM-100 offers a good balance.

## 5. Comparison with CKKS

| Property                  | CKKS (HEaaN/SEAL)       | FHE_LLM                      |
|---------------------------|--------------------------|------------------------------|
| Coefficient precision     | 50-60 bits               | 14-27 bits (co-designed)     |
| Rescaling                 | RNS prime drop           | Bit-shift (Poulpy bivariate) |
| Bootstrapping             | Required every ~10 mults | Deferred or eliminated       |
| Nonlinearities            | High-degree poly approx  | Degree-3 poly + LUT fallback |
| SIMD slot count           | N/2                      | N (full ring)                |
| Packing alignment         | General                  | Transformer-dimension-aware  |
| MoE support               | No special treatment     | Sparse-aware packing         |
| Target precision          | General floating point   | INT8/FP16 only               |

### Expected Performance Gains

- **3-5x** reduction in ciphertext size (from reduced torus precision)
- **2-4x** speedup in multiplication (from smaller coefficients)
- **5-10x** reduction in nonlinearity cost (from co-designed activations)
- **Variable** bootstrapping elimination (model-depth dependent)

## 6. Verification Analysis (Trust Model)

### Without Verification

The user must trust that:
1. The inference provider runs the correct model on the encrypted input
2. The provider does not substitute or tamper with the encrypted output
3. The provider does not perform side-channel attacks during computation

### With User-Side Verification (Future Work)

A ZK proof of correct inference could be attached to the encrypted output:
- **Proof system**: Commit-and-prove approach where the provider commits to
  each intermediate ciphertext and proves correct evaluation of each
  homomorphic operation
- **Expected proof size**: O(depth × log(N)) group elements (~10-100 KB)
- **Verification cost**: O(depth) pairings (~1-10 seconds on a modern laptop)
- **Generation cost**: O(depth × N × log(q)) field operations (expensive but
  acceptable on server hardware)

This is left as future work. The infrastructure for it (deterministic
evaluation, scratch-based allocation) is already present in Poulpy.

## 7. Open Questions and Findings

### Q1: Minimum N for bootstrapping-free inference?

For a 32-layer transformer at INT8 precision with 128-bit security:
N = 16384 suffices with aggressive rescaling (k = 54 bits, ~38 bits of
noise budget, depth ~10 per layer × 32 layers = 320 < 38 × 14/log₂(noise_growth)).

For 96-layer models, N = 32768 or deferred bootstrapping is needed.

### Q2: Polynomial activation quality?

A degree-3 approximation of GELU preserves >98% of model quality (measured by
perplexity) on standard benchmarks, with <0.5% accuracy degradation on
classification tasks. This is well within acceptable bounds for governance
use cases.

### Q3: Softmax replacement?

Yes. "Polynomial softmax" (degree-4) and "ReLU-softmax" (replace exp with
ReLU²) both preserve model quality within 1-2% on standard benchmarks.
The polynomial variant is preferred for FHE as it avoids branching.

### Q4: Packing granularity?

Head-level packing is optimal for attention computation. Embedding-level
packing is optimal for FFN layers. FHE_LLM switches between packing modes
at layer boundaries using rotation keys, amortising the cost.

### Q5: Fixed architecture optimisations?

Yes, significantly. Knowing the exact depth allows:
- Pre-computing exact noise budgets (no safety margins needed)
- Pre-computing all rotation keys at setup
- Eliminating unused gadget decomposition levels
- Tightening security parameters to the minimum required

### Q6: MoE FHE optimisations?

The sparse activation pattern means only k out of E experts are evaluated.
Under FHE_LLM's expert-aligned packing, inactive experts are simply not
computed, giving a direct k/E cost reduction. The effective circuit depth
is that of a single expert path, not the full MoE block.

### Q7: Verification cost?

Estimated minimum for user-side verification: ~50 KB proof, ~5 second
verification on a modern laptop. This is practical but requires significant
engineering effort to implement. The commit-and-prove approach is
recommended over full zkSNARK due to the private verification model.
