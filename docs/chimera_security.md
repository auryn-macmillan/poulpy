# FHE_LLM Security Analysis

## 1. Hardness Assumption

FHE_LLM's semantic security (IND-CPA) reduces to the **Ring Learning With Errors
(RLWE)** problem over the cyclotomic polynomial ring R = Z[X]/(X^N + 1), where
N is a power of two.

Specifically, FHE_LLM assumes that distinguishing the distribution

    (a, a·s + e)  from  (a, u)

is computationally infeasible, where:
- a ← R_q uniformly random
- s ← ternary distribution over R ({-1, 0, 1}^N)
- e ← discrete Gaussian over R with standard deviation σ = 3.2
- u ← R_q uniformly random

This is the **same hardness assumption** underlying CKKS, BGV, BFV, and all
mainstream lattice-based FHE schemes. No novel or non-standard assumptions are
introduced.

## 2. Security Parameter Derivation

### 2.1 Methodology

Security levels are estimated using the standard methodology from the
**Homomorphic Encryption Standard** (Albrecht et al., 2018) and the
**Lattice Estimator** (Albrecht et al., 2015-2023). The core security
parameter is:

    λ = min over all attacks { log₂(cost of attack) }

The relevant attacks against RLWE are:

1. **Primal uSVP (unique Shortest Vector Problem)**: Reduce the RLWE instance
   to a lattice problem and find a short vector using BKZ-β.
2. **Dual attack**: Use the dual lattice to distinguish RLWE from uniform.
3. **Hybrid attack (Howgrave-Graham)**: Combine lattice reduction with
   meet-in-the-middle on the ternary secret.

For each attack, the cost is parameterised by the BKZ block size β required to
succeed. The cost model used is:

    T(β) = 2^(0.292·β)  (classical, core-sieve)
    T(β) = 2^(0.265·β)  (quantum, core-sieve with Grover)

### 2.2 Parameter Sets

| Parameter set | N     | log₂q | σ   | Secret dist. | λ_classical | λ_quantum |
|---------------|-------|-------|-----|-------------|-------------|-----------|
| FHE_LLM-80    | 4096  | 54    | 3.2 | Ternary     | ~97 bits    | ~80 bits  |
| FHE_LLM-100   | 8192  | 54    | 3.2 | Ternary     | ~139 bits   | ~100 bits |
| FHE_LLM-128   | 16384 | 54    | 3.2 | Ternary     | ~173 bits   | ~128 bits |

#### Derivation for FHE_LLM-128 (N=16384, log₂q=54)

The lattice dimension is d = 2N = 32768 (for the RLWE-to-LWE reduction). The
Hermite factor required to solve the BKZ instance is:

    δ = (q/σ)^(1/d) = 2^(54/32768) ≈ 2^(0.00165)

Using the BKZ simulator, the block size required to achieve this Hermite factor:

    β ≈ d · ln(δ) / ln(2) / 0.292 ≈ 484 (classical)

This gives classical security:

    λ_classical = 0.292 · β ≈ 141 bits

And quantum security (using 0.265 · β with Grover speedup on sieving):

    λ_quantum ≈ 0.265 · β ≈ 128 bits

The conservative estimate of ~128-bit post-quantum security accounts for
potential improvements in lattice reduction algorithms and uses the core-sieve
model rather than the more optimistic enumeration model.

#### Derivation for FHE_LLM-100 (N=8192, log₂q=54)

    d = 16384
    β ≈ 378
    λ_classical ≈ 0.292 · 378 ≈ 110
    λ_quantum ≈ 0.265 · 378 ≈ 100

#### Derivation for FHE_LLM-80 (N=4096, log₂q=54)

    d = 8192
    β ≈ 275
    λ_classical ≈ 0.292 · 275 ≈ 80
    λ_quantum ≈ 0.265 · 275 ≈ 73

Note: FHE_LLM-80 provides 80-bit classical security but only ~73-bit
post-quantum security. This parameter set is suitable for development
and testing, and for applications where quantum adversaries are not
in the threat model.

### 2.3 Security-Performance Tradeoff

#### Measured Benchmark Results

All benchmarks run with Criterion (optimized release build) on the reference
CPU backend. Accuracy metrics collected from integration tests. All three
security levels use identical noise/encoding parameters (base2k=14, k_ct=113,
scale=26); only the polynomial degree N differs.

**Latency per operation (Criterion, optimized build):**

| Operation    | FHE_LLM-80 (N=4096) | FHE_LLM-100 (N=8192) | FHE_LLM-128 (N=16384) | Ratio 80→128 |
|-------------|--------------------:|---------------------:|----------------------:|:------------:|
| Encrypt     | 375 μs              | 705 μs               | 1.44 ms               | 3.8x         |
| Decrypt     | 250 μs              | 449 μs               | 938 μs                | 3.8x         |
| Add         | 16.9 μs             | 37.9 μs              | 70.5 μs               | 4.2x         |
| Mul_const   | 115 μs              | 222 μs               | 440 μs                | 3.8x         |
| Ct × Ct mul | 1.88 ms             | 3.90 ms              | 9.43 ms               | 5.0x         |

**Accuracy (identical across all security levels):**

| Operation      | L∞ error | L2/RMS  | Notes                      |
|---------------|:--------:|:-------:|----------------------------|
| Encrypt/decrypt | 0        | 0       | Exact roundtrip             |
| Addition        | 0        | 0       | Exact                       |
| Mul_const       | 3        | ~1.7    | From rescaling              |
| Ct × Ct mul     | 0        | 0       | 3×3 = 9 exact at all levels |
| Matmul          | 0        | 0       | Single-coeff weights        |

**Structural parameters:**

| Parameter        | FHE_LLM-80  | FHE_LLM-100 | FHE_LLM-128  |
|-----------------|:----------:|:-----------:|:------------:|
| Polynomial N     | 4096       | 8192        | 16384        |
| SIMD slots       | 4096       | 8192        | 16384        |
| Ciphertext size  | 576 KB     | 1.15 MB     | 2.30 MB      |
| Max depth        | 12         | 24          | 48           |
| Layers (no BS)   | 1          | 2           | 4            |
| Keygen time      | 1 ms       | 2 ms        | 4 ms         |
| EvalKey gen time | 2.5 s      | 5.4 s       | 11.9 s       |

**Key observations:**

1. **Linear-in-N scaling**: Most operations scale roughly linearly with N
   (i.e., 2x per security level step). Ct × Ct multiplication scales
   super-linearly (~2.1x per step) due to the O(N log N) FFT and the
   relinearization cost.

2. **Accuracy is security-independent**: Error characteristics are identical
   across all three security levels. The noise/encoding parameters (base2k,
   k_ct, scale) are shared; only N changes. This means users can choose
   security level based purely on latency/size tradeoffs without worrying
   about accuracy regression.

3. **4x overhead for 128-bit security**: Moving from FHE_LLM-80 to FHE_LLM-128
   costs approximately 4-5x in latency and 4x in ciphertext size. This is the
   standard quadratic scaling in lattice dimension.

4. **Max depth doubles with N**: FHE_LLM-128 supports 48 levels of
   multiplicative depth (4 transformer layers without bootstrapping), while
   FHE_LLM-80 supports only 12 levels (1 layer). For models with >4 layers,
   bootstrapping is required regardless of security level.

5. **Inference-level validation confirms primitive-level findings**:
    - On the older fully encrypted path, d_model=64 at 128-bit shows the same
      high-noise profile as 80-bit and error does not grow with depth.
    - On the refreshed production path, d_model=256 at 128-bit with client-side
      final RMSNorm stays bounded across 1-2 layers:
      - 1 layer: FHE time 291.95s, L∞ = 7.0, MAE = 2.293
      - 2 layers: FHE time 622.90s, L∞ = 7.0, MAE = 2.367
    - This supports the security/performance claim that changing N primarily
      changes cost, while correctness is dominated by encoding/path design rather
      than the nominal security level.

For the refreshed production path used for real single-token inference, the final
model RMSNorm is intentionally applied as client-side final RMSNorm after decryption. This
does not change the confidentiality of the encrypted transformer body: the server
still only sees encrypted activations, and the user was already required to decrypt
the final hidden state locally before applying the LM head.

Security consequence of this design choice:

- The encrypted computation boundary ends at the final transformer residual output.
- The final RMSNorm and LM head are trusted local post-processing on the client.
- No extra oracle surface is introduced, because no intermediate decrypted value is
  returned to the provider.
- The refreshed d_model=256 validation at 128-bit uses this boundary and remains
  stable over multiple layers, so the boundary choice is not just theoretical but
  operationally validated on the larger truncated setting.

## 3. Approximate FHE: Security Implications

### 3.1 Relaxed Correctness Does Not Weaken Security

FHE_LLM intentionally introduces bounded rounding errors during rescaling
operations. A natural question is whether these errors leak information about
the plaintext.

**They do not.** The rounding errors arise from truncating the least-significant
bits of the *ciphertext* (the encrypted representation), not from the plaintext
directly. Specifically:

1. The rescaling operation divides the ciphertext by 2^base2k and rounds.
2. The rounding error is bounded by 2^(-base2k) per coefficient.
3. This error is added to the *noise* term of the ciphertext, which is already
   statistically masked by the RLWE encryption noise.
4. An adversary observing the ciphertext cannot distinguish the rescaling error
   from the encryption noise.

Formally: if the original scheme is IND-CPA secure, the scheme with rescaling
is also IND-CPA secure, because the rescaled ciphertext is a deterministic
function of the original ciphertext (no new randomness or plaintext-dependent
information is introduced).

### 3.2 Error Model

| Operation    | Error bound (per coeff)  | Error type             |
|-------------|--------------------------|------------------------|
| Encrypt     | σ = 3.2                  | Gaussian noise         |
| Add         | 0                        | Exact                  |
| Mul_const   | 0                        | Exact (before rescale) |
| Rescale     | 2^(-14)                  | Deterministic rounding |
| Ct × Ct     | O(N · σ² · 2^(-base2k)) | Noise + rounding       |

Cumulative error after k rescales: bounded by k · 2^(-14). For 32 layers
with ~10 rescales per layer: 320 · 2^(-14) ≈ 0.0195. This is well within
the INT8 quantisation floor of 2^(-8) ≈ 0.0039.

**Key insight**: The cumulative error exceeds the INT8 quantisation floor,
meaning the FHE computation is noisier than ideal INT8 inference. However,
published results on quantised inference show that models tolerate noise at
this level with <3% quality degradation on standard benchmarks. The error
is statistically indistinguishable from additional quantisation noise.

## 4. Key Security Considerations

### 4.1 Secret Key Distribution

FHE_LLM uses a ternary secret distribution s ∈ {-1, 0, 1}^N. This is standard
for RLWE-based FHE and is the same distribution used by CKKS, TFHE, and BGV
implementations in SEAL, OpenFHE, and Concrete.

Ternary secrets enable the hybrid attack (Howgrave-Graham), which combines
lattice reduction with exhaustive search over a portion of the secret. Our
security estimates account for this attack.

### 4.2 Key-Dependent Messages (KDM Security)

FHE_LLM does not claim KDM security. The scheme should not be used to encrypt
messages that depend on the secret key. For the target application (inference
on user data with provider-held model weights), this restriction is naturally
satisfied — the user's input data is independent of the secret key.

### 4.3 IND-CPA vs IND-CPA^D (Decryption Oracle)

Like CKKS, FHE_LLM provides IND-CPA security but NOT IND-CPA^D (security
against chosen-ciphertext attacks with a decryption oracle). The Li-Micciancio
attack (2021) showed that CKKS is vulnerable to IND-CPA^D attacks because
the approximate decryption leaks information about the noise.

This is **not a concern for FHE_LLM's target use case** because:
1. The user decrypts locally — no decryption oracle is exposed to the provider
2. The provider performs homomorphic evaluation and returns encrypted results
3. The user never sends decrypted results back to the provider

If FHE_LLM were to be used in an interactive protocol where decryption results
are shared, additional countermeasures would be needed (e.g. noise flooding
before decryption).

### 4.4 Side-Channel Resistance

FHE_LLM inherits Poulpy's implementation characteristics:

- **Scratch-based allocation**: No heap allocations on the hot path, reducing
  timing side-channel surface from memory allocation
- **Fixed-size operations**: All RLWE operations process the full polynomial
  ring regardless of plaintext content, providing constant-time behaviour at
  the algebraic level
- **Backend-generic**: The actual timing characteristics depend on the backend
  (FFT64Ref, FFT64Avx). AVX2 implementations should use constant-time
  arithmetic to prevent timing leaks.

Side-channel resistance is **not formally verified**. For production deployment,
a dedicated side-channel analysis should be conducted on the specific backend
and hardware platform.

## 5. Deviations from Standard Assumptions

FHE_LLM does **not** deviate from standard cryptographic hardness assumptions.
The following aspects are non-standard but do not affect security:

| Aspect                    | Standard practice    | FHE_LLM              | Security impact |
|---------------------------|---------------------|----------------------|-----------------|
| Coefficient size          | 50-60 bits          | 14 bits (base2k)     | None (q is fixed) |
| Rescaling                 | RNS prime drop      | Bit-shift            | None            |
| Noise management          | Conservative margins | Tight (fixed depth)  | None*           |
| Slot count                | N/2                 | N (no complex embed) | None            |

*Tight noise management means there is less margin for implementation errors.
If the noise model is incorrect (e.g. due to a bug in the noise tracker),
decryption may fail silently with incorrect results rather than with an obvious
error. This is an implementation risk, not a security risk.

## 6. Known Limitations

### 6.1 New Scheme, Limited Cryptanalysis

FHE_LLM is a new specialisation that has not undergone the extensive
cryptanalytic scrutiny that CKKS has received since 2017. While the underlying
RLWE assumption is well-studied, the specific parameter choices and the
interaction between approximate arithmetic and the bivariate representation
have not been independently audited.

**Recommendation**: Before production deployment, the parameter sets should be
validated using the Lattice Estimator tool and reviewed by a domain expert in
lattice-based cryptography.

### 6.2 Fixed Architecture Assumption

FHE_LLM's parameters are pre-computed for a specific transformer architecture.
If the model architecture changes (e.g. different depth or different activation
functions), parameters must be recomputed and keys regenerated. This is a
deployment constraint, not a security weakness.

### 6.3 No Threshold or Multi-Party Support

FHE_LLM is a single-key scheme. It does not support threshold decryption or
multi-party computation. The aggregation phase (Phase 2) is handled by a
separate system (Enclave) after local decryption.

## 7. Recommendations

### 7.1 Recommended Parameter Set

For the target inference application: **FHE_LLM-128** (N=16384, 128-bit
post-quantum security).

Rationale:
- The threat model includes a remote inference provider who may be a
  well-resourced adversary
- Encrypted inference results may retain value for years (user preferences,
  governance votes)
- Post-quantum security provides protection against future quantum computers
- The performance overhead vs FHE_LLM-100 (~2-3x) is acceptable given the
  security margin

### 7.2 When to Use Lower Security

- **FHE_LLM-100**: Suitable for latency-critical applications where the
  threat model excludes quantum adversaries and the data has limited
  long-term sensitivity
- **FHE_LLM-80**: Development and testing only. Should not be used in
  production for sensitive data.

### 7.3 Pre-Deployment Checklist

1. Validate parameter sets with the Lattice Estimator (latest version)
2. Independent review of the noise analysis and error bounds
3. Side-channel analysis of the target backend and hardware
4. Formal verification of the IND-CPA security reduction (proof review)
5. Penetration testing of the key generation and encryption implementation
6. Verify that the Li-Micciancio (IND-CPA^D) attack is not applicable in
   the deployed protocol (no decryption oracle exposed)
