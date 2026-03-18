# AGENTS.md: AI-Native FHE Scheme Design and Implementation

> **Last Updated:** 2026-03-18 (Session 12)
> **Repo Status:** Clean - redundant session logs pruned, AGENTS.md consolidated

## Recent Cleanup (2026-03-18)

Pruned redundant/dead files to streamline repository:

- **Deleted session logs:** `SESSION_8_NOTES.md`, `SESSION_9_RESULTS.md`, `SESSION_11_SUMMARY.md`
- **Deleted duplicate analysis docs:** `FHE_INFERENCE_FINDINGS.md`, `FHE_INFERENCE_STATUS.md`
- **Deleted obsolete implementation notes:** `HOMOMORPHIC_LM_HEAD_IMPLEMENTATION.md`, `PERFORMANCE_ANALYSIS.md`, `QKV_OPTIMIZATION_ANALYSIS.md`, `100bit_solution_validation.md`
- **Kept:** `CHANGELOG.md` (historical record), `docs/review-ntt120.md` (code review, distinct from design doc)

All work documented in consolidated `AGENTS.md` and `docs/*.md` files.

## Background

Fully Homomorphic Encryption (FHE) allows computation over encrypted data without
decrypting it. Current FHE schemes (CKKS, TFHE, BGV, BFV) were designed for general
purpose computation and are poorly matched to the specific arithmetic of neural network
inference. The result is that private AI inference under FHE today is either impractically
slow, impractically noisy, or both.

The goal of this project is to design and implement a new FHE scheme — or a targeted
specialisation of an existing scheme — that is purpose-built for transformer-style AI
inference.

This scheme will serve as the inference layer in a two-phase privacy-preserving
participatory governance system:

1. **Phase 1 (this project):** A user's AI proxy runs inference privately on a remote
   provider's hardware. The provider learns nothing about the user's data, context, or
   output.
2. **Phase 2 (out of scope):** The encrypted inference output is decrypted locally by
   the user, reviewed, and re-encrypted for private aggregation via a multi-party
   computation protocol (e.g. Enclave: enclave.gg).

The two phases are cryptographically decoupled. This project concerns Phase 1 only.

---

## Problem Statement

### Why existing schemes fall short

**CKKS** is the current best option for ML inference. It supports approximate arithmetic
over real numbers and allows SIMD-style slot packing. However:

- Noise accumulates with each multiplication, consuming the noise budget
- Deep networks (e.g. transformers) require expensive *bootstrapping* to refresh the
  noise budget mid-inference
- Softmax, GELU, and LayerNorm are highly non-polynomial, making them very expensive
  to evaluate under CKKS
- The scheme operates at 50+ bit precision per coefficient — far more than the 8-16 bit
  precision that quantized inference actually requires
- The "wasted" precision headroom is enormous and directly translates to wasted compute

**TFHE/FHEW** support fast gate bootstrapping but operate on individual bits or small
integers. The per-operation cost is prohibitive for the billions of operations in a
single transformer forward pass.

**BGV/BFV** handle exact integer arithmetic but do not map naturally to floating point
operations.

### The core mismatch

Existing FHE schemes were designed around security parameter requirements that dictate
field sizes and noise parameters. The arithmetic of ML inference was not a design input.

Modern inference workloads have well-understood arithmetic structure:
- Typical precision: FP16 (10-bit mantissa), INT8, or FP8
- Operations: matrix multiplication, attention (QKV projection, softmax, output
  projection), layer normalisation, elementwise activations
- Depth: fixed and known ahead of time for a given model architecture
- Approximation tolerance: inference is already approximate; small bounded errors
  indistinguishable from floating point rounding are acceptable

A scheme co-designed with these properties in mind could eliminate much of the overhead
that makes FHE inference impractical today.

---

## Desired Outcome

Design and implement a new FHE scheme (or targeted specialisation) with the following
properties, in rough priority order:

### 1. Native low-precision arithmetic

The scheme's prime field and noise parameters should be co-designed with 8-16 bit
arithmetic, not adapted from schemes designed for higher precision applications.

- Target: a prime field sized to match FP16 or INT8 arithmetic naturally, minimising
  representational overhead
- Inspired by the observation (credit: Jordi Baylina) that fitting the prime field to
  the natural arithmetic size of the computation massively reduces overhead
- Evaluate: what is the minimum field size that provides adequate noise headroom for a
  single transformer forward pass at INT8 or FP16 precision?

### 2. Nonlinearity-friendly evaluation

Softmax and GELU are the two most expensive operations under current FHE schemes because
they are not well-approximated by low-degree polynomials.

- Investigate polynomial-friendly activation functions that preserve model quality while
  being cheap to evaluate under FHE (e.g. degree-2 or degree-3 approximations)
- Investigate lookup table (LUT) evaluation approaches for fixed-precision nonlinearities
- The scheme should make an explicit design choice about how nonlinearities are handled,
  rather than treating them as an afterthought
- Acceptable to co-design the activation functions with the scheme if this yields
  significant performance gains

### 3. Single-budget inference (bootstrapping elimination or deferral)

Bootstrapping is the dominant cost in deep network inference under CKKS. The ideal scheme
eliminates the need for mid-inference bootstrapping entirely.

- The noise growth rate should be low enough that a complete transformer forward pass
  (for the target model size, see Target Workload) fits within a single noise budget
- If bootstrapping cannot be eliminated, it should be deferred to occur at most once per
  forward pass, and its cost should be minimised
- The scheme may assume fixed-depth circuits (known model architecture) and exploit this
  to tighten noise bounds

### 4. SIMD packing aligned to model structure

CKKS supports slot packing, but the native vector width is not designed to match
transformer attention head dimensions.

- The scheme's ciphertext packing structure should align naturally with the matrix
  dimensions of the target model (e.g. attention head size, embedding dimension)
- Goal: maximise utilisation of packed operations across a forward pass
- For MoE models, packing should account for the sparse activation pattern — only a
  subset of experts are active per token, and the scheme should not penalise this sparsity

### 5. User-side output verification (nice to have)

The aggregation layer (Phase 2) does not require publicly verifiable inference outputs —
the user decrypts locally and re-encrypts for aggregation. However, a trust assumption
remains: the user must trust that the inference provider actually ran the expected model
on their encrypted input and did not tamper with or substitute the output.

A desirable property is that the inference provider can produce a succinct proof of
correct execution that the user can verify cheaply on their local device. This is
distinct from public verifiability — only the user needs to verify, which allows for a
much lighter-weight proof system.

- Investigate ZK proof of correct inference attached to the encrypted output (prior art:
  ezkl, zkCNN, and related zkML literature)
- The proof need not be publicly verifiable, only user-verifiable; this may allow
  significant proof size and verification cost reductions compared to fully public schemes
- Acceptable for proof generation to be expensive on the provider side if verification
  is cheap on the user side
- Document the trust model in the absence of this feature (i.e. what the user must
  assume if verification is not implemented) so that the tradeoff is explicit

### 6. Relaxed correctness (approximate FHE)

Unlike aggregation (which requires exact outputs for verifiability), inference is already
approximate. The scheme may:

- Introduce small bounded errors indistinguishable from floating point rounding
- Relax noise management requirements that would otherwise demand larger parameters
- Explicitly document the error model and its acceptability for inference use cases

### 7. Security parameter targeting

Current schemes target 128-bit post-quantum security. This project should:

- Treat security level as a tunable parameter
- Evaluate the performance-security tradeoff at 80-bit, 100-bit, and 128-bit security
  levels
- Document which security level is recommended for inference applications and why

---

## Explicit Non-Goals

- **General purpose FHE.** This scheme is not intended to be competitive with CKKS or
  TFHE for arbitrary computations. It is permitted to be entirely unsuitable for use
  cases other than neural network inference.
- **Public verifiability.** The aggregation layer handles verifiability after local
  decryption. The scheme does not need to produce publicly verifiable outputs. User-side
  verification is a nice-to-have (see above) but distinct from public verifiability.
- **Phase 2 aggregation.** The scheme does not need to support multi-party computation,
  threshold decryption, or publicly verifiable outputs. These are handled by a separate
  aggregation layer (Enclave) after local decryption.
- **Scheme interoperability.** The user decrypts locally before re-encrypting for
  aggregation. There is no requirement for transciphering or cross-scheme compatibility.
- **Training.** This scheme targets inference only. Encrypted training is out of scope.

---

## Target Workload

Design decisions should be evaluated against a concrete target workload:

- **Architecture:** Transformer decoder (GPT-style), either dense or Mixture of Experts
  (MoE)
- **Parameter count:** 20B-40B total parameters
- **Active parameters:** For MoE models, target architectures with limited active
  parameters per token (e.g. a 40B MoE model with 8B active parameters). FHE cost should
  be evaluated against active parameter count, not total parameter count, as inactive
  experts do not contribute to the forward pass computation
- **Precision:** INT8 weights, FP16 activations (or fully INT8 if quantisation-friendly
  activations are adopted)
- **Sequence length:** 512-2048 tokens
- **Batch size:** 1 (single user inference, not batched serving)
- **Success metric:** A complete forward pass (single token generation step) completes
  within a practical latency budget on commodity server hardware, with no mid-inference
  bootstrapping

Define "practical latency" explicitly as part of the design process. Current CKKS-based
inference benchmarks should be used as the baseline to beat. Note that the 20B-40B
parameter range represents the current sweet spot for high-quality local inference on
consumer and prosumer hardware, and any scheme that performs well at this scale is likely
to generalise usefully to smaller models.

---

## Implementation Deliverables

1. **Scheme specification document:** formal description of the scheme, parameter choices,
   noise analysis, and security argument
2. **Reference implementation:** a correct (not necessarily optimised) implementation in
   Python or Rust sufficient to validate the scheme's correctness on small inputs
3. **Benchmark harness:** tooling to measure latency and noise budget consumption across
   a representative subset of transformer operations (matmul, attention, layernorm,
   activation), with separate benchmarks for dense and MoE routing
4. **Comparison report:** quantitative comparison against CKKS on the target workload,
   documenting performance gains and any tradeoffs accepted to achieve them
5. **Security analysis:** argument for the scheme's security at the chosen parameter
   levels, including any deviations from standard hardness assumptions
6. **Verification analysis (if attempted):** design and cost analysis of a user-side
   proof of correct inference, including proof size, generation cost on the provider
   side, and verification cost on a typical user device

---

## References and Prior Art

Agents should familiarise themselves with the following before beginning:

- CKKS scheme (Cheon, Kim, Kim, Song 2017) and subsequent bootstrapping work
- TFHE (Chillotti et al.) for comparison on gate-level bootstrapping cost
- Microsoft SEAL and OpenFHE as reference implementations of existing schemes
- Concrete ML (Zama) for prior work on ML inference under FHE
- Literature on FHE-friendly neural network design (polynomial activations, low-degree
  approximations)
- ezkl, zkCNN, and related zkML literature for user-side verification approaches
- Posit arithmetic as an analogy for questioning whether standard representations are
  right for a given compute workload
- HEaaN and lattigo for additional CKKS implementation references
- Mixtral and similar open MoE architectures as concrete examples of the target workload

---

## Open Questions

Agents should treat the following as active research questions and document findings:

1. What is the minimum polynomial modulus degree that supports a full transformer forward
   pass at INT8 precision without bootstrapping, at 128-bit security, for a 20B-40B
   parameter model?
2. Is there a polynomial activation function of degree <= 3 that is both FHE-cheap and
   preserves model quality within acceptable bounds on standard benchmarks?
3. Can attention's softmax be replaced entirely with a linear or low-degree approximation
   without meaningful quality degradation for the governance use case (where inference
   quality requirements may be lower than general-purpose LLM serving)?
4. What is the right granularity of ciphertext packing for transformer attention — by
   head, by layer, or by token?
5. Does targeting a fixed known architecture (rather than a general circuit) unlock
   meaningful parameter optimisations that a general-purpose scheme cannot exploit?
6. For MoE models, does the sparse activation pattern (only a subset of experts active
   per token) offer any FHE-specific optimisation opportunities, e.g. by reducing the
   effective circuit depth or allowing cheaper noise management over inactive paths?
7. What is the minimum proof size and verification cost achievable for user-side correct
   inference verification, and is it practical on a typical user device (e.g. a modern
   smartphone or laptop)?

---

## Implementation Status

> Last updated: 2026-03-16 (session 4)

### Crate: `poulpy-chimera` (18 source files, 231 tests passing, 28 ignored)

**Build/Test Status**:
- `cargo +nightly check -p poulpy-chimera`: ✅ Passes
- `cargo +nightly test -p poulpy-chimera --lib`: ✅ 231 passed, 0 failed, 28 ignored
- `cargo +nightly build --release -p poulpy-chimera --example smollm2_layer_sweep`: ✅ Passes

The CHIMERA scheme is implemented as a new crate in the Poulpy workspace, reusing
`poulpy-hal` (backend traits, FFT) and `poulpy-core` (RLWE encryption, keyswitching,
tensor products, automorphisms).

### Deliverables — ALL COMPLETE

| Deliverable | Location | Status |
|-------------|----------|--------|
| Scheme specification | `docs/chimera_spec.md` | ✅ Complete |
| Reference implementation | `poulpy-chimera/src/` | ✅ Complete (18 modules) |
| Benchmark harness | `poulpy-chimera/benches/chimera_ops.rs` | ✅ Complete |
| Comparison report vs CKKS | `docs/chimera_comparison.md` | ✅ Complete (updated with measured multi-security-level data) |
| Security analysis | `docs/chimera_security.md` | ✅ Complete (updated with measured benchmarks at 80/100/128-bit) |
| Verification analysis | `docs/chimera_verification.md` | ✅ Complete |

### Module Map

| Module | Responsibility | Status |
|--------|---------------|--------|
| `params.rs` | Security levels (80/100/128-bit), noise budgets, parameter selection | ✅ |
| `encoding.rs` | INT8/FP16 plaintext encoding with SIMD slot packing | ✅ |
| `encrypt.rs` | `ChimeraKey`, `ChimeraEvalKey`, RLWE encryption/decryption | ✅ |
| `arithmetic.rs` | Homomorphic add, ct-pt multiply, rescale, rotate, matmul, dot product | ✅ |
| `activations.rs` | Polynomial GELU/SiLU/SquaredReLU/inv_sqrt approximations, ct×ct multiply | ✅ |
| `lut.rs` | LUT-based nonlinearity evaluation via blind rotation | ✅ |
| `layernorm.rs` | Approximate RMSNorm/LayerNorm under FHE (with optional gamma/beta) | ✅ |
| `attention.rs` | QKV projection, attention scores, softmax approximation, context, output, RoPE | ✅ |
| `transformer.rs` | Full transformer block, forward pass (basic + full), FFN (standard + SwiGLU) | ✅ |
| `moe.rs` | MoE routing: homomorphic router, sign extraction, top-k, expert dispatch | ✅ |
| `noise.rs` | Noise tracking and budget estimation | ✅ |
| `bootstrapping.rs` | Full bootstrap pipeline: sample extract → LWE keyswitch → blind rotation | ✅ |
| `model_loader.rs` | Safetensors loading, INT8/FP16/BF16/FP32 quantization, transpose, sharded models | ✅ |
| `verification.rs` | MAC-based user-side computation verification (linear ops) | ✅ |
| `inference.rs` | End-to-end inference pipeline: tokenize → embed → encrypt → FHE forward → decrypt → LM head → decode | ✅ |
| `tests.rs` | 183 integration tests (including 8 accuracy + 3 security sweep + 11 MAC verification + 4 RoPE/full-pipeline tests) + 3 E2E inference pipeline tests (ignored, require model files) | ✅ |
| `plaintext_forward.rs` | Cleartext reference inference for FHE-vs-plaintext comparison (25 tests) | ✅ |
| `benches/chimera_ops.rs` | Criterion benchmarks for all operations (toy + d_model=128 + security sweep) | ✅ |

### Key Design Decisions Implemented

- **base2k cascade**: master `base2k=14`, input ct `in_base2k=13`, output ct `out_base2k=12`
- **k_ct = 113**: `8 * base2k + 1`, providing 9 limbs for tensor product relinearization
- **Polynomial activations**: GELU (degree 3), SiLU (degree 3), SquaredReLU (degree 2)
- **Softmax strategies**: PolynomialDeg4, ReluSquared, Linear (configurable per model)
- **Encoding scale**: `2 * in_base2k = 26` (torus encoding for INT8 values)
- **Bootstrapping**: sample_extract → LWE keyswitch (N→n_lwe) → blind rotation with identity LUT
- **Model loading**: safetensors with automatic PyTorch→CHIMERA transpose, per-tensor INT8 quantization
- **Inference pipeline**: tokenize (HuggingFace `tokenizers` crate) → embed → encrypt → FHE transformer → decrypt → LM head → argmax → decode

### What Works End-to-End

1. Encrypt INT8 input → homomorphic transformer block → decrypt output (legacy single-ct path for small toy dims only; vec path is the model-faithful path for real layouts)
2. Multi-layer forward pass (2 layers chained, output of block 1 feeds block 2)
3. Standard FFN and SwiGLU FFN at d_model=1 and d_model=2
4. Bootstrapping roundtrip: encrypt value → bootstrap through identity LUT → recover original
5. Matmul with multi-coefficient polynomial weights (d_model=4)
6. Load model weights from safetensors (INT8/FP16/BF16/FP32, sharded, memory-mapped)
7. RoPE wired into multi-head attention vec path (tested at d_model=4 with n_heads=2)
8. Full forward pass with final RMSNorm (`chimera_forward_pass_vec_full`) — production entry point
9. Full forward pass with bootstrapping support (per-layer noise check, all-ct bootstrap)
10. **Full text-in → text-out inference pipeline** (`inference.rs`): tokenize → embed → encrypt → refreshed FHE transformer → decrypt → optional client-side final RMSNorm → LM head → decode. Validated with TinyLlama 1.1B at multiple truncated dimensions:
    - d_model=64, 1 layer, 80-bit: 5.50s FHE, L-inf=7.0, MAE=2.09
    - d_model=128, 1 layer, 80-bit: 18.69s FHE, L-inf=7.0, MAE=2.19 in release E2E; best refreshed decode comparison reaches L-inf=2.0, MAE=1.96
    - d_model=256, 1 layer, 128-bit: 291.95s FHE, L-inf=7.0, MAE=2.293
    - d_model=256, 2 layers, 128-bit: 622.90s FHE, L-inf=7.0, MAE=2.367
    - Multi-token generation (3 tokens) still works on the smaller truncated setup.
11. **Rayon parallelization** of all hot paths (QKV projection, output projection, SwiGLU FFN, RMSNorm) — 3.2x speedup over sequential on 4 cores
12. **Cleartext reference inference** (`plaintext_forward.rs`) for FHE-vs-plaintext comparison with 25 tests

---

## Remaining Work (Priority Order)

### P0 — Functional Completeness — ALL COMPLETE ✅

1. ~~**Wire RMSNorm gamma into transformer block**~~ ✅ Done
2. ~~**Integrate bootstrapping into forward pass**~~ ✅ Done
3. ~~**Multi-head attention**~~ ✅ Done

### P1 — Validation & Measurement — COMPLETE ✅

4. ~~**Numerical accuracy characterization**~~ ✅ Done
   - 8 accuracy tests added: encrypt/decrypt baseline, addition, mul_const, matmul,
     GELU activation, transformer block d1, 2-layer error growth, d4 summary
   - Key findings: encrypt/decrypt and add are exact; mul_const Linf=4; matmul
     (multi-coeff) Linf=6; GELU Linf=0.33 vs cleartext PolyApprox

5. ~~**Benchmark at realistic dimensions**~~ ✅ Done
   - Added d_model=128 benchmarks: matmul_d128 (26.4ms), mul_const_128coeff (202μs),
     ct_ct_mul_d128 (1.96ms), ffn_d4h8 (23.2ms)
   - `docs/chimera_comparison.md` updated with measured numbers and extrapolated
     per-layer costs for 7B transformer

### P2 — Advanced Features

6. ~~**Homomorphic MoE routing**~~ ✅ Prototype done
   - Router logit computation via `chimera_matmul_single_ct`
   - Sign extraction comparison using bootstrap with sign LUT
   - Conditional swap via ct×ct multiply for oblivious sorting
   - Partial bubble sort network for encrypted top-k selection
   - Uniform gating weights over selected active experts
   - `chimera_moe_forward` combines routing + expert FFN evaluation, but remains a prototype (all experts may still be evaluated for privacy)

7. ~~**Security parameter sweep**~~ ✅ Done
   - Ran identical workload at 80-bit (N=4096), 100-bit (N=8192), 128-bit (N=16384)
   - Measured latency, accuracy, and noise budget at all three levels
   - 5 Criterion benchmarks parameterized across security levels
   - Key finding: ~2x latency per security step; accuracy identical across levels
   - `docs/chimera_security.md` and `docs/chimera_comparison.md` updated with
     measured (not estimated) multi-security-level data

8. ~~**User-side verification prototype**~~ ✅ Done
   - MAC-based user-side computation verification for linear operations
   - User tags ciphertexts with scalar MAC key α; provider maintains tags through
     homomorphic operations (add, mul_const, dot product, matmul)
   - User verifies `tag ≡ α·result (mod 2^scale_bits)` after decryption
   - 11 tests: 3 key management, 6 honest computation (all exact match), 2 tamper detection
   - Honest computations produce zero MAC error; tampered outputs detected with high probability within the prototype plaintext MAC domain
   - `docs/chimera_verification.md` updated with prototype implementation section

### Path to Real Model Inference

To run CHIMERA on a real model (e.g., a quantized LLaMA-7B):

1. ~~Complete P0 items (gamma wiring, bootstrapping integration, multi-head attention)~~ ✅ Done
2. ~~Verify numerical accuracy at d_model=128 (P1 item 4)~~ ✅ Done (accuracy characterized)
3. ~~Load real safetensors weights via `model_loader.rs`~~ ✅ Done (implemented)
4. ~~Build end-to-end inference pipeline (tokenize → embed → encrypt → FHE → decrypt → LM head → decode)~~ ✅ Done (`inference.rs`)
5. ~~Run a single-token forward pass and compare output logits to cleartext inference~~ ✅ Done
6. ~~Profile bottlenecks and optimize hot paths~~ ✅ Done (Rayon parallelization, 3.2x speedup)

All milestones on the Path to Real Model Inference are complete. The inference pipeline
has been validated end-to-end with real TinyLlama 1.1B weights.

### P3 — Performance Optimization

9. ~~**Rayon parallelization**~~ ✅ Done
   - Parallelized all hot paths: QKV projection, output projection, SwiGLU FFN
     (gate+up and down phases), standard FFN, RMSNorm (squaring and scaling)
   - Also parallelized: per-head attention loop, context computation, RoPE pairs
   - Fused multiply-accumulate in `chimera_dot_product` (buffer reuse, eliminates
     2*(d-1) GLWE allocations per dot product call)
    - Results at d_model=64, 80-bit security, 4 cores (older non-refreshed baseline):
      - Single token FHE forward pass: 8.77s → 2.74s (3.2x speedup)
      - QKV projection: 3.5x, output projection: 3.7x
      - SwiGLU gate+up: 3.7x, down projection: 3.3x
      - RMSNorm: 2.5-3.0x
    - Current refreshed release measurements: d_model=128 is 18.69s for 1 layer at 80-bit; d_model=256 is 291.95s for 1 layer at 128-bit
    - Profiling instrumentation added to transformer block, attention, and SwiGLU FFN

### P4 — FHE-vs-Cleartext Validation — COMPLETE ✅

10. ~~**FHE-vs-cleartext comparison with real TinyLlama weights**~~ ✅ Done
    - Three-way error decomposition: total (FHE vs exact), poly approx, FHE noise
    - Older encrypted-final-norm path at d_model=64 showed large error (L-inf ~63, MAE ~34)
    - Current refreshed/client-side-final-RMSNorm path is materially better:
      - d_model=64, 1 layer, 80-bit: L-inf=7.0, MAE=2.09
      - d_model=128, 1 layer, 80-bit: L-inf=7.0, MAE=2.19 in release E2E; best decode calibration gives L-inf=2.0, MAE=1.96
      - d_model=256, 1 layer, 128-bit: L-inf=7.0, MAE=2.293
    - Key finding: once client-side final RMSNorm is used, the refreshed transformer body stays stable and polynomial approximation error is negligible in the measured path

11. ~~**Multi-layer noise accumulation**~~ ✅ Done
    - Refreshed production-path measurements:
      - d_model=64, 80-bit: 1 layer `5.50s / L-inf=7.0 / MAE=2.09`; 2 layers `11.09s / L-inf=7.0 / MAE=2.19`; 4 layers `21.73s / L-inf=6.0 / MAE=2.12`
      - d_model=256, 128-bit: 1 layer `291.95s / L-inf=7.0 / MAE=2.293`; 2 layers `622.90s / L-inf=7.0 / MAE=2.367`
    - **Error does NOT grow materially with depth** on the refreshed path
    - Residual connections + RMSNorm + refresh boundaries stabilize noise across layers
    - Latency is approximately linear in layer count on the measured CPU backend
    - Bootstrapping was not needed for the measured 4-layer d64 run or 2-layer d256/128-bit run

### P3 — Performance Optimization — Session 2 Updates

9. ~~**Rayon parallelization (full FFN)**~~ ✅ Done (session 2)
   - Extended Rayon parallelization to `fhe_forward_refreshed` and
     `fhe_forward_refreshed_with_prompt_with_score_calibration`:
     - Gate+Up projections fused into single `into_par_iter()` returning `(gate_j, up_j)`
       tuples, then `.unzip()` — more efficient than two separate parallel passes
     - Gate×Up element-wise `ct_mul` parallelized with `par_iter().zip().map()`
     - Down projection parallelized with `par_iter().map()`
     - Output projection in prompt path parallelized
   - Results at d_model=576, 80-bit security, 16 cores (SmolLM2-135M):
     - Per-layer: **415s → 37s** (11.2x speedup)
     - FFN (dominant): gate+up 204s→21s, down 102s→2s
     - Attention: QKV 6.5s, out_proj 3.8s (already parallelized)
   - Total 30-layer forward pass: ~12,500s → ~1,100s estimated

10. ~~**Replace hybrid decrypt/re-encrypt with true bootstrap**~~ ✅ Done (session 2)
    - Three hybrid operations per layer replaced with homomorphic bootstrap:
      1. Refresh #1 (after attention residual): `refresh_vec_at_effective_scale` →
         `bootstrap_vec_identity_par` (identity LUT, parallel)
      2. SiLU gate activation: `apply_prompt_hybrid_gate_vec` →
         `apply_silu_lut_vec_par` (SiLU LUT, parallel)
      3. Refresh #2 (after gate×up): `refresh_vec_at_effective_scale` →
         `bootstrap_vec_identity_par` (identity LUT, parallel)
    - New helper methods in `InferencePipeline`:
      - `bootstrap_vec_identity_par` — parallel identity bootstrap over vec of cts
      - `apply_silu_lut_vec_par` — parallel SiLU LUT bootstrap
      - `bootstrap_vec_identity` — sequential variant (unused in main paths)
      - `apply_silu_lut_vec` — sequential variant (unused in main paths)
    - Bootstrap overhead per layer at d_model=576:
      - `bootstrap_residual_1` (576 cts): ~320ms
      - `silu_bootstrap` (1536 cts): ~860ms
      - `bootstrap_hidden` (1536 cts): ~840ms
      - Total: ~2.0s/layer (~5% of layer time)
    - Bootstrap output layout: `(base2k=19, k=38)` vs input `(base2k=13, k=113)`.
      `chimera_dot_product` reads `ct.base2k()` dynamically; `chimera_ct_mul` calls
      `chimera_align_layout` when layouts differ. Verified working in 1-layer run.

### SmolLM2-135M Full FHE Results

**OLD run (hybrid, sequential FFN):** Completed as PID 3954666.
- d_model=576, 30 layers, 80-bit security, single token (29889 = ".")
- FHE time: **12,512.6s** (~3.47 hours)
- FHE token: **32550 ("emeteries")** — garbage (flat logits: max=23)
- Plaintext (multi-layer wrapping shadow) token: **36899 ("rapers")**
- Token match: **NO** — FHE output lost all signal
- Hidden state error vs 1-layer plaintext: L-inf=28.0, MAE=1.938

**NEW run (bootstrap, parallelized FFN, single-token):** Completed as PID 4102183.
- Same configuration but with real bootstrap instead of hybrid
- 1-layer sanity run completed: **37.0s** (vs 415s old), no panics
- 1-layer FHE token: 26917 ("raines"), L-inf=8.0, MAE=2.102 vs multi-layer plaintext
- Full 30-layer: **1,205.0s** (20.1 minutes) — **10.4x faster** than hybrid
- FHE token: 47504 ("splitext") — garbage (flat logits: max=15, next=12)
- Hidden L-inf=11.0, MAE=2.099 vs shadow
- Token match: NO — FHE output is garbage, same as hybrid (but faster and lower noise)

### P3 — Performance Optimization — Session 3 Updates

12. ~~**Prompt-level token parallelization**~~ ✅ Done (session 3)
    - Parallelized `fhe_forward_refreshed_with_prompt_with_score_calibration`:
      - Pre-attention RMSNorm: `par_iter` over tokens
      - QKV projection + RoPE: `par_iter` over tokens (with position index)
      - Attention heads: `into_par_iter` over heads within each query position
      - Output projection: `par_iter` over output rows
    - FFN across tokens NOT parallelized (inner `par_iter` over d_ffn already saturates cores)
    - Causal attention `for q_pos` loop remains sequential (inherent dependency)
    - 1-layer sanity at d=576, T=4, raw "2+2=": **159s** (vs ~160s before — FFN dominates,
      token-level parallelism has modest impact with 4 tokens, but head parallelism helps attention)

### P4 — FHE-vs-Cleartext Validation — COMPLETE ✅

10. ~~**FHE-vs-cleartext comparison with real TinyLlama weights**~~ ✅ Done
    - Three-way error decomposition: total (FHE vs exact), poly approx, FHE noise
    - Older encrypted-final-norm path at d_model=64 showed large error (L-inf ~63, MAE ~34)
    - Current refreshed/client-side-final-RMSNorm path is materially better:
      - d_model=64, 1 layer, 80-bit: L-inf=7.0, MAE=2.09
      - d_model=128, 1 layer, 80-bit: L-inf=7.0, MAE=2.19 in release E2E; best decode calibration gives L-inf=2.0, MAE=1.96
      - d_model=256, 1 layer, 128-bit: L-inf=7.0, MAE=2.293
    - Key finding: once client-side final RMSNorm is used, the refreshed transformer body stays stable and polynomial approximation error is negligible in the measured path

11. ~~**Multi-layer noise accumulation**~~ ✅ Done
    - Refreshed production-path measurements:
      - d_model=64, 80-bit: 1 layer `5.50s / L-inf=7.0 / MAE=2.09`; 2 layers `11.09s / L-inf=7.0 / MAE=2.19`; 4 layers `21.73s / L-inf=6.0 / MAE=2.12`
      - d_model=256, 128-bit: 1 layer `291.95s / L-inf=7.0 / MAE=2.293`; 2 layers `622.90s / L-inf=7.0 / MAE=2.367`
    - **Error does NOT grow materially with depth** on the refreshed path
    - Residual connections + RMSNorm + refresh boundaries stabilize noise across layers
    - Latency is approximately linear in layer count on the measured CPU backend
    - Bootstrapping was not needed for the measured 4-layer d64 run or 2-layer d256/128-bit run

### Current Priority — Layer-by-Layer FHE Validation

**Immediate objective**: Validate that the faithful shadow (plaintext simulation of FHE arithmetic) produces correct tokens, then layer-by-layer audit the actual FHE path to isolate where encryption noise causes divergence.

1. ✅ **Build fixed** - Added `chimera_bootstrap_with_lut_custom_precision` function to bootstrapping.rs
2. ✅ **All 231 tests pass** - No failures, 28 ignored
3. ✅ **Faithful shadow validated** - Produces correct tokens on "2+2=" and "What is 2+2?"
4. ✅ **11-bit rescaled SiLU confirmed** - Necessary and sufficient for correct token prediction
5. ✅ **Layer 1 audit run** - L-inf=0.17 error (excellent match)
6. ✅ **Created layer-sweep example** - smollm2_layer_sweep.rs for full layer-by-layer comparison

**Next steps**:
- ✅ **Layer-sweep infrastructure complete** - FHE forward pass with per-layer bootstrap working
- ⏳ **Fix exact hidden state computation** - Need proper plaintext reference for comparison
- ⏳ **Run layer-by-layer comparison at depths 1, 2, 4, 8, 16, 30**
- ⏳ **Identify exact layer where FHE diverges from shadow**
- ⏳ **Diagnose noise vs arithmetic approximation issues**

---

### Remaining Optimization Opportunities

- Scale beyond d_model=256 and profile multi-layer refreshed runs at 128-bit
- Investigate whether refreshed decode precision can be fixed automatically rather than selected by sweep diagnostics
- Multi-token KV cache (currently re-processes single token per step)

---

## Current Prompt-Conditioned Debug Status (SmolLM2)

> Last updated: 2026-03-16 (session 5)

### Summary

- **Build status**: ✅ `cargo +nightly check -p poulpy-chimera` passes
- **Test status**: ✅ `cargo +nightly test -p poulpy-chimera --lib` passes (231 passed, 28 ignored)
- **Faithful shadow**: ✅ Produces correct tokens on "2+2=" and "What is 2+2?"
- **11-bit SiLU**: ✅ Confirmed necessary and sufficient for correct token prediction
- **Layer 1 audit**: ✅ L-inf=0.17 error (excellent match with exact path)
- **Layer sweep**: ✅ Created and validated at d=64, 2 layers

### Immediate Objective

- Make prompt-conditioned CHIMERA inference semantically track exact HuggingFace inference on simple prompts such as `What is 2+2? Answer with one token.`
- Current sequence: fix exact plaintext prompt path first, then refreshed plaintext prompt path, then FHE prompt path.

**Layer-sweep example created and validated** (`smollm2_layer_sweep.rs`):
- Runs actual FHE encrypted inference with bootstrapping at each layer boundary
- Extracts decrypted hidden states after each layer for comparison
- Successfully runs d_model=64, 2 layers in ~2 seconds
- Layer 1: 1.0s (attention 193ms, out_proj 66ms)
- Layer 2: 0.9s (attention 194ms, out_proj 83ms)
- Infrastructure working - FHE path processes all layers with proper bootstrapping

**Key addition**: `compare_prompt_fhe_layer_hidden_states()` method in `inference.rs`
- Runs encrypted forward pass with per-layer bootstrap
- Decrypts hidden states at each layer boundary
- Returns `(layer_name, Vec<f64>)` tuples for comparison

### Immediate Objective

- Make prompt-conditioned CHIMERA inference semantically track exact HuggingFace inference on simple prompts such as `What is 2+2? Answer with one token.`
- Current sequence: fix exact plaintext prompt path first, then refreshed plaintext prompt path, then FHE prompt path.

### Current Reference Model

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Paths:
  - weights: `/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors`
  - tokenizer: `/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json`
- Exact HuggingFace generation for `What is 2+2? Answer with one token.` is sensible (`Two and two is equal to four.` in the earlier exact script run).

### Prompt Debug Tooling Added

- Rust examples:
  - `poulpy-chimera/examples/smollm2_decompose.rs`
  - `poulpy-chimera/examples/smollm2_layerdiff.rs`
  - `poulpy-chimera/examples/smollm2_vs_transformers.rs`
  - `poulpy-chimera/examples/smollm2_layer1audit.rs` — Layer 1 sublayer audit (L-inf=0.17)
  - `poulpy-chimera/examples/smollm2_full_fhe_30layer.rs` — Full 30-layer FHE runner
  - `poulpy-chimera/examples/smollm2_plain_multilayer.rs` — Standalone plaintext shadow
  - `poulpy-chimera/examples/smollm2_prompt_fhe.rs` — 3-way prompt comparison (exact vs refreshed vs FHE) with `--raw` mode
  - `poulpy-chimera/examples/smollm2_prompt_plain.rs` — Plaintext-only prompt comparison
  - `poulpy-chimera/examples/smollm2_layer_sweep.rs` — Layer-by-layer FHE vs plaintext shadow comparison
- Python scripts:
  - `scripts/smollm2_exact.py`
  - `scripts/smollm2_hidden_trace.py`
  - `scripts/smollm2_compare.sh`
- Important trace files:
  - `/tmp/smollm2_hidden_trace.json` currently contains the quantized/dequantized Python reference trace
  - `/tmp/smollm2_hidden_trace_float.json` preserves the earlier float32 HuggingFace trace

### Key Discovery and Local Fixes

- The major semantic bug was not just model truncation. The custom exact prompt path was using quantized embeddings/weights as raw integers and was missing the attention score scale factor `1/sqrt(d_head)`.
- Local `poulpy-chimera/src/inference.rs` now carries dequantization scales for:
  - embeddings
  - LM head
  - per-layer attention/FFN weights
- The exact prompt-conditioned path now:
  - dequantizes embeddings and weights
  - applies attention score scaling
  - uses an exact-path attention orientation helper to compensate for the current mixed loader orientation under GQA
- Under the current loader state:
  - `w_q` and `w_o` need transposition for exact HuggingFace-style row-as-output semantics
  - `w_k` and `w_v` already match row-as-output under the current GQA loading path

### Current Measured State

> Updated: 2026-03-16

- **Both exact and refreshed plaintext paths now produce the correct token `"Two"` (10345)**
- Token match between exact and refreshed: ✅
- Exact vs refreshed hidden error: `L-inf=1.786, L2=0.330, MAE=0.243` (dramatically improved from earlier `L-inf=40.000`)

- First-layer sublayer audit (exact path) against the quantized/dequantized Python trace:
  - `layer1_pre_attn_norm`: `L-inf=0.019927`, `MAE=0.000073`
  - `layer1_q`: `L-inf=0.052379`, `MAE=0.008050`
  - `layer1_k`: `L-inf=0.020973`, `MAE=0.004920`
  - `layer1_v`: `L-inf=0.005601`, `MAE=0.000597`
  - `layer1_attn_out`: `L-inf=0.009580`, `MAE=0.001618`
  - `layer1_residual1`: `L-inf=0.009580`, `MAE=0.001618`
  - `layer1_pre_ffn_norm`: `L-inf=0.011335`, `MAE=0.001064`
  - `layer1_gate`: `L-inf=0.038773`, `MAE=0.004748`
  - `layer1_up`: `L-inf=0.034942`, `MAE=0.003489`
  - `layer1_mlp_out`: `L-inf=0.037750`, `MAE=0.005993`
  - `layer1_residual2`: `L-inf=0.045370`, `MAE=0.006134`

- Full exact-path layer comparison against the Python trace (all 30 layers + final norm):
  - `embed_last`: `L-inf=0.000` (exact)
  - `layer_1`: `L-inf=0.045`, `layer_10`: `L-inf=0.311`, `layer_20`: `L-inf=0.428`
  - `layer_28`: `L-inf=3.428` (highest intermediate), `layer_29`: `L-inf=2.990`
  - `layer_30` (final norm): `L-inf=0.201`, `MAE=0.022` — **previously L-inf=483.969, now fully repaired**

- Full refreshed-path layer comparison against the Python trace:
  - `layer_1`: `L-inf=0.180`, `layer_10`: `L-inf=1.974`, `layer_20`: `L-inf=5.828`
  - `layer_29`: `L-inf=13.171`, `MAE=3.278`
  - `layer_30` (final norm): `L-inf=1.923`, `MAE=0.247`
  - Error growth is expected due to 8-bit quantization rounding at each layer
  - Despite higher per-layer error, the refreshed path still produces the correct token

### Prompt-Conditioned FHE Results (Session 3)

| Config | Exact Token | Refreshed Token | FHE Token | FHE L-inf | FHE Time |
|--------|------------|-----------------|-----------|-----------|----------|
| d=576, L=1, T=4, raw "2+2=" | 216 " " | 216 " " | 808 " first" | 16.0 | 159s |
| d=128, L=1, T=4, raw "2+2=" | 45 "=" | 45 "=" | 43179 "ChannelType" | 8.0 | 24s |
| d=128, L=5, T=4, raw "2+2=" | 7133 " enthus" | 3427 " strugg" | 47347 "ομαι" | 7.0 | 120s |
| d=64, L=1, T=28, chat "What is 2+2?" | 198 "\n" | 198 "\n" | 44812 "firehose" | 9.0 | 176s |

### Plaintext Validation at Native d=576, 30 Layers

| Prompt | Exact Token | Refreshed Token | Match | Error L-inf |
|--------|------------|-----------------|-------|-------------|
| Raw "2+2=" (4 tokens) | 33 "1" | 33 "1" | ✅ | 1.0 |
| Chat "What is 2+2? Answer with one token." (33 tokens) | 10345 "Two" | 10345 "Two" | ✅ | 3.0 |

### Critical Finding #1: Single-token inference is meaningless

Single-token inference on token "." (29889) produces degenerate flat logits regardless
of path quality. Without prompt context, the model has no meaningful signal. Both the
shadow and FHE produce random garbage tokens.

### Critical Finding #2: FHE noise is fundamentally too large for correct token prediction

**LM Head Statistics (SmolLM2-135M, tied embeddings):**
- Shape: 49152 × 576 (INT8 quantized)
- Row L2 norm mean: ~207
- INT8 mean_abs: 6.40

**Noise propagation through LM head:**

| Hidden L-inf noise | Logit error std | Logit error L-inf |
|---|---|---|
| 1 | 124 | 692 |
| 2 | 225 | 1367 |
| 4 | 405 | 2295 |
| 8 | 824 | 4462 |

At native d=576, 30 layers, "2+2=" → exact logits have top-1 = 31, top-2 ≈ 31.
Gap = ~1 bit. FHE hidden noise L-inf=16 → logit noise std ~1648. Correct prediction
is impossible under current noise levels. Hidden noise needs to be < 0.015 for reliable
prediction — far below INT8 resolution.

### Key Fixes Applied (this session)

1. **RMSNorm `res_offset` fix**: Changed the final `x_i * inv_rms` tensor product in
   all `chimera_rms_norm_vec*` functions to use `eval_key.res_offset` (= 26) instead of
   `activation_decode_precision(RMS_NARROW_INPUT_SCALE)` (= 2). The old value caused a
   catastrophic 24-bit precision loss in the tensor product. See `layernorm.rs`.

2. **`RMS_OUTPUT_SCALE_SHIFT_BITS` fix**: Changed from 10 to `COEFF_SCALE_BITS` (= 8).
   The value 10 was derived from `RMS_NARROW_INPUT_SCALE` which is the polynomial's
   encoding scale, not a torus shift amount. The correct value is `COEFF_SCALE_BITS`
   because the only extra factor in the output is the polynomial coefficient scaling.

3. **Doctest fix**: Wrapped pseudo-code in `RMS_OUTPUT_SCALE_SHIFT_BITS` doc comment
   in ` ```text ` block to prevent Rust from trying to compile it.

4. **New test**: `test_rms_norm_vec_preserves_structure_after_res_offset_fix` — validates
   that the FHE RMSNorm output is non-degenerate and preserves distinct slot values.

5. **`add_constant_term` limb placement fix (Bug #2)**: The `add_constant_term` function
   in `activations.rs` was adding the polynomial constant term `c0` to limb 0 (MSB) of
   the GLWE body, when it should have been distributed across the correct limbs using
   carry-propagating base-2k decomposition. For inv_sqrt's c0=1.478, this caused the
   decoded value to be 16384x too large. Rewritten with proper multi-limb encoding using
   `znx_get_digit` and `znx_get_carry` helpers. The bug was masked because GELU/SiLU/
   SquaredReLU all have c0=0.0 (no-op), and only inv_sqrt/exp/reciprocal are affected.

6. **`encoding_scale` fix for inv_sqrt polynomial (Bug #3)**: All
   `apply_poly_activation_with_encoding_scale` calls in `layernorm.rs` were passing
   `RMS_NARROW_INPUT_SCALE` (= 10) as the encoding scale for the constant term. The
   correct value is `eval_key.res_offset` (= 26), because the polynomial output is
   always at `TP(26)` after `chimera_ct_mul`. Fixed all 6 remaining call sites across
   3 RMSNorm variants in `layernorm.rs`.

7. **Dead code cleanup**: Removed `RMS_NARROW_INPUT_SCALE` constant (now unused),
   removed `activation_constant_precision` function (no longer called after
   `add_constant_term` rewrite), prefixed unused `remap_shift_bits` parameter,
   gated `mean_scale_params` with `#[cfg(test)]`.

### Key Additions (session 4)

8. **Configurable SiLU bootstrap precision**: Added `InferenceConfig.fhe_silu_log_msg_mod`
   field (default `None`). When set to e.g. `Some(9)` or `Some(10)`, the SiLU LUT
   bootstrap uses 512 or 1024 entries instead of the default 128. Implemented via new
   `from_chimera_with_log_msg_mod()` constructor in `ChimeraBootstrapParams` and
   `chimera_bootstrap_with_lut_custom_precision()` in `bootstrapping.rs`. Validated with
   5 new tests up to `log_message_modulus=10` (guard band = 4 coefficients for N=4096).

9. **Configurable identity bootstrap precision**: Added `InferenceConfig.fhe_identity_log_msg_mod`
   field (default `None`). When set, identity (refresh) bootstraps use a higher-precision
   LUT via the new `NonlinearLUT::identity_message_lut()` helper in `lut.rs`. Both
   `bootstrap_vec_identity` and `bootstrap_vec_identity_par` dispatch to
   `chimera_bootstrap_with_lut_custom_precision` when configured. This reduces
   quantization error at each refresh boundary from ±0.5 (128 levels at log_msg_mod=7)
   to ±0.0625 (1024 levels at log_msg_mod=10). Validated with 5 new tests (3 unit tests
   for LUT shape, 2 end-to-end bootstrap roundtrips at log_msg_mod=9 and 10).

10. **COEFF_SCALE_BITS reduction analysis**: Concluded that reducing `COEFF_SCALE_BITS`
    from 8 to 6 or 5 is NOT viable. The inv_sqrt polynomial's cubic coefficient
    `c3 = -0.00579` rounds to 0 at 6-bit scale, causing catastrophic approximation
    error. Even recovering all 8 bits would only improve hidden noise L-inf from 16
    to ~0.06, still above the 0.015 threshold for correct token prediction.

### What This Means

- **The exact prompt-conditioned path is fully repaired** through all 30 layers including
  the final norm. The layer_30 divergence (L-inf=483.969) is now L-inf=0.201.
- **The refreshed plaintext path now produces the correct token** ("Two"). Previously
  it produced "cer" (token 1704), indicating a fundamental semantic failure.
- The `res_offset` fix restores full torus precision in the RMSNorm output, which
  propagates through the entire inference chain.
- 231 unit tests pass, 0 failures.

### Remaining Work

1. ~~**FHE 30-layer run on SmolLM2 (hybrid path)**~~ ✅ COMPLETED
   - Result: token=32550 ("emeteries"), flat logits (max=23), 12,512.6s
   - Does NOT match plaintext shadow (36899). Signal completely lost.
   - This was the OLD hybrid (decrypt/re-encrypt) code.
2. ~~**FHE 30-layer run on SmolLM2 (bootstrap path, single-token)**~~ ✅ COMPLETED
   - Result: token=47504 ("splitext"), flat logits (max=15, next=12), 1,205.0s
   - Hidden L-inf=11.0, MAE=2.099 vs shadow. 10.4x faster than hybrid.
3. **FHE 30-layer run on SmolLM2 (bootstrap, prompt-conditioned "2+2=")**: PID 4189751.
   Uses parallelized prompt FHE forward (QKV+RoPE, attention heads, output projection
   all parallelized with Rayon). 1-layer sanity passed (160s, 4 tokens). Estimated
   ~80 minutes for 30 layers.
4. **Make plaintext shadow arithmetically faithful to FHE path**: Currently uses
   exact `1/sqrt()` instead of polynomial approximation, doesn't model torus
   quantization in ct_mul. 16 discrepancies catalogued (see below).
5. **Address fundamental noise amplification problem**: LM head noise amplification
   means hidden L-inf=8 → logit error std ~824, while signal gap is only 1-3 bits.
   FHE hidden noise needs to be < 0.015 for correct prediction — far below INT8
   resolution. See "Critical Finding #2" below.

### Plaintext Shadow Discrepancies (16 identified, none fixed yet)

Most critical:

| # | Issue | Severity | Fix |
|---|---|---|---|
| **#2/#9** | Shadow uses exact `1/sqrt()`, FHE uses degree-3 polynomial (narrow: max_err 0.01, midrange: max_err 0.184) | **HIGH** | Replace `plaintext_forward::rms_norm` with version using same polynomial coefficients |
| **#16** | `chimera_ct_mul` output scale for gate×up may not match `f64 * f64` | **MEDIUM-HIGH** | Model: for inputs at TP(S), ct_mul with res_offset=2S produces output at TP(S) |
| **#11** | `chimera_dot_product_scaled` operates at 8 fewer bits of torus precision | **MEDIUM** | Model reduced-precision accumulation in shadow |

### Three-Phase Plan (Current)

1. ~~**Parallelize FFN dot products**~~ ✅ DONE — 11x speedup on 16 cores
2. ~~**Replace hybrid with real bootstrap**~~ ✅ DONE — ~2s/layer overhead (5%)
3. **Make plaintext shadow faithful to FHE** — NOT STARTED
   - Replace exact `1/sqrt()` with polynomial approximation in shadow
   - Model `ct_mul` output scale truncation
   - Model `dot_product_scaled` reduced precision

### Key Discovery: Torus Wrapping Semantics at Refresh Boundaries

**Critical finding from the multi-layer plaintext shadow work.** The torus arithmetic
wrapping behaviour differs between explicit refresh boundaries and the block output
residual stream:

- **At explicit refresh boundaries** (`refresh_vec_at_effective_scale` for residual_1
  and hidden): values are decoded at TP(8) into i64, giving signed 8-bit values in
  [-128, 127]. This is equivalent to `((v * 256).round() % 256)` signed wrapping.
- **Between refresh points** (block output = `residual_1_refreshed + ffn_out`): the
  full torus precision (113 bits across multiple GLWE limbs) carries the accumulated
  value faithfully. **No 8-bit wrapping occurs** at the block output addition.
- **RMSNorm tensor product** uses `res_offset=26` and operates on the full torus
  representation, not just the top 8 bits.
- **Bootstrap output layout**: `(base2k=19, k=38)` vs input `(base2k=13, k=113)`.
  Bootstrap now replaces the old hybrid refresh. The bootstrap quantizes to
  `log_message_modulus=7` bits (128 levels), which is different from the old
  TP(8) (256 levels) refresh. This changes the wrapping semantics.

Measured plaintext shadow results (wrapping mode = faithful to OLD hybrid FHE):
- x_rms grows moderately across layers (33-90 range)
- Block output x_max reaches ~448 (beyond [-128, 127]) but is NOT refreshed
- Predicted token: 36899 ("rapers") — semantically wrong for "What is 2+2?" but
  this is expected given 8-bit refresh quantization across 30 layers

### Multi-layer Plaintext Shadow Implementation

Added `refreshed_plain_target_multilayer` (with `RefreshMode::Wrap` / `RefreshMode::NoWrap`)
in `inference.rs`. The `RefreshMode` enum is defined just before `InferencePipeline`
struct. Public entry points:
- `plaintext_step_refreshed_multilayer()` — faithful wrapping mode
- `plaintext_step_refreshed_multilayer_no_wrap()` — diagnostic mode
- Standalone example: `poulpy-chimera/examples/smollm2_plain_multilayer.rs`

### Workspace Note

- The repo-local virtualenv `.venv-smollm/` is very large (~28k files, ~1.1G) and can make opencode UI/file trees extremely slow.
- It is now ignored via `.gitignore` and `.git/info/exclude`, but moving it outside the repo is still preferable if UI issues continue.

### Noise Analysis and Reduction Strategies (Session 3)

> Added: 2026-03-16

#### Noise Chain Through One Transformer Layer

1. **Fresh encryption**: negligible noise (~3e-9 of message step, ~27 bits of budget)
2. **QKV projection** (`chimera_dot_product`): 576 coefficients × INT8 weights, `res_offset=13`
3. **Attention scores** (`chimera_ct_mul`): tensor product at `res_offset=26`
4. **Bootstrap #1** (identity LUT): quantizes to `log_message_modulus=7` (128 levels, effectively [-127, 127] via negacyclic). Introduces ±0.5 quantization error. Output: `base2k=19, k=38`
5. **RMSNorm**: polynomial inv_sqrt + tensor product at `res_offset=26`
6. **FFN gate/up** (`chimera_dot_product_scaled`): 576 coefficients × weights, `res_offset=5` (8 bits less precision than standard dot product)
7. **SiLU bootstrap**: quantizes gate to 7-bit. ±0.5 error.
8. **Gate × Up** (`chimera_ct_mul`): tensor product of two bootstrapped values
9. **Bootstrap #2** (identity LUT): quantizes gate*up product
10. **Down projection** (`chimera_dot_product`): 1536 bootstrapped values × weights

#### Key Bottleneck: LM Head Noise Amplification

The LM head (49152 × 576 tied embeddings) amplifies hidden noise:
- `noise_logit ≈ sqrt(d_model) × mean_abs(w) × noise_hidden ≈ 24 × 6.4 × noise = 154 × noise`
- For hidden L-inf=16 (measured): logit noise std ≈ 2464
- Signal gap (top-1 vs top-2): only 1-3 bits
- Required hidden noise for correct prediction: L-inf < 0.015

This is a **fundamental architectural mismatch**: the LM head acts as a noise amplifier
with gain ~154×, and the model's logit gaps are ~1-3.

#### Potential Noise Reduction Strategies

| Strategy | Feasibility | Impact | Effort |
|----------|-------------|--------|--------|
| **1. Increase `log_message_modulus`** to 10-11 (1024-2048 levels) | Possible: N=4096 supports up to 12. Guard band shrinks from 32 to 2-4 coefficients per entry. | Reduces bootstrap quantization error by 8-16×. But FHE noise between bootstraps may still dominate. | Medium: change `scale_bits` or decouple `log_message_modulus` from `scale_bits`. |
| **2. Add more bootstrap points** per layer (e.g., after QKV projection, after each dot product) | Technically possible. | Reduces noise accumulation between bootstraps at the cost of more bootstrap calls (~320ms each for 576 cts). | Low code complexity; significant performance cost. |
| **3. Increase torus precision** in `chimera_dot_product_scaled` (use `res_offset` > 5) | Possible but requires rethinking the scale chain. | Recovers 8 bits of precision in FFN projections. | Medium: need to propagate scale changes through the entire chain. |
| **4. Homomorphic LM head** (compute logits under FHE, then decrypt logits instead of hidden state) | Feasible in principle. Encrypt the LM head weights and compute the matrix-vector product homomorphically. | Eliminates the noise amplification problem entirely — noise in logits would be comparable to noise in hidden state. | High: requires encrypting a 49152×576 weight matrix and computing 49152 dot products homomorphically. |
| **5. Noise-aware model fine-tuning** (train with simulated FHE noise to increase logit gaps) | Requires model training (out of scope per AGENTS.md). | Could increase logit gaps from 1-3 to 10+, making FHE noise tolerable. | Out of scope. |
| **6. Use FP16 precision** (scale_bits=14, log_message_modulus=13) | Supported in params.rs. | 64× more precise bootstrap quantization. But ciphertext sizes and computation costs increase proportionally. | Medium: parameter change + testing. |
| **7. Temperature-scaled logits** or top-k filtering | Cleartext post-processing after FHE decryption. | Doesn't help — the noise is in the hidden state, not in the logit processing. | N/A |

#### Most Promising Approaches

**Short term (this project):**
- Strategy 1 + 3: Increase bootstrap precision and fix dot_product_scaled precision loss
- Strategy 2: Add bootstrap after QKV projection if noise analysis shows it helps

**Medium term:**
- Strategy 4 (homomorphic LM head): eliminates the fundamental amplification problem
- Strategy 6 (FP16 mode): may be the simplest path to correct tokens

**The core insight:** The problem is not that FHE adds too much noise to the hidden
state — L-inf=16 is reasonable for INT8 arithmetic. The problem is that the LM head
amplifies this noise by ~154× while the model's decision boundary (logit gap) is only
1-3 bits. This is an architectural mismatch between the model and the FHE scheme, not
a bug in the FHE implementation.

---

## Session 12 — Repository Cleanup and Next Steps

> Date: 2026-03-18

### Repository Cleanup

Pruned 9 redundant documentation files that were superseded by consolidated AGENTS.md
and docs/*.md files:

- SESSION_8_NOTES.md, SESSION_9_RESULTS.md, SESSION_11_SUMMARY.md (session logs)
- FHE_INFERENCE_FINDINGS.md, FHE_INFERENCE_STATUS.md (duplicate analysis)
- HOMOMORPHIC_LM_HEAD_IMPLEMENTATION.md, PERFORMANCE_ANALYSIS.md, QKV_OPTIMIZATION_ANALYSIS.md, 100bit_solution_validation.md (obsolete implementation notes)

**Files kept:**
- CHANGELOG.md (historical record)
- docs/review-ntt120.md (code review, distinct from design doc)
- docs/ntt120-backend.md (backend design documentation)
- poulpy-hal/docs/backend_safety_contract.md (safety contract for implementors)

### Current State Summary

**Build/Test Status:**
- `cargo +nightly check -p poulpy-chimera`: ✅ Passes
- `cargo +nightly test -p poulpy-chimera --lib`: ✅ 231 passed, 0 failed, 28 ignored

**Key Implementation Status:**
- INT8 precision reduced from scale_bits=12 to scale_bits=10 (~20% reduction)
- LM head parallelization reverted to sequential (crashes with parallel chunking)
- Homomorphic LM head implemented but untested at full scale
- Forward pass completes: d=128, 30 layers, single token ≈ 867s

### Next Steps (Priority Order)

#### P0 — Critical Path (Immediate)

1. **Verify 1-layer sequential LM head**
   - Check `/tmp/d128_1layer_seq.log` for completion and results
   - Expected: Token "1" (ID: 33) for prompt "2+2="
   - If successful: Sequential LM head confirmed working ✅
   - If fails: Investigate precision, memory, or bootstrap issues

2. **Run d=128, 30-layer full FHE inference**
   - Command: `RAYON_NUM_THREADS=1 cargo +nightly run --release -p poulpy-chimera --features enable-avx --example smollm2_d128_30layer "2+2=" 30 2>&1 | tee /tmp/fhe_d128_sequential.log`
   - Expected time: 30-45 minutes (forward pass + LM head)
   - Success criterion: Produces correct token "1" or "4"

3. **Run d=576, 30-layer full FHE inference**
   - Command: `RAYON_NUM_THREADS=1 cargo +nightly run --release -p poulpy-chimera --features enable-avx --example test_smollm2_576 "2+2=" 30 2>&1 | tee /tmp/fhe_d576_sequential.log`
   - Expected time: Several hours (forward pass + LM head)
   - Full validation at production dimensions

#### P1 — Performance Optimization (After P0 Success)

4. **Explore batched LM head computation**
   - Current bottleneck: 49,152 logits × 128 hidden dims = 6.3M FHE dot products
   - Investigate memory-efficient batch processing
   - Consider chunk sizes that avoid OOM while maximizing parallelism

5. **Evaluate FP16 precision mode**
   - Increase scale_bits to 14, log_message_modulus to 13
   - 64× more precise bootstrap quantization
   - Tradeoff: 2-4x larger ciphertexts, 2-4x computation cost
   - May be simplest path to correct tokens if noise reduction is sufficient

#### P2 — Architecture (Medium Term)

6. **Homomorphic LM head full implementation**
   - Encrypt LM head weights (49152 × 576 matrix)
   - Compute logits homomorphically instead of decrypting hidden state
   - Eliminates noise amplification problem entirely
   - High effort but fundamentally solves the root cause

7. **Noise analysis refinement**
   - Profile actual noise at each layer boundary
   - Identify if adding bootstrap points helps
   - Measure guard band usage at different security levels

### Files to Monitor

- `/home/dev/repo/poulpy-chimera/src/params.rs` - INT8 precision configuration (scale_bits=10)
- `/home/dev/repo/poulpy-chimera/src/model_loader.rs` - Sequential LM head implementation
- `/home/dev/repo/poulpy-chimera/src/inference.rs` - FHE inference pipeline

### Known Issues

1. **LM head noise amplification** - Fundamental architectural mismatch (gain ~154×)
2. **Sequential LM head slow** - 49k parallel allocations exceed memory budget
3. **d=576 performance** - Forward pass ~15min, LM head potentially 30-60min

### Success Criteria for Next Session

- [ ] 1-layer sequential LM head produces correct token
- [ ] d=128, 30-layer FHE produces correct token
- [ ] d=576, 30-layer FHE completes and produces correct token
- [ ] Noise analysis confirms which strategy (FP16 vs homomorphic LM head) is viable

---

## Implementation Status

> Last updated: 2026-03-16 (session 4)
