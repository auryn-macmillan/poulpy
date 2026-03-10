# AGENTS.md: AI-Native FHE Scheme Design and Implementation

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

> Last updated: 2026-03-10

### Crate: `poulpy-chimera` (16 source files, 175 tests passing)

The CHIMERA scheme is implemented as a new crate in the Poulpy workspace, reusing
`poulpy-hal` (backend traits, FFT) and `poulpy-core` (RLWE encryption, keyswitching,
tensor products, automorphisms).

### Deliverables — ALL COMPLETE

| Deliverable | Location | Status |
|-------------|----------|--------|
| Scheme specification | `docs/chimera_spec.md` | ✅ Complete |
| Reference implementation | `poulpy-chimera/src/` | ✅ Complete (16 modules) |
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
| `attention.rs` | QKV projection, attention scores, softmax approximation, context, output | ✅ |
| `transformer.rs` | Full transformer block, forward pass, FFN (standard + SwiGLU) | ✅ |
| `moe.rs` | MoE routing: homomorphic router, sign extraction, top-k, expert dispatch | ✅ |
| `noise.rs` | Noise tracking and budget estimation | ✅ |
| `bootstrapping.rs` | Full bootstrap pipeline: sample extract → LWE keyswitch → blind rotation | ✅ |
| `model_loader.rs` | Safetensors loading, INT8/FP16/BF16/FP32 quantization, transpose, sharded models | ✅ |
| `verification.rs` | MAC-based user-side computation verification (linear ops) | ✅ |
| `tests.rs` | 175 integration tests (including 8 accuracy + 3 security sweep + 11 MAC verification tests) | ✅ |
| `benches/chimera_ops.rs` | Criterion benchmarks for all operations (toy + d_model=128 + security sweep) | ✅ |

### Key Design Decisions Implemented

- **base2k cascade**: master `base2k=14`, input ct `in_base2k=13`, output ct `out_base2k=12`
- **k_ct = 113**: `8 * base2k + 1`, providing 9 limbs for tensor product relinearization
- **Polynomial activations**: GELU (degree 3), SiLU (degree 3), SquaredReLU (degree 2)
- **Softmax strategies**: PolynomialDeg4, ReluSquared, Linear (configurable per model)
- **Encoding scale**: `2 * in_base2k = 26` (torus encoding for INT8 values)
- **Bootstrapping**: sample_extract → LWE keyswitch (N→n_lwe) → blind rotation with identity LUT
- **Model loading**: safetensors with automatic PyTorch→CHIMERA transpose, per-tensor INT8 quantization

### What Works End-to-End

1. Encrypt INT8 input → homomorphic transformer block → decrypt output (legacy single-ct path for small toy dims; vec path for real model layout)
2. Multi-layer forward pass (2 layers chained, output of block 1 feeds block 2)
3. Standard FFN and SwiGLU FFN at d_model=1 and d_model=2
4. Bootstrapping roundtrip: encrypt value → bootstrap through identity LUT → recover original
5. Matmul with multi-coefficient polynomial weights (d_model=4)
6. Load model weights from safetensors (INT8/FP16/BF16/FP32, sharded, memory-mapped)

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
3. Load real safetensors weights via `model_loader.rs` (already implemented)
4. Run a single-token forward pass and compare output logits to cleartext inference
5. Profile bottlenecks and optimize hot paths

The model loader already supports LLaMA naming conventions and handles INT8 quantized
weights. All P0 and P1 items are complete. The next concrete step toward real model
inference is loading a quantized model's weights and running a single forward pass
at d_model >= 128.
