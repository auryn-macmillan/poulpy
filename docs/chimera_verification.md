# FHE_LLM Verification Analysis

## 1. Problem Statement

In FHE_LLM's deployment model, a user sends encrypted input to a remote inference
provider, who evaluates a transformer model homomorphically and returns an
encrypted result. The user decrypts locally.

The provider cannot learn the user's input or output (guaranteed by RLWE security).
However, a malicious provider could:

1. **Substitute the model**: Run a different (cheaper, biased) model instead of
   the agreed-upon one
2. **Tamper with the output**: Return an encrypted garbage value or a value
   computed from a different input
3. **Skip computation**: Return an encryption of zero or a random value
4. **Partial computation**: Run only part of the model (e.g. fewer layers)

The user, upon decryption, receives a plausible-looking output but has no way
to verify that the correct computation was performed.

### Trust Model Without Verification

In the absence of verification, the user must trust:
- The provider runs the correct model (model integrity)
- The provider evaluates the full forward pass on the user's encrypted input
  (computation integrity)
- The provider does not inject adversarial perturbations into intermediate
  ciphertexts (output integrity)

This is a **significant trust assumption**. For the governance use case,
where inference results feed into aggregation that affects collective
decisions, an undetected model substitution could systematically bias outcomes.

## 2. Verification Approaches

### 2.1 Full zkSNARK of FHE Evaluation

**Approach**: The provider generates a zero-knowledge succinct non-interactive
argument of knowledge (zkSNARK) proving that the encrypted output was computed
by correctly applying the homomorphic evaluation circuit to the encrypted input.

**Prior art**: ezkl, zkCNN (Liu et al., 2021), zkML (various).

**Cost analysis for FHE_LLM**:

| Component          | Estimate                          |
|--------------------|-----------------------------------|
| Circuit size       | O(L × N × log₂q) ≈ 10^12 gates  |
| Proof generation   | ~10-100 hours per forward pass    |
| Proof size         | ~200-500 bytes (Groth16)          |
| Verification time  | ~10 ms                            |

**Verdict**: Proof generation is completely impractical. The FHE evaluation
circuit for a single transformer forward pass contains trillions of arithmetic
gates (each NTT butterfly, each polynomial multiplication, each rescaling). Even
with recursive proof composition, generation time exceeds the inference time by
orders of magnitude.

### 2.2 Commit-and-Prove (Layer-by-Layer)

**Approach**: The provider commits to each intermediate ciphertext after each
transformer layer using a polynomial commitment scheme (e.g. KZG, FRI). A
separate proof shows that each commitment is consistent with the previous one
via correct evaluation of one layer's operations.

**Cost analysis**:

| Component          | Per layer               | Total (32 layers)       |
|--------------------|------------------------|-------------------------|
| Commitment         | O(N) group operations  | 32 × O(N)              |
| Proof generation   | O(N × log N) (FFT)    | ~minutes per layer      |
| Proof size         | O(1) group elements    | ~32 × 48 bytes ≈ 1.5 KB|
| Verification       | O(1) pairings per layer| ~32 pairings ≈ 100 ms  |

**Verdict**: More promising but still expensive. The main bottleneck is that
each "correct evaluation of one layer" includes NTT/FFT operations, polynomial
multiplications, and key-switching — each of which must be expressed as an
arithmetic circuit and proven. The per-layer proof generation would take
minutes to tens of minutes, making total proof generation ~1-10 hours for a
32-layer model.

### 2.3 MAC-Based Verification (Recommended Approach)

**Approach**: Adapt algebraic Message Authentication Codes (MACs) to the FHE
evaluation. The user tags each input ciphertext with a secret MAC. The
provider evaluates the circuit on both the ciphertexts and their MACs in
parallel. On receiving the output, the user verifies the MAC.

This is analogous to the SPDZ protocol's MAC-based verification, adapted for
single-server computation.

**Construction**:

1. **Setup**: User generates a random MAC key α ∈ R_q
2. **Input tagging**: For each input ciphertext ct_i, the user also sends
   tag_i = α · ct_i (computed locally before encryption, or as a separate
   encrypted value)
3. **Evaluation**: Provider evaluates the circuit on both {ct_i} and {tag_i}
   using identical homomorphic operations
4. **Verification**: User decrypts both the result ct_out and tag_out, and
   checks that tag_out = α · ct_out

If the provider deviates from the prescribed linear circuit, the MAC check will
fail with probability determined by the plaintext MAC domain and the alpha
distribution used by the prototype. This is useful as a lightweight integrity
check, but it is not equivalent to a large-field algebraic MAC.

**Cost analysis**:

| Component            | Estimate                           |
|----------------------|------------------------------------|
| Provider overhead    | 2x (evaluates two parallel circuits)|
| User-side cost       | One polynomial multiplication      |
| Communication        | 2x (send/receive tags alongside cts)|
| Verification         | O(N) multiplications + comparison  |
| Proof size           | 0 (implicit in the computation)    |
| Prototype soundness  | Bounded by plaintext MAC domain   |

**Verdict**: Practical. The provider's computation doubles (two parallel
FHE evaluations with the same circuit), but this is linear overhead —
the same 5-14 seconds per layer becomes 10-28 seconds. Verification is
a single polynomial multiplication on the user's device (~microseconds).

**Limitations**:
- Does not prove that the correct *model* was used, only that the same
  circuit was applied to both the ciphertext and the MAC
- Requires the user to know the evaluation circuit (model architecture)
  to verify consistency
- Doubles communication and provider computation

### 2.4 Probabilistic Checkpointing

**Approach**: The user encrypts a small number of known test vectors
alongside their actual input. The provider does not know which inputs
are test vectors. After decryption, the user checks the test vector
outputs against locally computed expected outputs.

**Construction**:

1. User selects k test inputs x_1, ..., x_k with known outputs y_1, ..., y_k
2. User encrypts all n+k inputs (real + test) and shuffles them
3. Provider evaluates the model on all n+k inputs
4. User decrypts all outputs, checks test outputs match expected values

**Cost analysis**:

| Component            | Estimate (k=5 test vectors)       |
|----------------------|-----------------------------------|
| Provider overhead    | (n+k)/n ≈ 1.05x (for n=100)     |
| User-side cost       | k forward passes (local, cleartext)|
| Communication        | (n+k)/n ≈ 1.05x                  |
| Verification         | k equality checks                 |
| Detection probability| 1 - (1 - k/(n+k))^m for m tampered|

**Verdict**: Lightweight and practical, but provides only probabilistic
guarantees. A provider that selectively tampers with a small fraction of
inputs has a non-negligible chance of avoiding detection.

For the governance use case (batch size 1), this approach is less useful
because the provider knows every ciphertext is the user's real input.

### 2.5 Hybrid: MAC + Architecture Attestation

The most practical verification system for FHE_LLM combines:

1. **MAC-based computation integrity** (Section 2.3): Proves the provider
   applied the agreed circuit correctly
2. **Trusted execution attestation**: The evaluation circuit (model weights
   and architecture) is committed to a public hash; the provider attests
   (via TEE, reputation, or stake) that it is running the committed circuit

This provides:
- **Computation integrity**: MAC verification for supported linear operations (prototype-strength, bounded by plaintext MAC domain)
- **Model integrity**: Attestation (trust-based, with cryptographic commitment
  to the model hash)

## 3. Recommended Design

### 3.1 For FHE_LLM v1 (Current Implementation)

**Prototype MAC verification for linear operations only**.

> The user can verify linear operations (addition, plaintext multiplication,
> dot products, matrix-vector products) using MAC tags in the plaintext MAC
> domain. Nonlinear operations (ciphertext-ciphertext multiplication, softmax,
> activations, layernorm) remain trusted. The FHE encryption guarantees data
> privacy; integrity is partial and prototype-grade.

### 3.2 For FHE_LLM v2 (Future Work)

Implement **MAC-based verification** (Section 2.3):

1. Extend `FHE_LLMKey` to include a MAC key α
2. Extend `FHE_LLM_encrypt` to produce both ct and tag = α · ct
3. Extend the evaluation API to accept paired (ct, tag) inputs
4. Add `FHE_LLM_verify(key, ct_out, tag_out) -> bool`

**Estimated implementation effort**: 2-4 weeks for a senior cryptography
engineer. The main complexity is ensuring the MAC propagation is correct
through all homomorphic operations (addition, multiplication, rescaling,
rotation).

**Performance impact**: 2x provider computation, 2x communication, ~0
user-side verification cost.

### 3.3 For FHE_LLM v3 (Aspirational)

Combine MAC verification with a lightweight ZK proof that the evaluation
circuit matches a committed model hash. This could use:

- A Merkle commitment to the model weights
- A ZK proof (Bulletproofs-style) that the weights used in each matmul
  step are consistent with the Merkle root
- Total additional proof size: ~50-100 KB
- Verification: ~1-5 seconds on a laptop

This is the "minimum viable verification" described in the project spec
(~50 KB proof, ~5 second verification). It requires significant engineering
but is feasible with existing ZK proof systems.

## 4. Comparison of Approaches

| Approach            | Soundness  | Provider cost | User cost  | Proof size | Practical? |
|---------------------|------------|-------------|------------|------------|------------|
| Full zkSNARK        | 2^(-128)   | 10-100 hours| ~10 ms     | ~300 B     | No         |
| Commit-and-prove    | 2^(-128)   | 1-10 hours  | ~100 ms    | ~1.5 KB    | Marginal   |
| MAC-based           | Prototype-domain | 2x    | ~1 μs      | 0          | Yes        |
| Probabilistic check | 1-(1-p)^m  | ~1x         | k inferences| 0         | Limited*   |
| Hybrid MAC+attest   | Prototype-domain + trust | 2x | ~1 μs | 0 | Yes |

*Limited for batch size 1 (the target workload).

## 5. Open Questions

1. **Can the MAC overhead be reduced below 2x?** Possible approaches include
   batch MAC verification (amortise one MAC across multiple ciphertexts) or
   sparse MAC tagging (tag only a random subset of intermediate values).

2. **Is there a practical ZK proof for "correct NTT"?** The NTT is the core
   bottleneck in proving FHE operations. A specialised ZK proof for NTT
   correctness could make the commit-and-prove approach feasible.

3. **Can the model commitment be verified without ZK?** If the model weights
   are public (e.g. open-weight models like Mixtral), the user can verify
   the model hash directly. ZK is only needed when the model weights are
   private (provider's proprietary model).

4. **What is the interaction with Phase 2 (Enclave aggregation)?** The
   aggregation layer may provide additional verification guarantees (e.g.
   if multiple users' inferences must be consistent with the same model).
   This could reduce the per-user verification burden.

## 6. Summary

| Property                    | Status          | Notes                        |
|-----------------------------|-----------------|------------------------------|
| Data privacy (input/output) | Guaranteed      | RLWE IND-CPA security        |
| Computation integrity       | Partially verified | Linear ops only, prototype MAC domain |
| Model integrity             | Not verified    | Trust-based / out of scope   |
| Recommended next step       | Stronger MAC domain | Larger plaintext domain or richer tagging |
| Aspirational v3             | MAC + ZK model  | ~50 KB proof, ~5 sec verify  |
| User-side verification cost | ~1 μs (MAC)     | Practical on any device      |
| Provider-side overhead      | 2x (MAC)        | Acceptable for inference     |

## 7. Prototype Implementation

A working MAC-based verification prototype is implemented in
`poulpy-FHE_LLM/src/verification.rs` (11 tests, all passing).

### API

```rust
// User-side: key generation and tagging
let mac = MacKey::new(7);                         // or MacKey::from_seed(seed)
let tagged = FHE_LLM_mac_tag(&module, &mac, &ct); // produces TaggedCiphertext

// Provider-side: homomorphic operations on tagged ciphertexts
let tagged_sum   = tagged_add(&module, &tagged_a, &tagged_b);
let tagged_prod  = tagged_mul_const(&module, &tagged, &weights);
let tagged_dot   = tagged_dot_product(&module, &tagged_vec, &weights);
let tagged_matmul = tagged_matmul_single_ct(&module, &tagged, &weight_rows);

// User-side: verification after receiving results
let result = FHE_LLM_mac_verify(&module, &key, &params, &mac, &tagged_out, n_slots, tolerance);
assert!(result.passed);
```

### Key Design Decisions

1. **Modular arithmetic comparison**: The FHE plaintext space is
   `Z_{2^scale_bits}` (modulus 256 for INT8). MAC verification compares
   `decrypt(tag) ≡ α · decrypt(ct) (mod 2^scale_bits)`, not over the integers.
   This correctly handles the case where `α · m` wraps around the plaintext
   modulus.

2. **Scalar MAC key**: The prototype uses a small scalar `α ∈ {3,5,7,9,11,13}`
   as the MAC key. This keeps noise amplification bounded while providing
   non-trivial verification. Production deployments should use a larger α
   sampled from the full plaintext ring for stronger soundness.

3. **Linear operations only**: The MAC relation `tag = α · ct` is preserved
   through addition, plaintext multiplication, and dot products. Ciphertext-
   ciphertext multiplication (tensor product) breaks the MAC relation
   because `α · ct_a · α · ct_b = α² · ct_a · ct_b ≠ α · (ct_a · ct_b)`.

### Test Results

| Test | MAC α | Operations | max_err | Result |
|------|-------|-----------|---------|--------|
| Identity (tag then verify) | 5 | none | 0 | ✅ PASS |
| Addition | 3 | add(a, b) | 0 | ✅ PASS |
| Mul_const | 7 | mul_const(ct, 3) | 0 | ✅ PASS |
| Dot product | 5 | dot(cts, weights) | 0 | ✅ PASS |
| Matmul | 5 | matmul_single_ct | 0 | ✅ PASS |
| Chained ops | 3 | mul→add→mul | 0 | ✅ PASS |
| Tampered output | 5 | substituted ct | 117 | ✅ DETECTED |
| Wrong weights | 7 | provider used w=5 not w=3 | 116 | ✅ DETECTED |

All honest computations produce **zero** MAC error (exact match modulo the
plaintext modulus). All dishonest computations produce large MAC errors and
are reliably detected.
