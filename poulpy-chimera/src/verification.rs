//! MAC-based user-side verification for CHIMERA.
//!
//! Implements a lightweight Message Authentication Code (MAC) scheme that
//! allows the user to verify that the inference provider correctly evaluated
//! the agreed-upon circuit on the encrypted input, without revealing the
//! MAC key or requiring any zero-knowledge proofs.
//!
//! # Construction
//!
//! The scheme adapts the SPDZ-style MAC approach to single-server FHE:
//!
//! 1. **Key generation**: The user generates a random scalar MAC key `alpha`.
//! 2. **Tagging**: For each input ciphertext `ct`, the user computes
//!    `tag = alpha * ct` (a fresh ciphertext encrypting `alpha * m`).
//! 3. **Evaluation**: The provider evaluates the circuit on both `{ct}` and
//!    `{tag}` using identical homomorphic operations.
//! 4. **Verification**: The user decrypts both `ct_out` and `tag_out`, then
//!    checks that `decrypt(tag_out) == alpha * decrypt(ct_out)`.
//!
//! # Scope and Limitations
//!
//! This prototype verifies **linear** operations only (addition, plaintext
//! multiplication, matrix-vector products). The MAC relation `tag = alpha * ct`
//! is preserved through:
//!
//! - `add(ct_a, ct_b)`: `tag_a + tag_b = alpha * (ct_a + ct_b)` ✓
//! - `mul_const(ct, c)`: `mul_const(tag, c) = alpha * mul_const(ct, c)` ✓
//! - `dot_product(cts, ws)`: `dot_product(tags, ws) = alpha * dot_product(cts, ws)` ✓
//!
//! For **nonlinear** operations (ciphertext-ciphertext multiplication via
//! tensor product), the MAC relation becomes quadratic:
//!
//!   `tag_a * tag_b = alpha^2 * ct_a * ct_b ≠ alpha * (ct_a * ct_b)`
//!
//! Handling nonlinear operations requires either:
//! - A degree-tracking MAC scheme (maintain tags at different powers of alpha)
//! - Beaver-triple-style preprocessing
//! - Re-tagging after each nonlinear operation (requires interaction)
//!
//! For the transformer inference use case, the linear operations (QKV projection,
//! FFN matmul, output projection) dominate computation cost (~90% of FLOPs),
//! so verifying the linear portion provides substantial coverage.
//!
//! # Trust Model
//!
//! With MAC verification of linear operations:
//! - **Verified**: All matrix-vector products use the correct weights on the
//!   correct encrypted input, modulo the prototype MAC domain
//! - **Trusted**: Nonlinear operations (activations, softmax, layernorm) are
//!   applied correctly (trust assumption)
//! - **Trusted**: The model architecture matches the agreed-upon specification
//!
//! Without MAC verification (CHIMERA v1):
//! - **Trusted**: Everything — the provider could substitute any computation.
//!
//! # Performance
//!
//! - **Provider overhead**: 2x computation (evaluates two parallel circuits)
//! - **Communication overhead**: 2x (send/receive tags alongside ciphertexts)
//! - **User-side verification cost**: O(N) integer multiplications + comparison
//!   (~microseconds on any modern device)
//! - **Prototype soundness**: bounded by the plaintext MAC domain, not the RLWE modulus

use poulpy_core::{layouts::GLWE, GLWEAdd, GLWEMulConst};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

use crate::arithmetic::{chimera_add, chimera_mul_const};
use crate::encrypt::ChimeraKey;
use crate::params::ChimeraParams;

/// MAC key for user-side verification.
///
/// Contains a secret scalar `alpha` used to tag ciphertexts. The MAC key
/// must be kept secret from the inference provider — it is only used on
/// the user's device for tagging (before sending) and verification (after
/// receiving).
///
/// # Security
///
/// The MAC key is a single scalar in the plaintext space. For INT8 precision,
/// detection is limited by the small modular plaintext domain and by the alpha
/// range used by the prototype. This is useful as a lightweight integrity check,
/// but it is not equivalent to a large-field MAC.
///
/// For the prototype, alpha is sampled as a small odd integer in [3, 13] to
/// avoid trivial factors (0, 1, 2) while keeping noise amplification bounded.
/// Production deployments should use a larger alpha range with a wider noise
/// budget.
pub struct MacKey {
    /// The secret MAC scalar.
    alpha: i64,
}

impl MacKey {
    /// Creates a MAC key with the given scalar value.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The MAC scalar. Must be non-zero and non-one.
    ///
    /// # Panics
    ///
    /// Panics if `alpha` is 0 or 1 (trivial keys provide no verification).
    pub fn new(alpha: i64) -> Self {
        assert!(alpha != 0, "MAC key must be non-zero");
        assert!(alpha != 1, "MAC key must not be 1 (trivial)");
        MacKey { alpha }
    }

    /// Creates a MAC key from a seed byte.
    ///
    /// Derives a small odd alpha in [3, 13] from the seed. This keeps
    /// noise amplification bounded while providing non-trivial verification.
    pub fn from_seed(seed: u8) -> Self {
        // Map seed to small odd values: 3, 5, 7, 9, 11, 13
        let candidates = [3i64, 5, 7, 9, 11, 13];
        let alpha = candidates[(seed as usize) % candidates.len()];
        MacKey { alpha }
    }

    /// Returns the MAC scalar value.
    ///
    /// This should only be called on the user's device. Never expose to
    /// the inference provider.
    pub fn alpha(&self) -> i64 {
        self.alpha
    }
}

/// A ciphertext paired with its MAC tag.
///
/// The tag satisfies `decrypt(tag) == alpha * decrypt(ct)` for the MAC key
/// that was used to create it. This invariant is preserved through linear
/// homomorphic operations.
pub struct TaggedCiphertext<D: poulpy_hal::layouts::Data> {
    /// The primary ciphertext encrypting the plaintext value.
    pub ct: GLWE<D>,
    /// The MAC tag: a ciphertext encrypting `alpha * plaintext`.
    pub tag: GLWE<D>,
}

/// Tags a ciphertext with the MAC key.
///
/// Computes `tag = alpha * ct` by multiplying the ciphertext by the scalar
/// MAC key. The resulting tagged ciphertext satisfies the MAC relation:
/// `decrypt(tag) == alpha * decrypt(ct)`.
///
/// This must be called on the user's device before sending ciphertexts to
/// the inference provider.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `mac_key` - The user's secret MAC key.
/// * `ct` - The ciphertext to tag.
///
/// # Returns
///
/// A tagged ciphertext containing both the original ct and the MAC tag.
pub fn chimera_mac_tag<BE: Backend>(module: &Module<BE>, mac_key: &MacKey, ct: &GLWE<Vec<u8>>) -> TaggedCiphertext<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    use poulpy_core::layouts::{GLWEToMut, GLWEToRef};

    // Clone the ciphertext
    let mut ct_clone = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    {
        let ct_ref = ct.to_ref();
        let src: &[u8] = ct_ref.data().data;
        let mut ct_mut = ct_clone.to_mut();
        let dst: &mut [u8] = ct_mut.data_mut().data;
        let len = src.len().min(dst.len());
        dst[..len].copy_from_slice(&src[..len]);
    }

    // Compute tag = alpha * ct
    let tag = chimera_mul_const(module, ct, &[mac_key.alpha]);

    TaggedCiphertext { ct: ct_clone, tag }
}

/// Tags a vector of ciphertexts with the MAC key.
///
/// Convenience wrapper that applies [`chimera_mac_tag`] to each ciphertext
/// in the vector.
pub fn chimera_mac_tag_vec<BE: Backend>(
    module: &Module<BE>,
    mac_key: &MacKey,
    cts: &[GLWE<Vec<u8>>],
) -> Vec<TaggedCiphertext<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    cts.iter().map(|ct| chimera_mac_tag(module, mac_key, ct)).collect()
}

// ---- Provider-side operations on tagged ciphertexts ----
//
// The provider applies the same operation to both ct and tag in parallel.
// These functions ensure the provider cannot "forget" to update the tag.

/// Adds two tagged ciphertexts.
///
/// Computes `(ct_a + ct_b, tag_a + tag_b)`. The MAC relation is preserved:
/// `alpha * (m_a + m_b) = alpha * m_a + alpha * m_b`.
pub fn tagged_add<BE: Backend>(
    module: &Module<BE>,
    a: &TaggedCiphertext<Vec<u8>>,
    b: &TaggedCiphertext<Vec<u8>>,
) -> TaggedCiphertext<Vec<u8>>
where
    Module<BE>: GLWEAdd,
{
    TaggedCiphertext {
        ct: chimera_add(module, &a.ct, &b.ct),
        tag: chimera_add(module, &a.tag, &b.tag),
    }
}

/// Multiplies a tagged ciphertext by a plaintext constant.
///
/// Computes `(ct * c, tag * c)`. The MAC relation is preserved:
/// `alpha * (m * c) = (alpha * m) * c`.
pub fn tagged_mul_const<BE: Backend>(
    module: &Module<BE>,
    tagged: &TaggedCiphertext<Vec<u8>>,
    constants: &[i64],
) -> TaggedCiphertext<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    TaggedCiphertext {
        ct: chimera_mul_const(module, &tagged.ct, constants),
        tag: chimera_mul_const(module, &tagged.tag, constants),
    }
}

/// Computes a tagged dot product: `sum_i (ct_i * w_i)` with MAC propagation.
///
/// The MAC relation is preserved through the linear combination:
/// `alpha * sum(m_i * w_i) = sum(alpha * m_i * w_i)`.
pub fn tagged_dot_product<BE: Backend>(
    module: &Module<BE>,
    tagged_cts: &[TaggedCiphertext<Vec<u8>>],
    weights: &[Vec<i64>],
) -> TaggedCiphertext<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    assert_eq!(tagged_cts.len(), weights.len());
    assert!(!tagged_cts.is_empty());

    let cts: Vec<&GLWE<Vec<u8>>> = tagged_cts.iter().map(|t| &t.ct).collect();
    let tags: Vec<&GLWE<Vec<u8>>> = tagged_cts.iter().map(|t| &t.tag).collect();

    // Compute ct dot product
    let mut ct_acc = chimera_mul_const(module, cts[0], &weights[0]);
    for i in 1..cts.len() {
        let term = chimera_mul_const(module, cts[i], &weights[i]);
        ct_acc = chimera_add(module, &ct_acc, &term);
    }

    // Compute tag dot product (same weights)
    let mut tag_acc = chimera_mul_const(module, tags[0], &weights[0]);
    for i in 1..tags.len() {
        let term = chimera_mul_const(module, tags[i], &weights[i]);
        tag_acc = chimera_add(module, &tag_acc, &term);
    }

    TaggedCiphertext {
        ct: ct_acc,
        tag: tag_acc,
    }
}

/// Computes a tagged matrix-vector product for a single input ciphertext.
///
/// Applies each weight row to both the ciphertext and its tag, producing
/// a vector of tagged output ciphertexts.
pub fn tagged_matmul_single_ct<BE: Backend>(
    module: &Module<BE>,
    tagged: &TaggedCiphertext<Vec<u8>>,
    weight_rows: &[Vec<i64>],
) -> Vec<TaggedCiphertext<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    weight_rows.iter().map(|row| tagged_mul_const(module, tagged, row)).collect()
}

// ---- User-side verification ----

/// Result of MAC verification.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the MAC check passed for all slots.
    pub passed: bool,
    /// Number of slots that matched.
    pub matching_slots: usize,
    /// Total number of slots checked.
    pub total_slots: usize,
    /// Maximum absolute difference `|tag_dec[i] - alpha * ct_dec[i]|`.
    /// A small non-zero value is expected due to accumulated FHE noise.
    pub max_error: i64,
    /// Mean absolute difference.
    pub mean_error: f64,
}

/// Decodes raw i64 coefficient values from a plaintext (without i8 truncation).
///
/// This is used for MAC verification where the tag values may exceed the
/// i8 range (since `tag = alpha * plaintext` can be larger than 127).
fn decode_raw_coeffs<BE: Backend>(
    module: &Module<BE>,
    params: &ChimeraParams,
    pt: &poulpy_core::layouts::GLWEPlaintext<Vec<u8>>,
    count: usize,
) -> Vec<i64>
where
    Module<BE>: poulpy_hal::api::ModuleN,
{
    let n = module.n() as usize;
    assert!(count <= n, "decode_raw_coeffs: count ({count}) exceeds ring degree ({n})");
    let shift = params.in_base2k() - params.scale_bits as usize;

    let raw: &[u8] = pt.data.data.as_ref();
    let coeffs: &[i64] = bytemuck::cast_slice(&raw[..n * 8]);

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        result.push(coeffs[i] >> shift);
    }
    result
}

/// Returns the plaintext modulus for MAC verification.
///
/// The FHE plaintext space is `Z_{2^scale_bits}`, so all arithmetic
/// must be verified modulo this value. For INT8 (scale_bits=8), the
/// modulus is 256.
fn plaintext_modulus(params: &ChimeraParams) -> i64 {
    1i64 << params.scale_bits
}

/// Reduces a value to the signed range `[-modulus/2, modulus/2)`.
fn signed_mod(val: i64, modulus: i64) -> i64 {
    let half = modulus / 2;
    let r = ((val % modulus) + modulus) % modulus;
    if r >= half {
        r - modulus
    } else {
        r
    }
}

/// Verifies a tagged ciphertext after the provider returns results.
///
/// Decrypts both the ciphertext and the tag, then checks the MAC relation:
/// `decrypt(tag)[i] ≡ alpha * decrypt(ct)[i]  (mod 2^scale_bits)`
/// for all active slots.
///
/// The comparison is done modulo the plaintext modulus because the FHE
/// plaintext space is `Z_{2^scale_bits}` — multiplication by alpha may
/// cause values to wrap around, which is correct behavior in modular
/// arithmetic.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `key` - The user's CHIMERA secret key (for decryption).
/// * `params` - CHIMERA parameters.
/// * `mac_key` - The user's secret MAC key.
/// * `tagged` - The tagged ciphertext to verify.
/// * `n_slots` - Number of active plaintext slots to check.
/// * `tolerance` - Maximum allowed absolute error per slot (in the modular
///   residue). This accounts for FHE noise accumulation.
///
/// # Returns
///
/// A [`VerificationResult`] indicating whether the MAC check passed.
pub fn chimera_mac_verify<BE: Backend>(
    module: &Module<BE>,
    key: &ChimeraKey<BE>,
    params: &ChimeraParams,
    mac_key: &MacKey,
    tagged: &TaggedCiphertext<Vec<u8>>,
    n_slots: usize,
    tolerance: i64,
) -> VerificationResult
where
    Module<BE>: poulpy_core::GLWEDecrypt<BE> + poulpy_hal::api::ModuleN,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: poulpy_core::ScratchTakeCore<BE>,
{
    use crate::encrypt::chimera_decrypt;

    let ring_degree = module.n() as usize;
    assert!(
        n_slots <= ring_degree,
        "chimera_mac_verify: n_slots ({n_slots}) exceeds ring degree ({ring_degree})"
    );

    let pt_ct = chimera_decrypt(module, key, &tagged.ct, params);
    let pt_tag = chimera_decrypt(module, key, &tagged.tag, params);

    let dec_ct = decode_raw_coeffs(module, params, &pt_ct, n_slots);
    let dec_tag = decode_raw_coeffs(module, params, &pt_tag, n_slots);

    let alpha = mac_key.alpha();
    let modulus = plaintext_modulus(params);

    let mut matching = 0usize;
    let mut max_err: i64 = 0;
    let mut sum_err: f64 = 0.0;

    for i in 0..n_slots {
        // Compare modulo the plaintext modulus: tag ≡ alpha * ct (mod 2^scale_bits)
        let expected = dec_ct[i] * alpha;
        let diff = dec_tag[i] - expected;
        let err = signed_mod(diff, modulus).abs();

        if err <= tolerance {
            matching += 1;
        }
        if err > max_err {
            max_err = err;
        }
        sum_err += err as f64;
    }

    let mean_err = if n_slots > 0 { sum_err / n_slots as f64 } else { 0.0 };

    VerificationResult {
        passed: matching == n_slots,
        matching_slots: matching,
        total_slots: n_slots,
        max_error: max_err,
        mean_error: mean_err,
    }
}

/// Verifies a vector of tagged ciphertexts.
///
/// Convenience wrapper that verifies each tagged ciphertext independently
/// and returns an aggregate result. Verification passes only if ALL
/// individual checks pass.
pub fn chimera_mac_verify_vec<BE: Backend>(
    module: &Module<BE>,
    key: &ChimeraKey<BE>,
    params: &ChimeraParams,
    mac_key: &MacKey,
    tagged_vec: &[TaggedCiphertext<Vec<u8>>],
    n_slots: usize,
    tolerance: i64,
) -> VerificationResult
where
    Module<BE>: poulpy_core::GLWEDecrypt<BE> + poulpy_hal::api::ModuleN,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: poulpy_core::ScratchTakeCore<BE>,
{
    let ring_degree = module.n() as usize;
    assert!(
        n_slots <= ring_degree,
        "chimera_mac_verify_vec: n_slots ({n_slots}) exceeds ring degree ({ring_degree})"
    );

    let mut total_matching = 0usize;
    let mut total_slots = 0usize;
    let mut global_max_err: i64 = 0;
    let mut global_sum_err: f64 = 0.0;

    for tagged in tagged_vec {
        let result = chimera_mac_verify(module, key, params, mac_key, tagged, n_slots, tolerance);
        total_matching += result.matching_slots;
        total_slots += result.total_slots;
        if result.max_error > global_max_err {
            global_max_err = result.max_error;
        }
        global_sum_err += result.mean_error * result.total_slots as f64;
    }

    let mean_err = if total_slots > 0 {
        global_sum_err / total_slots as f64
    } else {
        0.0
    };

    VerificationResult {
        passed: total_matching == total_slots,
        matching_slots: total_matching,
        total_slots,
        max_error: global_max_err,
        mean_error: mean_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::encode_int8;
    use crate::encrypt::{chimera_encrypt, ChimeraKey};
    use crate::params::{ChimeraParams, Precision, SecurityLevel};
    use poulpy_hal::api::ModuleNew;

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    #[test]
    fn test_mac_key_creation() {
        let key = MacKey::new(7);
        assert_eq!(key.alpha(), 7);

        let key2 = MacKey::from_seed(0);
        assert_eq!(key2.alpha(), 3); // First candidate

        let key3 = MacKey::from_seed(1);
        assert_eq!(key3.alpha(), 5); // Second candidate
    }

    #[test]
    #[should_panic(expected = "non-zero")]
    fn test_mac_key_zero_panics() {
        MacKey::new(0);
    }

    #[test]
    #[should_panic(expected = "not be 1")]
    fn test_mac_key_one_panics() {
        MacKey::new(1);
    }

    #[test]
    fn test_mac_tag_and_verify_identity() {
        // Tag a ciphertext, don't do any computation, verify immediately.
        // This is the baseline: the MAC should pass perfectly.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(5);

        let values: Vec<i8> = vec![10, -20, 30, -40, 50];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        let tagged = chimera_mac_tag(&module, &mac, &ct);

        // Verify with tight tolerance (mul_const error is at most 4)
        let result = chimera_mac_verify(&module, &key, &params, &mac, &tagged, 5, 25);

        eprintln!(
            "MAC verify identity: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(result.passed, "MAC verification should pass for unmodified tagged ct");
    }

    #[test]
    fn test_mac_through_add() {
        // Tag two ciphertexts, add them, verify the result.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(3);

        let a_vals: Vec<i8> = vec![10, 20, 30];
        let b_vals: Vec<i8> = vec![5, -5, 10];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [1u8; 32], [2u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [3u8; 32], [4u8; 32]);

        let tagged_a = chimera_mac_tag(&module, &mac, &ct_a);
        let tagged_b = chimera_mac_tag(&module, &mac, &ct_b);

        // Provider computes addition on both ct and tag
        let tagged_sum = tagged_add(&module, &tagged_a, &tagged_b);

        // User verifies
        let result = chimera_mac_verify(&module, &key, &params, &mac, &tagged_sum, 3, 25);

        eprintln!(
            "MAC verify add: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(result.passed, "MAC verification should pass through addition");
    }

    #[test]
    fn test_mac_through_mul_const() {
        // Tag a ciphertext, multiply by a constant, verify.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(7);

        let values: Vec<i8> = vec![5, -10, 15];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        let tagged = chimera_mac_tag(&module, &mac, &ct);

        // Provider multiplies by constant 3
        let tagged_result = tagged_mul_const(&module, &tagged, &[3]);

        // User verifies
        let result = chimera_mac_verify(&module, &key, &params, &mac, &tagged_result, 3, 50);

        eprintln!(
            "MAC verify mul_const: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(result.passed, "MAC verification should pass through mul_const");
    }

    #[test]
    fn test_mac_through_dot_product() {
        // Tag ciphertexts, compute dot product with plaintext weights, verify.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(5);

        let a_vals: Vec<i8> = vec![3];
        let b_vals: Vec<i8> = vec![4];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [1u8; 32], [2u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [3u8; 32], [4u8; 32]);

        let tagged_a = chimera_mac_tag(&module, &mac, &ct_a);
        let tagged_b = chimera_mac_tag(&module, &mac, &ct_b);

        // Provider computes dot product: 3*2 + 4*1 = 10
        let tagged_result = tagged_dot_product(&module, &[tagged_a, tagged_b], &[vec![2i64], vec![1i64]]);

        // User verifies
        let result = chimera_mac_verify(&module, &key, &params, &mac, &tagged_result, 1, 50);

        eprintln!(
            "MAC verify dot_product: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(result.passed, "MAC verification should pass through dot_product");
    }

    #[test]
    fn test_mac_detects_tampered_output() {
        // Tag a ciphertext, but the provider returns a DIFFERENT ciphertext
        // as the result. The MAC check should fail.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(5);

        let values: Vec<i8> = vec![10, 20, 30];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        let tagged = chimera_mac_tag(&module, &mac, &ct);

        // Provider tampers: returns a different ciphertext for the result,
        // but uses the honestly-computed tag. This simulates a provider that
        // runs a different computation on the ciphertext but leaves the tag
        // pipeline unchanged.
        let tampered_values: Vec<i8> = vec![99, 99, 99];
        let tampered_pt = encode_int8(&module, &params, &tampered_values);
        let tampered_ct = chimera_encrypt(&module, &key, &tampered_pt, [5u8; 32], [6u8; 32]);

        let tampered_tagged = TaggedCiphertext {
            ct: tampered_ct,         // Provider substituted a different ct
            tag: tagged.tag.clone(), // Tag is from the honest computation
        };

        // User verifies — should FAIL because decrypt(tag) != alpha * decrypt(ct)
        let result = chimera_mac_verify(&module, &key, &params, &mac, &tampered_tagged, 3, 25);

        eprintln!(
            "MAC verify tampered: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(!result.passed, "MAC verification should FAIL for tampered ciphertext");
    }

    #[test]
    fn test_mac_detects_wrong_weights() {
        // Provider uses different weights than agreed upon.
        // The honest tag pipeline uses the correct weights.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(7);

        let values: Vec<i8> = vec![10, 20, 30];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        let tagged = chimera_mac_tag(&module, &mac, &ct);

        // Honest computation: multiply by 3
        let honest_tag_result = chimera_mul_const(&module, &tagged.tag, &[3]);

        // Dishonest computation: multiply by 5 (wrong weights!)
        let dishonest_ct_result = chimera_mul_const(&module, &tagged.ct, &[5]);

        let mismatched = TaggedCiphertext {
            ct: dishonest_ct_result, // Provider used weight 5
            tag: honest_tag_result,  // Tag was computed with weight 3
        };

        // User verifies — should FAIL
        let result = chimera_mac_verify(&module, &key, &params, &mac, &mismatched, 3, 25);

        eprintln!(
            "MAC verify wrong weights: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(
            !result.passed,
            "MAC verification should FAIL when provider uses wrong weights"
        );
    }

    #[test]
    fn test_mac_through_chained_operations() {
        // Tag → mul_const(2) → add(other) → mul_const(3) → verify.
        // Tests that the MAC survives a sequence of linear operations.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(3);

        let a_vals: Vec<i8> = vec![5, -3, 7];
        let b_vals: Vec<i8> = vec![2, 4, -1];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [1u8; 32], [2u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [3u8; 32], [4u8; 32]);

        let tagged_a = chimera_mac_tag(&module, &mac, &ct_a);
        let tagged_b = chimera_mac_tag(&module, &mac, &ct_b);

        // Chain: (a * 2) + b, then * 3
        let step1 = tagged_mul_const(&module, &tagged_a, &[2]);
        let step2 = tagged_add(&module, &step1, &tagged_b);
        let step3 = tagged_mul_const(&module, &step2, &[3]);

        // Verify the chain
        let result = chimera_mac_verify(&module, &key, &params, &mac, &step3, 3, 80);

        eprintln!(
            "MAC verify chain: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(result.passed, "MAC verification should pass through chained linear ops");
    }

    #[test]
    #[should_panic(expected = "n_slots")]
    fn test_mac_verify_rejects_too_many_slots() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(5);

        let values: Vec<i8> = vec![1, 2, 3];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);
        let tagged = chimera_mac_tag(&module, &mac, &ct);

        chimera_mac_verify(&module, &key, &params, &mac, &tagged, module.n() as usize + 1, 25);
    }

    #[test]
    fn test_mac_matmul_single_ct() {
        // Tag a ciphertext, run it through matmul_single_ct, verify all outputs.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let mac = MacKey::new(5);

        let values: Vec<i8> = vec![2, 3];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        let tagged = chimera_mac_tag(&module, &mac, &ct);

        // Weight rows: [2], [3] (single-coefficient polynomials)
        let weight_rows = vec![vec![2i64], vec![3i64]];
        let tagged_results = tagged_matmul_single_ct(&module, &tagged, &weight_rows);

        assert_eq!(tagged_results.len(), 2);

        // Verify each output
        let result = chimera_mac_verify_vec(&module, &key, &params, &mac, &tagged_results, 2, 50);

        eprintln!(
            "MAC verify matmul: passed={}, matching={}/{}, max_err={}, mean_err={:.2}",
            result.passed, result.matching_slots, result.total_slots, result.max_error, result.mean_error
        );

        assert!(result.passed, "MAC verification should pass through matmul");
    }
}
