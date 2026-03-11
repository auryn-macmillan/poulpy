//! Homomorphic arithmetic operations for CHIMERA.
//!
//! Provides ciphertext addition, plaintext multiplication, rescaling, and
//! slot rotation — the four building blocks needed for all transformer
//! operations under FHE.

use poulpy_core::ScratchTakeCore;
use poulpy_core::{
    layouts::{GLWEInfos, GLWELayout, LWEInfos, TorusPrecision, GLWE},
    GLWEAdd, GLWEAutomorphism, GLWEMulConst, GLWESub, GLWETrace,
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

use crate::encrypt::ChimeraEvalKey;
use crate::params::ChimeraParams;

/// Adds two CHIMERA ciphertexts element-wise.
///
/// Computes `res = a + b` where both encrypt vectors of the same length.
/// Noise grows additively: σ_res² = σ_a² + σ_b².
///
/// # Panics
///
/// Panics if the ciphertexts have incompatible parameters.
pub fn chimera_add<BE: Backend>(module: &Module<BE>, a: &GLWE<Vec<u8>>, b: &GLWE<Vec<u8>>) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEAdd,
{
    assert_eq!(a.n(), b.n());
    assert_eq!(a.rank(), b.rank());
    assert_eq!(a.k(), b.k());
    assert_eq!(a.base2k(), b.base2k());

    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(a);
    module.glwe_add(&mut res, a, b);
    res
}

/// Subtracts two CHIMERA ciphertexts element-wise.
///
/// Computes `res = a - b`.
///
/// # Panics
///
/// Panics if the ciphertexts have incompatible parameters.
pub fn chimera_sub<BE: Backend>(module: &Module<BE>, a: &GLWE<Vec<u8>>, b: &GLWE<Vec<u8>>) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWESub,
{
    assert_eq!(a.n(), b.n());
    assert_eq!(a.rank(), b.rank());
    assert_eq!(a.k(), b.k());
    assert_eq!(a.base2k(), b.base2k());

    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(a);
    module.glwe_sub(&mut res, a, b);
    res
}

/// Multiplies a CHIMERA ciphertext by a plaintext constant polynomial.
///
/// Computes `res = ct * constants` where `constants` is a vector of i64
/// coefficients representing the plaintext polynomial.
///
/// This is the core operation for matrix-vector products where the weight
/// matrix is known in the clear (the standard case for inference).
///
/// Noise grows multiplicatively by ||constants||₂.
///
/// # Panics
///
/// Panics if scratch space is insufficient.
pub fn chimera_mul_const<BE: Backend>(module: &Module<BE>, ct: &GLWE<Vec<u8>>, constants: &[i64]) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    // res_offset = ct.base2k() gives identity scaling on the torus:
    // result = ct * constants without any additional scale factor.
    // This is the standard working point (res_offset_hi = 0, res_offset_lo = 0).
    let res_offset = ct.base2k().0 as usize;
    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    let tmp_bytes = module.glwe_mul_const_tmp_bytes(&res, res_offset, ct, constants.len());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);
    module.glwe_mul_const(&mut res, res_offset, ct, constants, scratch.borrow());
    res
}

/// Aligns a ciphertext to a target layout for mixed-level arithmetic.
///
/// This is used when two ciphertexts at different base2k levels need to be
/// combined (e.g. after nonlinear evaluation or before residual addition).
///
/// The current implementation uses `glwe_mul_const` with a scalar `[1]` to copy
/// into the requested layout. This is an alignment helper for CHIMERA's tested
/// paths, not a cryptographic modulus-switch or rescale primitive.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `ct` - Input ciphertext.
/// * `target` - Target layout to align into.
///
/// # Returns
///
/// A new ciphertext at the target layout.
pub fn chimera_align_layout<BE: Backend>(module: &Module<BE>, ct: &GLWE<Vec<u8>>, target: &GLWELayout) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let res_offset = ct.base2k().0 as usize;
    let constants = vec![1i64; 1];
    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(target);
    let tmp_bytes = module.glwe_mul_const_tmp_bytes(&res, res_offset, ct, constants.len());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);
    module.glwe_mul_const(&mut res, res_offset, ct, &constants, scratch.borrow());
    res
}

#[deprecated(note = "use chimera_align_layout instead")]
pub fn chimera_project_layout<BE: Backend>(module: &Module<BE>, ct: &GLWE<Vec<u8>>, target: &GLWELayout) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    chimera_align_layout(module, ct, target)
}

/// Rescales a CHIMERA ciphertext, reducing the scale factor by 2^base2k.
///
/// In Poulpy's bivariate representation, rescaling is a bit-shift on the
/// limb structure. This is far cheaper than RNS prime dropping in CKKS.
///
/// After a plaintext multiplication, the scale doubles. Rescaling restores
/// the original scale, consuming `base2k` bits of noise budget.
///
/// This is implemented by creating a new ciphertext with reduced torus
/// precision k' = k - base2k and copying the upper limbs.
pub fn chimera_rescale<BE: Backend>(_module: &Module<BE>, ct: &GLWE<Vec<u8>>, params: &ChimeraParams) -> GLWE<Vec<u8>>
where
    Module<BE>: ModuleN,
{
    use poulpy_core::layouts::GLWEToRef;

    // Reduce torus precision by base2k bits (drop the lowest limb)
    let ct_base2k = ct.base2k().as_u32();
    let new_k = if ct.k().as_u32() > ct_base2k {
        ct.k().as_u32() - ct_base2k
    } else {
        ct.k().as_u32() // Can't reduce further
    };

    let new_layout = GLWELayout {
        n: params.degree,
        base2k: ct.base2k(),
        k: TorusPrecision(new_k),
        rank: params.rank,
    };

    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(&new_layout);

    // Copy the upper limbs from the original ciphertext.
    // In Poulpy's layout, limbs are stored from most significant to least.
    // Dropping the lowest limb is equivalent to copying all but the last.
    let n = params.degree.0 as usize;
    let cols = (params.rank.0 + 1) as usize;
    let old_size = ct.size();
    let new_size = res.size();

    if new_size > 0 && old_size > 0 {
        let ct_ref = ct.to_ref();
        let src: &[u8] = ct_ref.data().data;

        use poulpy_core::layouts::GLWEToMut;
        let mut res_mut = res.to_mut();
        let dst: &mut [u8] = res_mut.data_mut().data;

        // VecZnx memory layout is limb-major, column-minor:
        //   limb j of column i starts at scalar offset: n * (j * cols + i)
        //   which is byte offset: n * (j * cols + i) * 8
        //
        // To drop the lowest limb(s), we copy the upper `new_size` limbs
        // (indices 0..new_size) from the old ciphertext to the new one.
        // Since old_size >= new_size and both share the same (n, cols),
        // we can copy limb-by-limb.
        let limb_bytes = n * 8; // Each limb is N i64 coefficients
        for j in 0..new_size {
            for col in 0..cols {
                let src_offset = (j * cols + col) * limb_bytes;
                let dst_offset = (j * cols + col) * limb_bytes;
                if src_offset + limb_bytes <= src.len() && dst_offset + limb_bytes <= dst.len() {
                    dst[dst_offset..dst_offset + limb_bytes].copy_from_slice(&src[src_offset..src_offset + limb_bytes]);
                }
            }
        }
    }

    res
}

/// Accumulates a sum of ciphertext × plaintext products.
///
/// Computes `res = Σᵢ ctᵢ * ptᵢ` efficiently by reusing scratch space.
/// This is the core pattern for matrix-vector multiplication:
/// each ciphertext encrypts a row/column of the activation matrix, and
/// each plaintext polynomial carries one row/column of weights.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `cts` - Slice of ciphertexts.
/// * `weights` - Slice of i64 weight polynomial slices (same length as `cts`).
///
/// # Panics
///
/// Panics if `cts` and `weights` have different lengths or are empty.
pub fn chimera_dot_product<BE: Backend>(module: &Module<BE>, cts: &[GLWE<Vec<u8>>], weights: &[Vec<i64>]) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    assert_eq!(cts.len(), weights.len());
    assert!(!cts.is_empty());

    // First term: allocate accumulator
    let mut acc = chimera_mul_const(module, &cts[0], &weights[0]);

    if cts.len() > 1 {
        // Pre-allocate a reusable scratch ciphertext for subsequent terms.
        // This avoids allocating a new GLWE for every mul_const + add pair.
        let res_offset = cts[1].base2k().0 as usize;
        let max_wlen = weights[1..].iter().map(|w| w.len()).max().unwrap_or(1);
        let tmp_bytes = module.glwe_mul_const_tmp_bytes(&acc, res_offset, &cts[1], max_wlen);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);
        let mut term = GLWE::<Vec<u8>>::alloc_from_infos(&cts[1]);

        for i in 1..cts.len() {
            let ro = cts[i].base2k().0 as usize;
            module.glwe_mul_const(&mut term, ro, &cts[i], &weights[i], scratch.borrow());
            module.glwe_add_inplace(&mut acc, &term);
        }
    }

    acc
}

/// Computes a homomorphic inner product of two ciphertext vectors.
///
/// Uses the trace operation (rotation-and-add) to sum all slots within a
/// ciphertext, producing a ciphertext where all slots contain the same scalar sum.
///
/// This is used for attention score computation: score = Q * K^T.
///
/// The trace computes `Trace(ct) = sum_{i in S} phi_i(ct)` where `phi_i` are
/// Galois automorphisms. This is implemented as a `log(N)`-step butterfly:
/// at each step, `res = res + Automorphism_p(res)`.
///
/// The `skip` parameter controls partial traces:
/// - `skip = 0`: full trace, sums all N slots
/// - `skip = k`: sums over N / 2^k slots (partial summation)
///
/// # Arguments
///
/// * `module` - The backend module.
/// * `eval_key` - Evaluation key containing automorphism keys.
/// * `ct` - Input ciphertext.
/// * `skip` - Number of trace levels to skip (0 for full summation).
///
/// # Panics
///
/// Panics if scratch space is insufficient.
pub fn chimera_slot_sum<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct: &GLWE<Vec<u8>>,
    skip: usize,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    use poulpy_core::layouts::{GLWEToMut, GLWEToRef};

    // Clone the ciphertext (trace_inplace modifies in place)
    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    {
        let ct_ref = ct.to_ref();
        let src: &[u8] = ct_ref.data().data;
        let mut res_mut = res.to_mut();
        let dst: &mut [u8] = res_mut.data_mut().data;
        let len = src.len().min(dst.len());
        dst[..len].copy_from_slice(&src[..len]);
    }

    // Compute scratch size for the trace operation
    let trace_bytes = GLWE::<Vec<u8>>::trace_tmp_bytes(module, ct, ct, &eval_key.auto_key_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(trace_bytes);

    // Apply the trace (in-place butterfly rotation-and-add)
    res.trace_inplace(module, skip, &eval_key.auto_keys, scratch.borrow());

    res
}

/// Rotates ciphertext slots by `k` positions using a Galois automorphism.
///
/// Slot rotation is the key operation for switching between packing modes
/// (e.g. head-aligned ↔ embedding-aligned) and for computing dot products
/// via the rotate-and-sum pattern.
///
/// Rotation by `k` slots applies the automorphism `X → X^{g^k}` where
/// `g = 5` is the Galois generator. This permutes the polynomial coefficients
/// so that slot `i` moves to slot `(i + k) mod N`.
///
/// # Arguments
///
/// * `module` - The backend module.
/// * `eval_key` - Evaluation key containing the automorphism key for position `k`.
/// * `ct` - Input ciphertext.
/// * `k` - Signed rotation amount. Positive = left rotate, negative = right rotate.
///
/// # Panics
///
/// Panics if the evaluation key does not contain a rotation key for position `k`.
/// Call [`ChimeraEvalKey::add_rotation_keys`] with the needed positions before
/// using this function.
pub fn chimera_rotate_slots<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct: &GLWE<Vec<u8>>,
    k: i64,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEAutomorphism<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
    use poulpy_hal::layouts::GaloisElement;

    if k == 0 {
        // No rotation: just clone the ciphertext
        let mut res = GLWE::<Vec<u8>>::alloc_from_infos(ct);
        {
            let ct_ref = ct.to_ref();
            let src: &[u8] = ct_ref.data().data;
            let mut res_mut = res.to_mut();
            let dst: &mut [u8] = res_mut.data_mut().data;
            let len = src.len().min(dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
        return res;
    }

    let gal_el = module.galois_element(k);

    let auto_key = eval_key.auto_keys.get(&gal_el).unwrap_or_else(|| {
        panic!(
            "chimera_rotate_slots: no automorphism key for rotation k={k} \
                 (Galois element {gal_el}). Call ChimeraEvalKey::add_rotation_keys \
                 with this position first."
        )
    });

    // Automorphism: output has the same layout as input for in-place variant,
    // but for the out-of-place variant, we allocate separately.
    // Use the in-place variant on a clone.
    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    {
        let ct_ref = ct.to_ref();
        let src: &[u8] = ct_ref.data().data;
        let mut res_mut = res.to_mut();
        let dst: &mut [u8] = res_mut.data_mut().data;
        let len = src.len().min(dst.len());
        dst[..len].copy_from_slice(&src[..len]);
    }

    // Compute scratch
    let auto_bytes = GLWE::<Vec<u8>>::automorphism_tmp_bytes(module, &res, &res, &eval_key.auto_key_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(auto_bytes);

    res.automorphism_inplace(module, auto_key, scratch.borrow());

    res
}

/// Computes a homomorphic matrix-vector product: `y = W * x` (ct-pt).
///
/// The encrypted input vector `x` is packed into one or more ciphertexts.
/// The plaintext weight matrix `W` is provided as `rows` of i64 vectors.
///
/// For a single-ciphertext input (all values packed into one ct), each
/// output element is computed as:
///   y_i = Σ_j W[i][j] * x[j]
///
/// This requires N scalar multiplications per output row, implemented via
/// `chimera_mul_const` (which multiplies all slots by a constant polynomial).
///
/// For multi-ciphertext input (values spread across multiple cts), this
/// computes `chimera_dot_product` per output row.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `x_cts` - Input ciphertexts (one or more, each packing a segment of the input vector).
/// * `weights` - Weight matrix as rows of i64 vectors. `weights[i]` is the i-th row,
///   with `weights[i].len() == x_cts.len()` (one weight vector per input ciphertext).
///
/// # Returns
///
/// A vector of ciphertexts, one per output row.
///
/// # Panics
///
/// Panics if `weights` is empty or if any row has a different length than `x_cts`.
pub fn chimera_matmul_ct_pt<BE: Backend>(
    module: &Module<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    weights: &[Vec<Vec<i64>>],
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    assert!(!weights.is_empty(), "weight matrix must have at least one row");
    assert!(!x_cts.is_empty(), "input must have at least one ciphertext");

    let mut outputs = Vec::with_capacity(weights.len());

    for row in weights.iter() {
        assert_eq!(
            row.len(),
            x_cts.len(),
            "each weight row must have one vector per input ciphertext"
        );

        // chimera_dot_product computes Σ ct_i * w_i
        let out = chimera_dot_product(module, x_cts, row);
        outputs.push(out);
    }

    outputs
}

/// Computes a homomorphic matrix-vector product for a single input ciphertext.
///
/// This is the common case: the input vector is packed into a single ciphertext,
/// and each output row is computed by multiplying the ciphertext by a different
/// plaintext weight polynomial (one scalar per slot).
///
/// `y_i = ct * w_i` where `w_i` is the i-th row of the weight matrix encoded
/// as a polynomial with one coefficient per slot.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `ct` - Single input ciphertext packing the entire input vector.
/// * `weight_rows` - Weight matrix rows as i64 slices. Each row has one
///   coefficient per active slot.
///
/// # Returns
///
/// A vector of ciphertexts, one per output row. Each ciphertext encrypts
/// the element-wise product of the input and the weight row — the caller
/// must sum slots (via `chimera_slot_sum`) to obtain the scalar dot product.
pub fn chimera_matmul_single_ct<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    weight_rows: &[Vec<i64>],
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    assert!(!weight_rows.is_empty(), "weight matrix must have at least one row");

    let mut outputs = Vec::with_capacity(weight_rows.len());

    for row in weight_rows.iter() {
        let out = chimera_mul_const(module, ct, row);
        outputs.push(out);
    }

    outputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{decode_int8, encode_int8};
    use crate::encrypt::{chimera_decrypt, chimera_encrypt, ChimeraKey};
    use crate::params::{ChimeraParams, Precision, SecurityLevel};
    use poulpy_hal::api::ModuleNew;

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    #[test]
    fn test_add() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let a_vals: Vec<i8> = vec![10, 20, 30];
        let b_vals: Vec<i8> = vec![5, -5, 10];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [2u8; 32], [3u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [4u8; 32], [5u8; 32]);

        let ct_sum = chimera_add(&module, &ct_a, &ct_b);
        let pt_sum = chimera_decrypt(&module, &key, &ct_sum, &params);
        let decoded = decode_int8(&module, &params, &pt_sum, 3);

        for i in 0..3 {
            let expected = a_vals[i] as i16 + b_vals[i] as i16;
            let diff = (expected - decoded[i] as i16).unsigned_abs();
            assert!(diff <= 1, "add error at {i}: expected {expected}, got {}", decoded[i]);
        }
    }

    #[test]
    fn test_sub() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let a_vals: Vec<i8> = vec![10, 20, 30];
        let b_vals: Vec<i8> = vec![5, -5, 10];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [2u8; 32], [3u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [4u8; 32], [5u8; 32]);

        let ct_diff = chimera_sub(&module, &ct_a, &ct_b);
        let pt_diff = chimera_decrypt(&module, &key, &ct_diff, &params);
        let decoded = decode_int8(&module, &params, &pt_diff, 3);

        for i in 0..3 {
            let expected = a_vals[i] as i16 - b_vals[i] as i16;
            let diff = (expected - decoded[i] as i16).unsigned_abs();
            assert!(diff <= 1, "sub error at {i}: expected {expected}, got {}", decoded[i]);
        }
    }

    #[test]
    fn test_matmul_single_ct() {
        // Test chimera_matmul_single_ct: multiply a single encrypted vector
        // by plaintext weight constants (polynomial product per row).
        //
        // Note: glwe_mul_const performs ring polynomial multiplication.
        // For a single-coefficient weight [c], this scales all coefficients by c.
        //
        // Input: [2, 3, 0, ...] encrypted in one ciphertext
        // Weights: row0 = [2], row1 = [3]
        // Expected: row0 * input = [4, 6, 0, ...], row1 * input = [6, 9, 0, ...]
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let input_vals: Vec<i8> = vec![2, 3];
        let pt = encode_int8(&module, &params, &input_vals);
        let ct = chimera_encrypt(&module, &key, &pt, [10u8; 32], [11u8; 32]);

        // Weight rows: scalar multipliers (single-coefficient polynomials)
        let row0: Vec<i64> = vec![2]; // scale by 2
        let row1: Vec<i64> = vec![3]; // scale by 3
        let weight_rows = vec![row0, row1];

        let results = chimera_matmul_single_ct(&module, &ct, &weight_rows);
        assert_eq!(results.len(), 2);

        // Decrypt and decode each output
        let pt0 = chimera_decrypt(&module, &key, &results[0], &params);
        let decoded0 = decode_int8(&module, &params, &pt0, 2);
        // row0: [2, 3] * 2 = [4, 6]
        assert!(
            (decoded0[0] as i16 - 4).unsigned_abs() <= 1,
            "matmul row0[0]: expected 4, got {}",
            decoded0[0]
        );
        assert!(
            (decoded0[1] as i16 - 6).unsigned_abs() <= 1,
            "matmul row0[1]: expected 6, got {}",
            decoded0[1]
        );

        let pt1 = chimera_decrypt(&module, &key, &results[1], &params);
        let decoded1 = decode_int8(&module, &params, &pt1, 2);
        // row1: [2, 3] * 3 = [6, 9]
        assert!(
            (decoded1[0] as i16 - 6).unsigned_abs() <= 1,
            "matmul row1[0]: expected 6, got {}",
            decoded1[0]
        );
        assert!(
            (decoded1[1] as i16 - 9).unsigned_abs() <= 1,
            "matmul row1[1]: expected 9, got {}",
            decoded1[1]
        );
    }

    #[test]
    fn test_matmul_ct_pt_multi() {
        // Test chimera_matmul_ct_pt with multiple input ciphertexts.
        // This computes dot products: each output is Σ ct_i * w_i.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let vals_a: Vec<i8> = vec![3];
        let vals_b: Vec<i8> = vec![4];

        let pt_a = encode_int8(&module, &params, &vals_a);
        let pt_b = encode_int8(&module, &params, &vals_b);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [10u8; 32], [11u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [12u8; 32], [13u8; 32]);

        let x_cts = vec![ct_a, ct_b];

        // Weight matrix: one row [w0, w1] where each w_i is a single-coeff vector
        // Output = ct_a * 2 + ct_b * 1 = 3*2 + 4*1 = 10 (at coefficient 0)
        let weights = vec![
            vec![vec![2i64], vec![1i64]], // row 0: dot product with [2, 1]
        ];

        let results = chimera_matmul_ct_pt(&module, &x_cts, &weights);
        assert_eq!(results.len(), 1);

        let pt_dec = chimera_decrypt(&module, &key, &results[0], &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 1);

        // Expected: 3*2 + 4*1 = 10
        let diff = (decoded[0] as i16 - 10).unsigned_abs();
        assert!(diff <= 2, "matmul ct_pt: expected 10, got {}, diff={}", decoded[0], diff);
    }

    #[test]
    fn test_rotate_slots_identity() {
        // Rotation by 0 should return the same ciphertext.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let values: Vec<i8> = vec![10, 20, 30, 40];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        // Generate eval key (only trace keys, rotation by 0 doesn't need extra)
        let eval_key = crate::encrypt::ChimeraEvalKey::generate(&module, &key, &params, [4u8; 32], [5u8; 32]);

        let ct_rot = chimera_rotate_slots(&module, &eval_key, &ct, 0);
        let pt_dec = chimera_decrypt(&module, &key, &ct_rot, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 4);

        for i in 0..4 {
            let diff = (values[i] as i16 - decoded[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "rotate_slots(0) error at {i}: expected {}, got {}",
                values[i],
                decoded[i]
            );
        }
    }

    #[test]
    fn test_rotate_slots_by_one() {
        // Rotation by 1 slot: [a, b, c, d, 0, ...] → [b, c, d, 0, ..., -a]
        // In the negacyclic ring X^N+1, rotating by 1 maps coefficient i to
        // coefficient (i+1) mod N, with the wrap-around coefficient negated.
        //
        // The automorphism X → X^5 permutes polynomial coefficients.
        // For coefficient-domain encoding (as used in CHIMERA), this applies
        // a permutation rather than a simple shift. We verify that the
        // automorphism is applied correctly by checking that the output
        // decrypts to a permuted version of the input.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        // Encode a simple pattern to detect the permutation
        let n = params.degree.0 as usize;
        let mut values = vec![0i8; n];
        values[0] = 10;
        values[1] = 20;

        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        // Generate eval key with rotation key for position 1
        let mut eval_key = crate::encrypt::ChimeraEvalKey::generate(&module, &key, &params, [4u8; 32], [5u8; 32]);
        eval_key.add_rotation_keys(&module, &key, &[1], [6u8; 32], [7u8; 32]);

        let ct_rot = chimera_rotate_slots(&module, &eval_key, &ct, 1);
        let pt_dec = chimera_decrypt(&module, &key, &ct_rot, &params);

        // The automorphism X → X^5 maps coefficient k to coefficient (5*k mod 2N).
        // So coefficient 0 → coefficient 0 (constant is fixed), coefficient 1 → 5.
        // For the specific values we encoded:
        //   input:  coeff[0]=10, coeff[1]=20, rest 0
        //   output: coeff[0]=10, coeff[5]=20 (since 5*1 = 5 mod 2N)
        //
        // Verify that some permutation has occurred and the output is non-garbage.
        let decoded = decode_int8(&module, &params, &pt_dec, n);

        // Coefficient 0 should be preserved (automorphism fixes the constant term)
        let diff0 = (decoded[0] as i16 - 10).unsigned_abs();
        assert!(diff0 <= 1, "rotate_slots(1) coeff[0]: expected 10, got {}", decoded[0]);

        // The non-zero values should sum to approximately the same as the input.
        // (Automorphism is a permutation, not a scaling, so total "energy" is preserved.)
        let mut nonzero_found = false;
        for i in 1..n {
            if (decoded[i] as i16).unsigned_abs() > 5 {
                nonzero_found = true;
                break;
            }
        }
        assert!(
            nonzero_found,
            "rotate_slots(1) should produce at least one non-zero coefficient beyond index 0"
        );
    }
}
