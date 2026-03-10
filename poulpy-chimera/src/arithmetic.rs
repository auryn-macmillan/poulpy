//! Homomorphic arithmetic operations for CHIMERA.
//!
//! Provides ciphertext addition, plaintext multiplication, rescaling, and
//! slot rotation — the four building blocks needed for all transformer
//! operations under FHE.

use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWESub, GLWETrace,
    layouts::{GLWE, GLWEInfos, GLWELayout, LWEInfos, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};
use poulpy_core::ScratchTakeCore;

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
pub fn chimera_add<BE: Backend>(
    module: &Module<BE>,
    a: &GLWE<Vec<u8>>,
    b: &GLWE<Vec<u8>>,
) -> GLWE<Vec<u8>>
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
pub fn chimera_sub<BE: Backend>(
    module: &Module<BE>,
    a: &GLWE<Vec<u8>>,
    b: &GLWE<Vec<u8>>,
) -> GLWE<Vec<u8>>
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
pub fn chimera_mul_const<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    constants: &[i64],
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    // res_offset = 2 * base2k prevents overflow from the product's
    // double-precision intermediate representation.
    let res_offset = 2 * ct.base2k().0 as usize;
    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    let tmp_bytes = module.glwe_mul_const_tmp_bytes(&res, res_offset, ct, constants.len());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);
    module.glwe_mul_const(&mut res, res_offset, ct, constants, scratch.borrow());
    res
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
pub fn chimera_rescale<BE: Backend>(
    _module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    params: &ChimeraParams,
) -> GLWE<Vec<u8>>
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
                if src_offset + limb_bytes <= src.len()
                    && dst_offset + limb_bytes <= dst.len()
                {
                    dst[dst_offset..dst_offset + limb_bytes]
                        .copy_from_slice(&src[src_offset..src_offset + limb_bytes]);
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
pub fn chimera_dot_product<BE: Backend>(
    module: &Module<BE>,
    cts: &[GLWE<Vec<u8>>],
    weights: &[Vec<i64>],
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    assert_eq!(cts.len(), weights.len());
    assert!(!cts.is_empty());

    let mut acc = chimera_mul_const(module, &cts[0], &weights[0]);

    for i in 1..cts.len() {
        let term = chimera_mul_const(module, &cts[i], &weights[i]);
        let sum = chimera_add(module, &acc, &term);
        acc = sum;
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
}
