//! RLWE encryption and decryption for CHIMERA.
//!
//! Wraps `poulpy-core`'s GLWE secret-key encryption with CHIMERA-specific
//! parameter handling, providing a simpler interface for encrypting and
//! decrypting plaintext vectors.

use std::collections::HashMap;

use crate::params::ChimeraParams;
use poulpy_core::ScratchTakeCore;
use poulpy_core::{
    layouts::{
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPrepared, GLWESecretPreparedFactory,
            GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory,
        },
        Dsize, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretTensorFactory,
        GLWETensorKey, GLWETensorKeyLayout, LWEInfos, GLWE,
    },
    GLWEAutomorphismKeyEncryptSk, GLWEEncryptSk, GLWETensorKeyEncryptSk, GLWETrace,
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
    source::Source,
};

/// Key material for CHIMERA encryption/decryption.
///
/// Holds both the raw secret and its DFT-prepared form for fast operations.
/// On drop, the raw secret is overwritten with zeros via the exposed Poulpy API.
/// The prepared secret remains a known limitation until poulpy-core exposes a
/// mutable buffer API for it.
///
pub struct ChimeraKey<BE: Backend> {
    /// The GLWE layout parameters.
    pub layout: GLWELayout,
    /// Raw GLWE secret key.
    pub secret: GLWESecret<Vec<u8>>,
    /// DFT-prepared secret for fast polynomial multiplication.
    pub prepared: GLWESecretPrepared<Vec<u8>, BE>,
}

impl<BE: Backend> Drop for ChimeraKey<BE> {
    fn drop(&mut self) {
        self.secret.fill_zero();
    }
}

impl<BE: Backend> ChimeraKey<BE>
where
    Module<BE>: ModuleN + GLWESecretPreparedFactory<BE>,
{
    /// Generates a new CHIMERA key pair from the given parameters.
    ///
    /// # Arguments
    ///
    /// * `module` - The backend module (ring arithmetic tables).
    /// * `params` - CHIMERA parameter set.
    /// * `seed` - 32-byte seed for deterministic key generation.
    ///
    /// # Note on base2k
    ///
    /// The key's GLWE layout uses `in_base2k = params.base2k - 1` for the
    /// ciphertext base2k, matching the tensor product parameter convention
    /// from poulpy-core. See [`ChimeraParams`] for details on the base2k cascade.
    pub fn generate(module: &Module<BE>, params: &ChimeraParams, seed: [u8; 32]) -> Self {
        let in_base2k = params.in_base2k();

        let layout = GLWELayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(in_base2k as u32),
            k: params.k_ct,
            rank: params.rank,
        };

        let mut source_xs = Source::new(seed);

        let mut secret = GLWESecret::<Vec<u8>>::alloc_from_infos(&layout);
        secret.fill_ternary_prob(0.5, &mut source_xs);

        let mut prepared = GLWESecretPrepared::<Vec<u8>, BE>::alloc(module, params.rank);
        prepared.prepare(module, &secret);

        ChimeraKey {
            layout,
            secret,
            prepared,
        }
    }
}

/// Evaluation key material for homomorphic operations that require
/// ciphertext-ciphertext multiplication (tensor product + relinearization)
/// and slot rotation/summation (automorphism keys for trace and rotation).
///
/// This includes polynomial activation evaluation, attention score computation,
/// and any other operation that multiplies two ciphertexts together.
///
/// ## Multi-level tensor keys
///
/// Deep computations like SwiGLU require chaining tensor products: the output
/// of one tensor product (at `out_base2k`) becomes the input to the next.
/// The primary tensor key handles fresh ciphertexts at `in_base2k`; the
/// secondary (level-2) tensor key handles ciphertexts at `out_base2k`.
pub struct ChimeraEvalKey<BE: Backend> {
    /// Prepared tensor key for relinearization after tensor product.
    pub tensor_key_prepared: GLWETensorKeyPrepared<Vec<u8>, BE>,
    /// Layout of the tensor key (for scratch size calculations).
    pub tensor_key_layout: GLWETensorKeyLayout,
    /// Layout of the output ciphertext after tensor product + relinearization.
    /// Has reduced base2k compared to the input.
    pub output_layout: GLWELayout,
    /// Layout of the tensor product intermediate result.
    pub tensor_layout: GLWELayout,
    /// The res_offset parameter for tensor_apply (= 2 * in_base2k).
    pub res_offset: usize,

    // --- Level-2 tensor key (for chained tensor products) ---
    /// Prepared level-2 tensor key for operations on post-tensor-product ciphertexts.
    /// Accepts inputs at `out_base2k` and produces outputs at `out_base2k - 1`.
    /// `None` if chained tensor products are not needed.
    pub tensor_key_l2_prepared: Option<GLWETensorKeyPrepared<Vec<u8>, BE>>,
    /// Layout of the level-2 tensor key.
    pub tensor_key_l2_layout: Option<GLWETensorKeyLayout>,
    /// Layout of the output after level-2 tensor product + relinearization.
    pub output_l2_layout: Option<GLWELayout>,
    /// Layout of the level-2 tensor product intermediate result.
    pub tensor_l2_layout: Option<GLWELayout>,
    /// The res_offset for level-2 tensor_apply (= 2 * out_base2k).
    pub res_offset_l2: Option<usize>,

    /// Prepared automorphism keys for slot rotation/summation (trace + rotation).
    /// Keyed by Galois element (e.g. -1, g^1, g^2, ...).
    /// Contains both trace Galois elements and any additional rotation keys.
    pub auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<Vec<u8>, BE>>,
    /// Layout of the automorphism keys.
    pub auto_key_layout: GLWEAutomorphismKeyLayout,
}

impl<BE: Backend> ChimeraEvalKey<BE>
where
    Module<BE>: ModuleN
        + GLWESecretPreparedFactory<BE>
        + GLWESecretTensorFactory<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    /// Generates evaluation keys from the secret key.
    ///
    /// Creates both:
    /// - A tensor key for ciphertext-ciphertext multiplication via
    ///   `glwe_tensor_apply` + `glwe_tensor_relinearize`.
    /// - Automorphism keys for all Galois elements required by the trace
    ///   operation (slot summation / rotation).
    ///
    /// # Parameters
    ///
    /// * `module` - The backend module.
    /// * `key` - The CHIMERA secret key.
    /// * `params` - CHIMERA parameter set.
    /// * `seed_a` - Seed for mask sampling.
    /// * `seed_e` - Seed for error sampling.
    pub fn generate(
        module: &Module<BE>,
        key: &ChimeraKey<BE>,
        params: &ChimeraParams,
        seed_a: [u8; 32],
        seed_e: [u8; 32],
    ) -> Self {
        let base2k = params.base2k.0 as usize;
        let k = params.k_ct.0 as usize;
        let rank = params.rank;

        // --- Tensor key generation ---

        // Following the parameter pattern from poulpy-core's tensor test:
        //   in_base2k = base2k - 1  (input ct base2k)
        //   out_base2k = base2k - 2 (output ct base2k after tensor product)
        //   tsk_base2k = base2k     (tensor key base2k)
        //   res_offset = 2 * in_base2k
        let in_base2k = if base2k > 1 { base2k - 1 } else { base2k };
        let out_base2k = if base2k > 2 { base2k - 2 } else { base2k };
        let tsk_base2k = base2k;

        let tsk_layout = GLWETensorKeyLayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(tsk_base2k as u32),
            k: poulpy_core::layouts::TorusPrecision((k + tsk_base2k) as u32),
            rank,
            dnum: poulpy_core::layouts::Dnum(k.div_ceil(tsk_base2k) as u32),
            dsize: Dsize(1),
        };

        let mut tsk = GLWETensorKey::<Vec<u8>>::alloc_from_infos(&tsk_layout);
        let mut source_xa = Source::new(seed_a);
        let mut source_xe = Source::new(seed_e);

        let encrypt_bytes = GLWETensorKey::encrypt_sk_tmp_bytes(module, &tsk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(encrypt_bytes);
        tsk.encrypt_sk(module, &key.secret, &mut source_xa, &mut source_xe, scratch.borrow());

        let mut tsk_prep = GLWETensorKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(module, &tsk_layout);
        let prep_bytes = tsk_prep.prepare_tmp_bytes(module, &tsk_layout);
        let mut prep_scratch: ScratchOwned<BE> = ScratchOwned::alloc(prep_bytes);
        tsk_prep.prepare(module, &tsk, prep_scratch.borrow());

        let output_layout = GLWELayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(out_base2k as u32),
            k: params.k_ct,
            rank,
        };

        let tensor_layout = GLWELayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(out_base2k as u32),
            k: params.k_ct,
            rank,
        };

        let res_offset = 2 * in_base2k;

        // --- Level-2 tensor key generation ---
        // For chained tensor products (e.g. SwiGLU: SiLU activation then ct*ct mul).
        // Level-2 accepts inputs at out_base2k and produces outputs at out_base2k - 1.
        let l2_in_base2k = out_base2k;
        let l2_out_base2k = if out_base2k > 1 { out_base2k - 1 } else { out_base2k };
        let l2_tsk_base2k = out_base2k + 1; // = in_base2k
        let l2_k = k; // same torus precision

        let tsk_l2_layout = GLWETensorKeyLayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(l2_tsk_base2k as u32),
            k: poulpy_core::layouts::TorusPrecision((l2_k + l2_tsk_base2k) as u32),
            rank,
            dnum: poulpy_core::layouts::Dnum(l2_k.div_ceil(l2_tsk_base2k) as u32),
            dsize: Dsize(1),
        };

        let mut tsk_l2 = GLWETensorKey::<Vec<u8>>::alloc_from_infos(&tsk_l2_layout);
        let l2_encrypt_bytes = GLWETensorKey::encrypt_sk_tmp_bytes(module, &tsk_l2_layout);
        let mut l2_scratch: ScratchOwned<BE> = ScratchOwned::alloc(l2_encrypt_bytes);
        tsk_l2.encrypt_sk(module, &key.secret, &mut source_xa, &mut source_xe, l2_scratch.borrow());

        let mut tsk_l2_prep = GLWETensorKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(module, &tsk_l2_layout);
        let l2_prep_bytes = tsk_l2_prep.prepare_tmp_bytes(module, &tsk_l2_layout);
        let mut l2_prep_scratch: ScratchOwned<BE> = ScratchOwned::alloc(l2_prep_bytes);
        tsk_l2_prep.prepare(module, &tsk_l2, l2_prep_scratch.borrow());

        let output_l2_layout = GLWELayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(l2_out_base2k as u32),
            k: params.k_ct,
            rank,
        };

        let tensor_l2_layout = GLWELayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(l2_out_base2k as u32),
            k: params.k_ct,
            rank,
        };

        let res_offset_l2 = 2 * l2_in_base2k;

        // --- Automorphism key generation ---
        // The automorphism key uses key_base2k = base2k - 1 (following the trace test pattern).
        let key_base2k = if base2k > 1 { base2k - 1 } else { base2k };
        let k_autokey = k + key_base2k;
        let dsize: usize = 1;
        let dnum = k.div_ceil(key_base2k * dsize);

        let auto_key_layout = GLWEAutomorphismKeyLayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(key_base2k as u32),
            k: poulpy_core::layouts::TorusPrecision(k_autokey as u32),
            rank,
            dsize: poulpy_core::layouts::Dsize(dsize as u32),
            dnum: poulpy_core::layouts::Dnum(dnum as u32),
        };

        // Get the Galois elements needed for the trace operation
        let gal_els: Vec<i64> = GLWE::<Vec<u8>>::trace_galois_elements(module);

        // Allocate scratch for automorphism key encryption + preparation
        let auto_encrypt_bytes = GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &auto_key_layout);
        let auto_scratch_size = auto_encrypt_bytes;
        let mut auto_scratch: ScratchOwned<BE> = ScratchOwned::alloc(auto_scratch_size);

        let mut auto_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<Vec<u8>, BE>> = HashMap::new();
        let mut tmp_key = GLWEAutomorphismKey::<Vec<u8>>::alloc_from_infos(&auto_key_layout);

        for &gal_el in &gal_els {
            tmp_key.encrypt_sk(
                module,
                gal_el,
                &key.secret,
                &mut source_xa,
                &mut source_xe,
                auto_scratch.borrow(),
            );
            let mut atk_prepared = GLWEAutomorphismKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(module, &tmp_key);
            atk_prepared.prepare(module, &tmp_key, auto_scratch.borrow());
            auto_keys.insert(gal_el, atk_prepared);
        }

        ChimeraEvalKey {
            tensor_key_prepared: tsk_prep,
            tensor_key_layout: tsk_layout,
            output_layout,
            tensor_layout,
            res_offset,
            tensor_key_l2_prepared: Some(tsk_l2_prep),
            tensor_key_l2_layout: Some(tsk_l2_layout),
            output_l2_layout: Some(output_l2_layout),
            tensor_l2_layout: Some(tensor_l2_layout),
            res_offset_l2: Some(res_offset_l2),
            auto_keys,
            auto_key_layout,
        }
    }

    /// Generates additional automorphism keys for slot rotation by the given
    /// positions, adding them to the existing `auto_keys` map.
    ///
    /// Each rotation position `k` requires an automorphism key for the Galois
    /// element `5^k mod 2N`. Positions that already have keys (e.g. from the
    /// trace set) are skipped.
    ///
    /// # Arguments
    ///
    /// * `module` - The backend module.
    /// * `key` - The CHIMERA secret key.
    /// * `rotation_positions` - Signed rotation amounts. Positive = left rotate,
    ///   negative = right rotate.
    /// * `seed_a` - Seed for mask sampling.
    /// * `seed_e` - Seed for error sampling.
    pub fn add_rotation_keys(
        &mut self,
        module: &Module<BE>,
        key: &ChimeraKey<BE>,
        rotation_positions: &[i64],
        seed_a: [u8; 32],
        seed_e: [u8; 32],
    ) {
        use poulpy_hal::layouts::GaloisElement;

        let mut source_xa = Source::new(seed_a);
        let mut source_xe = Source::new(seed_e);

        let auto_encrypt_bytes = GLWEAutomorphismKey::encrypt_sk_tmp_bytes(module, &self.auto_key_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(auto_encrypt_bytes);
        let mut tmp_key = GLWEAutomorphismKey::<Vec<u8>>::alloc_from_infos(&self.auto_key_layout);

        for &pos in rotation_positions {
            let gal_el = module.galois_element(pos);

            // Skip if we already have this key (e.g. from trace generation)
            if self.auto_keys.contains_key(&gal_el) {
                continue;
            }

            tmp_key.encrypt_sk(module, gal_el, &key.secret, &mut source_xa, &mut source_xe, scratch.borrow());
            let mut atk_prepared = GLWEAutomorphismKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(module, &tmp_key);
            atk_prepared.prepare(module, &tmp_key, scratch.borrow());
            self.auto_keys.insert(gal_el, atk_prepared);
        }
    }
}

/// Encrypts a GLWE plaintext under CHIMERA parameters.
///
/// Returns an owned GLWE ciphertext. Seeds for mask and error sampling
/// are derived from the provided seeds.
///
/// # Arguments
///
/// * `module` - The backend module.
/// * `key` - CHIMERA key material.
/// * `pt` - The plaintext to encrypt.
/// * `seed_a` - Seed for mask sampling.
/// * `seed_e` - Seed for error sampling.
pub fn chimera_encrypt<BE: Backend>(
    module: &Module<BE>,
    key: &ChimeraKey<BE>,
    pt: &GLWEPlaintext<Vec<u8>>,
    seed_a: [u8; 32],
    seed_e: [u8; 32],
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let mut ct = GLWE::<Vec<u8>>::alloc_from_infos(&key.layout);
    let mut source_xa = Source::new(seed_a);
    let mut source_xe = Source::new(seed_e);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWE::encrypt_sk_tmp_bytes(module, &key.layout));

    ct.encrypt_sk(module, pt, &key.prepared, &mut source_xa, &mut source_xe, scratch.borrow());

    ct
}

/// Decrypts a GLWE ciphertext under CHIMERA parameters.
///
/// Returns an owned GLWE plaintext. The plaintext layout is derived from
/// the ciphertext's actual layout (base2k, k), not from `params` directly,
/// to handle ciphertexts at different levels (input vs post-tensor-product).
///
/// # Arguments
///
/// * `module` - The backend module.
/// * `key` - CHIMERA key material.
/// * `ct` - The ciphertext to decrypt.
/// * `_params` - CHIMERA parameter set (reserved for future use).
pub fn chimera_decrypt<BE: Backend>(
    module: &Module<BE>,
    key: &ChimeraKey<BE>,
    ct: &GLWE<Vec<u8>>,
    _params: &ChimeraParams,
) -> GLWEPlaintext<Vec<u8>>
where
    Module<BE>: poulpy_core::GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    use poulpy_core::layouts::GLWEPlaintextLayout;

    // Use the ciphertext's actual layout so decryption works for both
    // input ciphertexts (in_base2k) and post-tensor-product ciphertexts (out_base2k).
    let pt_layout = GLWEPlaintextLayout {
        n: ct.n(),
        base2k: ct.base2k(),
        k: ct.k(),
    };

    let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&pt_layout);

    // Use a layout matching the ciphertext for scratch sizing.
    // The ciphertext may be at in_base2k (fresh encryption) or out_base2k
    // (after tensor product), so we derive scratch from the ct itself.
    let ct_layout = GLWELayout {
        n: ct.n(),
        base2k: ct.base2k(),
        k: ct.k(),
        rank: key.layout.rank,
    };
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(GLWE::decrypt_tmp_bytes(module, &ct_layout));

    ct.decrypt(module, &mut pt, &key.prepared, scratch.borrow());

    pt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{decode_int8, encode_int8};
    use crate::params::{ChimeraParams, Precision, SecurityLevel};
    use poulpy_hal::api::ModuleNew;

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    #[test]
    fn test_encrypt_decrypt_int8() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());

        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let values: Vec<i8> = vec![10, -20, 30, -40, 50];
        let pt = encode_int8(&module, &params, &values);

        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);
        let pt_dec = chimera_decrypt(&module, &key, &ct, &params);

        let decoded = decode_int8(&module, &params, &pt_dec, values.len());

        // Allow small error from encryption noise
        for (v, d) in values.iter().zip(decoded.iter()) {
            let diff = (*v as i16 - *d as i16).unsigned_abs();
            assert!(diff <= 1, "encrypt/decrypt error too large: {v} vs {d}");
        }
    }
}
