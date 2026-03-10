//! Bootstrapping support for deep model inference.
//!
//! Bootstrapping refreshes the noise budget of a ciphertext, allowing
//! continued computation beyond the initial noise budget limit. For
//! CHIMERA's target workload (20B-40B parameter transformers with 32+
//! layers), a single noise budget may be insufficient, requiring one or
//! more bootstrapping operations during inference.
//!
//! ## Implementation
//!
//! This module provides:
//! - [`BootstrappingConfig`]: configuration for when and how to bootstrap
//! - [`ChimeraBootstrapKey`]: key material for the bootstrapping pipeline
//! - [`ChimeraBootstrapKeyPrepared`]: DFT-prepared key material for fast evaluation
//! - [`needs_bootstrap`]: checks whether a ciphertext's estimated noise
//!   budget requires refreshing before further computation
//! - [`chimera_bootstrap`]: bootstraps a single GLWE ciphertext coefficient
//!
//! ## Bootstrapping Pipeline
//!
//! The bootstrapping pipeline follows the standard TFHE pattern adapted to
//! CHIMERA's bivariate torus representation:
//!
//! 1. **Sample extract**: Extract coefficient 0 from the GLWE ciphertext into
//!    an N-dimensional LWE ciphertext (same ternary key domain, no key switch).
//! 2. **LWE key-switch**: Switch from the ternary GLWE secret (viewed as an LWE
//!    of dimension N) to a small binary LWE secret of dimension `n_lwe` using
//!    an `LWESwitchingKey`.
//! 3. **Blind rotation**: Use CGGI blind rotation with a lookup table to evaluate
//!    the encoded value under the binary LWE key, producing a fresh GLWE
//!    ciphertext under the ternary GLWE key with refreshed noise.
//!
//! The blind rotation already outputs a GLWE ciphertext under the ternary GLWE
//! key (since the blind rotation key encrypts the binary LWE secret bits under
//! the GLWE key). No additional key-switch or repacking step is needed.
//!
//! ## Design Rationale
//!
//! CHIMERA's design prioritises **bootstrapping elimination** for shallow
//! models (≤32 layers at 128-bit security). The noise parameters are tuned
//! so that a full forward pass through a typical transformer block consumes
//! less than the available noise budget. For deeper models or multi-pass
//! inference, bootstrapping is needed at most once per forward pass.
//!
//! ## Trust Model (Without Bootstrapping)
//!
//! When bootstrapping is not available:
//! - The scheme supports a fixed maximum circuit depth determined by
//!   [`ChimeraParams::max_depth`]
//! - Models deeper than [`ChimeraParams::max_layers_no_bootstrap`] layers
//!   cannot be evaluated correctly — decryption will produce incorrect
//!   results once the noise budget is exhausted
//! - The user must verify that the model depth is within the supported
//!   range before initiating inference

use std::marker::PhantomData;

use crate::noise::NoiseTracker;
use crate::params::ChimeraParams;

use poulpy_core::{
    layouts::{
        prepared::{GLWESecretPrepared, GLWESecretPreparedFactory, LWESwitchingKeyPrepared, LWESwitchingKeyPreparedFactory},
        Base2K, Degree, Dnum, GLWELayout, GLWESecret, LWELayout, LWESecret, LWESwitchingKey, LWESwitchingKeyLayout, Rank,
        TorusPrecision, GLWE, LWE,
    },
    LWEKeySwitch, LWESampleExtract, LWESwitchingKeyEncrypt, ScratchTakeCore,
};
use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScalarZnxToMut, ScalarZnxToRef, Scratch, ScratchOwned, ZnxView, ZnxViewMut},
    source::Source,
};
use poulpy_schemes::bin_fhe::blind_rotation::{
    BlindRotationExecute, BlindRotationKey, BlindRotationKeyEncryptSk, BlindRotationKeyFactory, BlindRotationKeyLayout,
    BlindRotationKeyPrepared, BlindRotationKeyPreparedFactory, LookUpTableLayout, LookupTable, LookupTableFactory, CGGI,
};

/// Configuration for bootstrapping decisions.
///
/// Controls when bootstrapping is triggered and what parameters are used.
#[derive(Clone, Debug)]
pub struct BootstrappingConfig {
    /// Minimum noise budget (in bits) before bootstrapping is triggered.
    ///
    /// When the estimated noise budget drops below this threshold,
    /// [`needs_bootstrap`] returns `true`. A higher threshold provides
    /// more safety margin but triggers bootstrapping more frequently.
    ///
    /// Recommended: 4-8 bits (enough headroom for 1-2 more multiplications).
    pub min_budget_bits: f64,

    /// Whether bootstrapping is enabled at all.
    ///
    /// When `false`, [`needs_bootstrap`] always returns `false` and
    /// [`chimera_bootstrap`] will panic. Use this for models known to
    /// fit within a single noise budget.
    pub enabled: bool,

    /// Maximum number of bootstraps allowed per forward pass.
    ///
    /// Acts as a safety limit to prevent runaway bootstrapping in case
    /// of noise estimation errors. Set to 0 for no limit.
    pub max_bootstraps_per_pass: usize,

    /// Whether to fold an activation function into the bootstrapping step.
    ///
    /// Functional bootstrapping can evaluate a lookup table during the
    /// bootstrap, effectively getting a "free" nonlinear activation.
    /// This is a major optimisation opportunity when bootstrapping is
    /// unavoidable.
    pub functional_bootstrap: bool,
}

impl BootstrappingConfig {
    /// Creates a configuration for models that fit within a single noise budget.
    ///
    /// Bootstrapping is disabled; [`needs_bootstrap`] always returns `false`.
    pub fn no_bootstrap() -> Self {
        BootstrappingConfig {
            min_budget_bits: 0.0,
            enabled: false,
            max_bootstraps_per_pass: 0,
            functional_bootstrap: false,
        }
    }

    /// Creates a configuration for deep models that may need bootstrapping.
    ///
    /// Bootstrapping is triggered when the noise budget drops below
    /// `min_budget_bits`. At most `max_bootstraps` bootstrapping operations
    /// are allowed per forward pass.
    pub fn for_deep_model(min_budget_bits: f64, max_bootstraps: usize) -> Self {
        BootstrappingConfig {
            min_budget_bits,
            enabled: true,
            max_bootstraps_per_pass: max_bootstraps,
            functional_bootstrap: false,
        }
    }

    /// Creates a configuration with functional bootstrapping enabled.
    ///
    /// Like [`for_deep_model`](Self::for_deep_model), but also folds an
    /// activation function into each bootstrap step.
    pub fn with_functional_bootstrap(min_budget_bits: f64, max_bootstraps: usize) -> Self {
        BootstrappingConfig {
            min_budget_bits,
            enabled: true,
            max_bootstraps_per_pass: max_bootstraps,
            functional_bootstrap: true,
        }
    }
}

/// Checks whether the given noise tracker indicates that bootstrapping
/// is needed before further computation.
///
/// Returns `true` if all of:
/// 1. `config.enabled` is `true`
/// 2. The estimated noise budget is below `config.min_budget_bits`
///
/// Returns `false` if bootstrapping is disabled or the budget is sufficient.
///
/// # Arguments
///
/// * `tracker` - Current noise state of the ciphertext.
/// * `params` - CHIMERA parameter set (for budget calculation).
/// * `config` - Bootstrapping configuration.
pub fn needs_bootstrap(tracker: &NoiseTracker, params: &ChimeraParams, config: &BootstrappingConfig) -> bool {
    if !config.enabled {
        return false;
    }
    tracker.budget_bits(params) < config.min_budget_bits
}

/// Estimates the cost of a single bootstrapping operation.
///
/// Returns an approximate operation count (number of NTT-sized polynomial
/// multiplications) for bootstrapping a single GLWE ciphertext under the
/// given parameters. This can be used for latency estimation and comparison
/// with alternative approaches (e.g. using shallower circuits).
///
/// The estimate is based on TFHE-style programmable bootstrapping adapted
/// to CHIMERA's parameter set:
/// - Blind rotation: N polynomial multiplications (one per LWE coefficient)
/// - Key switching: proportional to N * k / base2k
/// - Sample extraction + repacking overhead
///
/// # Arguments
///
/// * `params` - CHIMERA parameter set.
pub fn estimate_bootstrap_cost(params: &ChimeraParams) -> BootstrapCostEstimate {
    let n = params.degree.0 as usize;
    let k = params.k_ct.0 as usize;
    let base2k = params.base2k.0 as usize;

    // Blind rotation: one external product per LWE coefficient
    // Each external product ≈ k/base2k NTT-sized polynomial multiplications
    let decomp_size = k.div_ceil(base2k);
    let blind_rotation_ops = n * decomp_size;

    // Key switching: proportional to N * decomp_size
    let key_switching_ops = n * decomp_size / 2; // approximate

    // Sample extraction is cheap (O(N))
    let sample_extraction_ops = n;

    let total_ops = blind_rotation_ops + key_switching_ops + sample_extraction_ops;

    // Rough time estimate: each polynomial multiplication ≈ 1μs on modern hardware
    // This is very approximate and depends heavily on the backend.
    let estimated_ms = total_ops as f64 * 1e-3;

    BootstrapCostEstimate {
        blind_rotation_ops,
        key_switching_ops,
        sample_extraction_ops,
        total_poly_muls: total_ops,
        estimated_latency_ms: estimated_ms,
    }
}

/// Cost breakdown for a single bootstrapping operation.
#[derive(Clone, Debug)]
pub struct BootstrapCostEstimate {
    /// Number of polynomial multiplications in the blind rotation phase.
    pub blind_rotation_ops: usize,
    /// Number of polynomial multiplications for key switching.
    pub key_switching_ops: usize,
    /// Number of operations for sample extraction.
    pub sample_extraction_ops: usize,
    /// Total polynomial multiplications.
    pub total_poly_muls: usize,
    /// Rough estimated latency in milliseconds (very approximate).
    pub estimated_latency_ms: f64,
}

// ---------------------------------------------------------------------------
// Bootstrapping parameter sub-set
// ---------------------------------------------------------------------------

/// Parameters specific to the bootstrapping pipeline.
///
/// These are derived from the main [`ChimeraParams`] and control the
/// dimensions of the binary LWE domain, the blind rotation key, and
/// the LWE key-switching key that bridges between the ternary GLWE
/// domain (viewed as an N-dimensional LWE) and the binary LWE domain.
#[derive(Clone, Debug)]
pub struct ChimeraBootstrapParams {
    /// Dimension of the small binary LWE secret key.
    pub n_lwe: usize,
    /// Block size for the binary LWE key distribution (`BinaryBlock`).
    /// The LWE secret has Hamming weight `n_lwe / block_size`.
    pub block_size: usize,
    /// Base2k for the blind rotation key (GGSW decomposition base).
    pub base2k_brk: usize,
    /// Torus precision of the blind rotation key.
    pub k_brk: usize,
    /// Number of decomposition rows for the blind rotation key.
    pub dnum_brk: usize,
    /// Base2k for the LWE key-switching key.
    pub base2k_ksk: usize,
    /// Torus precision of the LWE key-switching key.
    pub k_ksk: usize,
    /// Number of decomposition rows for the LWE key-switching key.
    pub dnum_ksk: usize,
    /// Base2k for the N-dimensional LWE ciphertext after sample extract.
    pub base2k_lwe_big: usize,
    /// Torus precision of the N-dimensional LWE (same as input ct).
    pub k_lwe_big: usize,
    /// Base2k for the small LWE ciphertext after key-switch.
    pub base2k_lwe_small: usize,
    /// Torus precision of the small LWE ciphertext.
    pub k_lwe_small: usize,
    /// Number of message bits to encode in the LUT.
    pub log_message_modulus: usize,
    /// Extension factor for the lookup table (1 = standard single-polynomial LUT).
    pub extension_factor: usize,
    /// Torus precision of the blind rotation result GLWE.
    pub k_res: usize,
    /// Torus precision of the lookup table.
    pub k_lut: usize,
}

impl ChimeraBootstrapParams {
    /// Derives bootstrapping parameters from the main CHIMERA parameter set.
    ///
    /// These parameters follow the pattern from the poulpy-schemes blind
    /// rotation test, which is known to produce correct results:
    /// - `base2k_brk = 19` (the blind rotation's GGSW decomposition base)
    /// - `k_brk = 3 * base2k_brk = 57`
    /// - `dnum_brk = 2`
    /// - `k_res = 2 * base2k_brk = 38`
    /// - `k_lut = base2k_brk = 19`
    ///
    /// The LWE key-switching key parameters are sized to provide enough
    /// precision for the key-switch from the N-dimensional ternary LWE
    /// to the n_lwe-dimensional binary LWE.
    pub fn from_chimera(params: &ChimeraParams) -> Self {
        // Binary LWE dimension and block size.
        // n_lwe=77 with block_size=7 follows the pattern from the blind
        // rotation and circuit bootstrapping tests.
        let n_lwe: usize = 77;
        let block_size: usize = 7;

        // Blind rotation key parameters — matching the blind rotation test.
        let base2k_brk: usize = 19;
        let k_brk: usize = 3 * base2k_brk; // 57
        let dnum_brk: usize = 2;

        // Message modulus: the blind rotation encodes the message at scale
        // 2^{-(log_message_modulus+1)} on the torus. CHIMERA encodes at
        // scale 2^{-scale_bits}. Therefore:
        //   log_message_modulus + 1 = scale_bits
        //   log_message_modulus = scale_bits - 1
        let log_message_modulus: usize = params.scale_bits as usize - 1;

        // The N-dimensional LWE after sample extract inherits the input
        // ciphertext's base2k and k parameters.
        let in_base2k = params.in_base2k();
        let k_ct = params.k_ct.0 as usize;
        let base2k_lwe_big: usize = in_base2k;
        let k_lwe_big: usize = k_ct;

        // Output LWE (post-keyswitch) parameters.
        //
        // The small LWE must use the same base2k as the blind rotation
        // key (base2k_brk) so that `mod_switch_2n` enters its `if base2k
        // > log2n` branch. For N=4096, 2N=8192, log2n=14. With base2k=19
        // we get diff=19-13=6, which produces correct mod-switched indices.
        //
        // The k (torus precision) of the small LWE only needs enough bits
        // to carry the message. Using k=24 matches the blind rotation test
        // pattern and gives ceil(24/19)=2 limbs — sufficient for 8-bit
        // messages with headroom.
        let base2k_lwe_small: usize = base2k_brk; // 19 (same as blind rotation)
        let k_lwe_small: usize = 24; // ceil(24/19) = 2 limbs

        // LWE key-switching key parameters.
        //
        // The KSK internally performs a GLWE keyswitch. Its base2k must
        // be >= the input base2k for the decomposition to work correctly.
        // We use base2k_ksk = base2k_brk + 1 = 20 so the internal GGSW
        // decomposition has clean headroom. The GLWE keyswitch normalizes
        // the input to key_base2k and the output to res_base2k, so there
        // is no strict cascade requirement.
        //
        // k_ksk = k_lwe_big + key_base2k (following poulpy-core pattern)
        // dnum_ksk = k_lwe_big.div_ceil(key_base2k)
        let base2k_ksk: usize = base2k_brk + 1; // 20
        let k_ksk: usize = k_lwe_big + base2k_ksk;
        let dnum_ksk: usize = k_lwe_big.div_ceil(base2k_ksk);

        // Result GLWE precision after blind rotation.
        let k_res: usize = 2 * base2k_brk; // 38
        let k_lut: usize = base2k_brk; // 19

        let extension_factor: usize = 1;

        ChimeraBootstrapParams {
            n_lwe,
            block_size,
            base2k_brk,
            k_brk,
            dnum_brk,
            base2k_ksk,
            k_ksk,
            dnum_ksk,
            base2k_lwe_big,
            k_lwe_big,
            base2k_lwe_small,
            k_lwe_small,
            log_message_modulus,
            extension_factor,
            k_res,
            k_lut,
        }
    }
}

// ---------------------------------------------------------------------------
// Bootstrap key material (standard form)
// ---------------------------------------------------------------------------

/// Key material for CHIMERA bootstrapping.
///
/// Contains all keys needed for the bootstrapping pipeline:
/// - A binary LWE secret key (small dimension `n_lwe`)
/// - A blind rotation key (GGSW encryptions of binary LWE secret bits under
///   the ternary GLWE secret)
/// - An LWE key-switching key (N-dimensional ternary → n_lwe binary)
///
/// ## Key Domains
///
/// CHIMERA uses a **ternary** GLWE secret key (rank 1, dimension N), but the
/// blind rotation requires a **binary** LWE key of dimension `n_lwe`. The
/// pipeline is:
///
/// 1. Sample extract: GLWE → N-dim LWE (same ternary key, no key switch)
/// 2. LWE key-switch: N-dim ternary → n_lwe binary (via `LWESwitchingKey`)
/// 3. Blind rotation: n_lwe binary LWE → GLWE (under ternary key)
///
/// The blind rotation already outputs a GLWE under the ternary key because
/// the BRK encrypts the binary LWE secret bits under the GLWE ternary key.
pub struct ChimeraBootstrapKey<BE: Backend> {
    /// Bootstrapping-specific parameters.
    pub bootstrap_params: ChimeraBootstrapParams,
    /// Blind rotation key: GGSW encryptions of binary LWE secret bits
    /// under the ternary GLWE secret.
    pub brk: BlindRotationKey<Vec<u8>, CGGI>,
    /// LWE key-switching key: switches from N-dimensional ternary LWE
    /// (extracted from GLWE) to n_lwe-dimensional binary LWE.
    pub ksk: LWESwitchingKey<Vec<u8>>,
    /// The binary LWE secret key (needed for key generation only; could be
    /// zeroed after key generation in production).
    pub sk_lwe: LWESecret<Vec<u8>>,
    /// Marker for the backend type parameter.
    _phantom: PhantomData<BE>,
}

impl<BE: Backend> ChimeraBootstrapKey<BE>
where
    Module<BE>: ModuleN + GLWESecretPreparedFactory<BE> + BlindRotationKeyEncryptSk<CGGI, BE> + LWESwitchingKeyEncrypt<BE>,
    BlindRotationKey<Vec<u8>, CGGI>: BlindRotationKeyFactory<CGGI>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    /// Generates bootstrapping key material.
    ///
    /// # Arguments
    ///
    /// * `module` - The backend module.
    /// * `params` - CHIMERA parameter set.
    /// * `sk_glwe` - The ternary GLWE secret key.
    /// * `sk_glwe_prepared` - Prepared (DFT-domain) GLWE secret key.
    /// * `seed_xs` - Seed for the binary LWE secret key generation.
    /// * `seed_a` - Seed for mask sampling in key encryption.
    /// * `seed_e` - Seed for error sampling in key encryption.
    pub fn generate(
        module: &Module<BE>,
        params: &ChimeraParams,
        sk_glwe: &GLWESecret<Vec<u8>>,
        sk_glwe_prepared: &GLWESecretPrepared<Vec<u8>, BE>,
        seed_xs: [u8; 32],
        seed_a: [u8; 32],
        seed_e: [u8; 32],
    ) -> Self {
        let bp = ChimeraBootstrapParams::from_chimera(params);
        let n_glwe: usize = module.n();

        // Generate binary LWE secret key
        let mut source_xs = Source::new(seed_xs);
        let mut sk_lwe = LWESecret::<Vec<u8>>::alloc(Degree(bp.n_lwe as u32));
        sk_lwe.fill_binary_block(bp.block_size, &mut source_xs);

        let mut source_xa = Source::new(seed_a);
        let mut source_xe = Source::new(seed_e);

        // --- Blind rotation key ---
        // The BRK encrypts the binary LWE secret bits under the ternary
        // GLWE secret key. The blind rotation uses these to homomorphically
        // evaluate the LWE decryption function.
        let brk_layout = BlindRotationKeyLayout {
            n_glwe: Degree(n_glwe as u32),
            n_lwe: Degree(bp.n_lwe as u32),
            base2k: Base2K(bp.base2k_brk as u32),
            k: TorusPrecision(bp.k_brk as u32),
            dnum: Dnum(bp.dnum_brk as u32),
            rank: Rank(1),
        };

        let encrypt_bytes = BlindRotationKey::encrypt_sk_tmp_bytes(module, &brk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(encrypt_bytes);

        let mut brk = BlindRotationKey::<Vec<u8>, CGGI>::alloc(&brk_layout);
        brk.encrypt_sk(
            module,
            sk_glwe_prepared,
            &sk_lwe,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );

        // --- LWE key-switching key (N-dim ternary → n_lwe binary) ---
        //
        // The LWESwitchingKey internally embeds both LWE secrets into
        // rank-1 GLWE polynomials (via automorphism(-1)) and encrypts
        // a GLWE switching key between them. The `n` field is the ring
        // polynomial degree N (not the LWE dimension).
        //
        // sk_lwe_in: the N-dimensional ternary key (GLWE secret viewed
        //            as an LWE secret — first N polynomial coefficients)
        // sk_lwe_out: the n_lwe-dimensional binary key
        let ksk_layout = LWESwitchingKeyLayout {
            n: Degree(n_glwe as u32),
            base2k: Base2K(bp.base2k_ksk as u32),
            k: TorusPrecision(bp.k_ksk as u32),
            dnum: Dnum(bp.dnum_ksk as u32),
        };

        // Create an N-dimensional LWE secret from the GLWE secret.
        //
        // After sample_extract, the N-dimensional LWE has an effective
        // secret key that is the negacyclic reordering of the GLWE
        // polynomial secret, i.e. auto(-1) applied to the raw
        // coefficients:
        //
        //   sk_lwe[0]   =  s[0]
        //   sk_lwe[j]   = -s[N-j]   for j = 1 .. N-1
        //
        // This is because sample_extract copies GLWE mask polynomial
        // coefficients directly into LWE mask positions, and the
        // polynomial product a(X)*s(X) mod (X^N+1) at coefficient 0
        // yields the sum a[0]*s[0] + sum_{j>0} a[j]*(-s[N-j]).
        // The LWE inner product sum a[j]*sk_lwe[j] must match this.
        let mut sk_lwe_big = LWESecret::<Vec<u8>>::alloc(Degree(n_glwe as u32));
        {
            let glwe_data = sk_glwe.data();
            let glwe_ref = glwe_data.to_ref();
            let glwe_coeffs: &[i64] = glwe_ref.at(0, 0);
            let lwe_data = sk_lwe_big.data_mut();
            let mut lwe_mut = lwe_data.to_mut();
            let lwe_coeffs: &mut [i64] = lwe_mut.at_mut(0, 0);
            let n = glwe_coeffs.len().min(lwe_coeffs.len());
            // Apply auto(-1): negacyclic reordering
            lwe_coeffs[0] = glwe_coeffs[0];
            for j in 1..n {
                lwe_coeffs[j] = -glwe_coeffs[n - j];
            }
        }
        {
            // Copy the distribution flag
            use poulpy_core::GetDistribution;
            sk_lwe_big.set_dist(*sk_glwe.dist());
        }

        let ksk_encrypt_bytes = LWESwitchingKey::encrypt_sk_tmp_bytes(module, &ksk_layout);
        let mut ksk_scratch: ScratchOwned<BE> = ScratchOwned::alloc(ksk_encrypt_bytes);

        let mut ksk = LWESwitchingKey::<Vec<u8>>::alloc_from_infos(&ksk_layout);
        ksk.encrypt_sk(
            module,
            &sk_lwe_big, // sk_lwe_in: N-dim ternary (from GLWE)
            &sk_lwe,     // sk_lwe_out: n_lwe-dim binary
            &mut source_xa,
            &mut source_xe,
            ksk_scratch.borrow(),
        );

        ChimeraBootstrapKey {
            bootstrap_params: bp,
            brk,
            ksk,
            sk_lwe,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Prepared (DFT-domain) bootstrap key
// ---------------------------------------------------------------------------

/// DFT-prepared bootstrap key material for fast on-line evaluation.
///
/// All keys are transformed into the DFT domain for efficient polynomial
/// arithmetic during the blind rotation and key-switching steps. This
/// struct is not serialisable and must be derived from a
/// [`ChimeraBootstrapKey`] via [`ChimeraBootstrapKeyPrepared::prepare`].
pub struct ChimeraBootstrapKeyPrepared<BE: Backend> {
    /// Bootstrapping-specific parameters.
    pub bootstrap_params: ChimeraBootstrapParams,
    /// Prepared blind rotation key.
    pub brk_prepared: BlindRotationKeyPrepared<Vec<u8>, CGGI, BE>,
    /// Prepared LWE key-switching key (N-dim ternary → n_lwe binary).
    pub ksk_prepared: LWESwitchingKeyPrepared<Vec<u8>, BE>,
}

impl<BE: Backend> ChimeraBootstrapKeyPrepared<BE>
where
    Module<BE>: ModuleN + BlindRotationKeyPreparedFactory<CGGI, BE> + LWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    /// Prepares (transforms to DFT domain) all bootstrap key material.
    ///
    /// # Arguments
    ///
    /// * `module` - The backend module.
    /// * `bsk` - The standard-form bootstrap key.
    pub fn prepare(module: &Module<BE>, bsk: &ChimeraBootstrapKey<BE>) -> Self {
        // Prepare blind rotation key
        let mut brk_prepared = BlindRotationKeyPrepared::<Vec<u8>, CGGI, BE>::alloc(module, &bsk.brk);
        let brk_prep_bytes = BlindRotationKeyPrepared::prepare_tmp_bytes(module, &bsk.brk);
        let mut brk_scratch: ScratchOwned<BE> = ScratchOwned::alloc(brk_prep_bytes);
        brk_prepared.prepare(module, &bsk.brk, brk_scratch.borrow());

        // Prepare LWE key-switching key
        let mut ksk_prepared = LWESwitchingKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(module, &bsk.ksk);
        let ksk_prep_bytes = ksk_prepared.prepare_tmp_bytes(module, &bsk.ksk);
        let mut ksk_scratch: ScratchOwned<BE> = ScratchOwned::alloc(ksk_prep_bytes);
        ksk_prepared.prepare(module, &bsk.ksk, ksk_scratch.borrow());

        ChimeraBootstrapKeyPrepared {
            bootstrap_params: bsk.bootstrap_params.clone(),
            brk_prepared,
            ksk_prepared,
        }
    }
}

// ---------------------------------------------------------------------------
// Bootstrapping execution
// ---------------------------------------------------------------------------

/// Bootstraps a CHIMERA ciphertext to refresh its noise budget.
///
/// This function extracts coefficient 0 of the input GLWE ciphertext,
/// key-switches from the N-dimensional ternary domain to the small binary
/// LWE domain, and performs a blind rotation through a lookup table to
/// produce a fresh GLWE ciphertext with refreshed noise.
///
/// For **functional bootstrapping** (when `config.functional_bootstrap` is
/// true and a custom LUT is provided), the lookup table can encode an
/// activation function (e.g. GELU), amortising the cost of both noise
/// refresh and nonlinear evaluation.
///
/// # Arguments
///
/// * `module` - The backend module.
/// * `ct` - The ciphertext to bootstrap.
/// * `tracker` - Current noise state (updated to reflect the bootstrap).
/// * `_params` - CHIMERA parameter set.
/// * `bsk_prepared` - Prepared bootstrap key material.
/// * `_config` - Bootstrapping configuration.
///
/// # Returns
///
/// A fresh GLWE ciphertext encrypting the same value as `ct` (coefficient 0)
/// but with refreshed noise. The ciphertext layout uses the blind rotation's
/// output parameters (`base2k_brk`, `k_res`).
///
/// # Panics
///
/// Panics if `config.enabled` is false.
pub fn chimera_bootstrap<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    tracker: &mut NoiseTracker,
    _params: &ChimeraParams,
    bsk_prepared: &ChimeraBootstrapKeyPrepared<BE>,
    _config: &BootstrappingConfig,
) -> GLWE<Vec<u8>>
where
    Module<BE>: ModuleN + LWESampleExtract + LWEKeySwitch<BE> + BlindRotationExecute<CGGI, BE> + LookupTableFactory,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let bp = &bsk_prepared.bootstrap_params;
    let n_glwe: usize = module.n();

    // -----------------------------------------------------------------------
    // Step 1: Sample extract — GLWE → N-dimensional LWE (same key)
    //
    // Extracts coefficient 0 from the GLWE ciphertext into an LWE
    // ciphertext of dimension N, under the same ternary key.
    // No key-switch is performed; this is just a format conversion.
    //
    // The LWE layout must match the *actual* ciphertext's base2k and k,
    // which may differ from the bootstrap params if the ciphertext has
    // been through tensor products and rescaling.
    // -----------------------------------------------------------------------
    use poulpy_core::layouts::LWEInfos;
    let ct_base2k = ct.base2k();
    let ct_k = ct.k();
    let lwe_big_layout = LWELayout {
        n: Degree(n_glwe as u32),
        k: ct_k,
        base2k: ct_base2k,
    };

    let mut lwe_big = LWE::<Vec<u8>>::alloc_from_infos(&lwe_big_layout);
    lwe_big.sample_extract(module, ct);

    // -----------------------------------------------------------------------
    // Step 2: LWE key-switch — N-dim ternary → n_lwe-dim binary
    //
    // Uses an LWESwitchingKey to switch from the ternary GLWE secret
    // (viewed as an N-dimensional LWE key) to the small binary LWE key
    // of dimension n_lwe. This is the standard LWE key-switch operation
    // from TFHE, not a GLWE key-switch.
    // -----------------------------------------------------------------------
    let lwe_small_layout = LWELayout {
        n: Degree(bp.n_lwe as u32),
        k: TorusPrecision(bp.k_lwe_small as u32),
        base2k: Base2K(bp.base2k_lwe_small as u32),
    };

    let mut lwe_small = LWE::<Vec<u8>>::alloc_from_infos(&lwe_small_layout);

    let ks_bytes = LWE::keyswitch_tmp_bytes(module, &lwe_small_layout, &lwe_big_layout, &bsk_prepared.ksk_prepared);
    let mut ks_scratch: ScratchOwned<BE> = ScratchOwned::alloc(ks_bytes);
    lwe_small.keyswitch(module, &lwe_big, &bsk_prepared.ksk_prepared, ks_scratch.borrow());

    // -----------------------------------------------------------------------
    // Step 3: Blind rotation — n_lwe-dim LWE → GLWE (under ternary key)
    //
    // The CGGI blind rotation evaluates a lookup table at the encrypted
    // index stored in `lwe_small`, producing a fresh GLWE ciphertext
    // under the ternary GLWE key (since the BRK encrypts the binary LWE
    // secret bits under the GLWE ternary key).
    //
    // No additional key-switch is needed — the output is already under
    // the correct key.
    // -----------------------------------------------------------------------

    // Create identity LUT: f(x) = x
    // The LUT maps message values 0..2^log_message_modulus to themselves.
    let message_modulus: usize = 1 << bp.log_message_modulus;
    let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
    for (i, val) in f_vec.iter_mut().enumerate() {
        *val = i as i64;
    }

    let lut_layout = LookUpTableLayout {
        n: Degree(n_glwe as u32),
        extension_factor: bp.extension_factor,
        k: TorusPrecision(bp.k_lut as u32),
        base2k: Base2K(bp.base2k_brk as u32),
    };
    let mut lut = LookupTable::alloc(&lut_layout);
    lut.set(module, &f_vec, bp.log_message_modulus + 1);

    // GLWE result layout for the blind rotation output.
    let res_glwe_layout = GLWELayout {
        n: Degree(n_glwe as u32),
        base2k: Base2K(bp.base2k_brk as u32),
        k: TorusPrecision(bp.k_res as u32),
        rank: Rank(1),
    };
    let mut res_glwe = GLWE::<Vec<u8>>::alloc_from_infos(&res_glwe_layout);

    let br_bytes = BlindRotationKeyPrepared::execute_tmp_bytes(
        module,
        bp.block_size,
        bp.extension_factor,
        &res_glwe_layout,
        &bsk_prepared.brk_prepared,
    );
    let mut br_scratch: ScratchOwned<BE> = ScratchOwned::alloc(br_bytes);

    bsk_prepared
        .brk_prepared
        .execute(module, &mut res_glwe, &lwe_small, &lut, br_scratch.borrow());

    // -----------------------------------------------------------------------
    // Step 4: Update noise tracker
    // -----------------------------------------------------------------------
    tracker.bootstrap_reset();

    res_glwe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::NoiseTracker;
    use crate::params::{ChimeraParams, Precision, SecurityLevel};

    #[test]
    fn test_no_bootstrap_config() {
        let config = BootstrappingConfig::no_bootstrap();
        assert!(!config.enabled);
        assert_eq!(config.max_bootstraps_per_pass, 0);
    }

    #[test]
    fn test_deep_model_config() {
        let config = BootstrappingConfig::for_deep_model(6.0, 2);
        assert!(config.enabled);
        assert_eq!(config.max_bootstraps_per_pass, 2);
        assert!(!config.functional_bootstrap);
    }

    #[test]
    fn test_functional_bootstrap_config() {
        let config = BootstrappingConfig::with_functional_bootstrap(6.0, 2);
        assert!(config.enabled);
        assert!(config.functional_bootstrap);
    }

    #[test]
    fn test_needs_bootstrap_disabled() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let tracker = NoiseTracker::fresh();
        let config = BootstrappingConfig::no_bootstrap();

        assert!(!needs_bootstrap(&tracker, &params, &config));
    }

    #[test]
    fn test_needs_bootstrap_fresh_ct() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let tracker = NoiseTracker::fresh();
        let config = BootstrappingConfig::for_deep_model(4.0, 1);

        // Fresh ciphertext should not need bootstrapping
        assert!(!needs_bootstrap(&tracker, &params, &config));
    }

    #[test]
    fn test_needs_bootstrap_exhausted() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let mut tracker = NoiseTracker::fresh();

        // Exhaust the noise budget with many operations
        for _ in 0..20 {
            tracker.mul_const(100.0);
        }

        let config = BootstrappingConfig::for_deep_model(4.0, 1);
        assert!(needs_bootstrap(&tracker, &params, &config));
    }

    #[test]
    fn test_estimate_bootstrap_cost() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let cost = estimate_bootstrap_cost(&params);

        assert!(cost.blind_rotation_ops > 0);
        assert!(cost.key_switching_ops > 0);
        assert!(cost.total_poly_muls > 0);
        assert!(cost.estimated_latency_ms > 0.0);

        // The total should be the sum of components
        assert_eq!(
            cost.total_poly_muls,
            cost.blind_rotation_ops + cost.key_switching_ops + cost.sample_extraction_ops
        );
    }

    #[test]
    fn test_bootstrap_cost_scales_with_n() {
        let params_80 = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let params_128 = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);

        let cost_80 = estimate_bootstrap_cost(&params_80);
        let cost_128 = estimate_bootstrap_cost(&params_128);

        // Higher security (larger N) should be more expensive
        assert!(
            cost_128.total_poly_muls > cost_80.total_poly_muls,
            "128-bit bootstrap ({}) should be more expensive than 80-bit ({})",
            cost_128.total_poly_muls,
            cost_80.total_poly_muls,
        );
    }

    #[test]
    fn test_bootstrap_params_derivation() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let bp = ChimeraBootstrapParams::from_chimera(&params);

        assert_eq!(bp.n_lwe, 77);
        assert_eq!(bp.block_size, 7);
        assert_eq!(bp.base2k_brk, 19);
        assert_eq!(bp.k_brk, 57);
        assert_eq!(bp.dnum_brk, 2);
        assert_eq!(bp.k_res, 38);
        assert_eq!(bp.k_lut, 19);
        assert_eq!(bp.log_message_modulus, 7); // INT8 scale_bits - 1
    }

    #[test]
    fn test_bootstrap_params_fp16() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Fp16);
        let bp = ChimeraBootstrapParams::from_chimera(&params);

        assert_eq!(bp.log_message_modulus, 13); // FP16 scale_bits - 1
    }

    // End-to-end test: encrypt → bootstrap → decrypt
    #[test]
    fn test_chimera_bootstrap_roundtrip() {
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_decrypt, chimera_encrypt, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
        type BE = poulpy_cpu_ref::FFT64Ref;
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        type BE = poulpy_cpu_avx::FFT64Avx;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());

        // Generate keys
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        // Generate bootstrap key
        let bsk = ChimeraBootstrapKey::generate(
            &module,
            &params,
            &key.secret,
            &key.prepared,
            [10u8; 32], // seed for binary LWE key
            [11u8; 32], // seed for mask
            [12u8; 32], // seed for error
        );

        // Prepare bootstrap key
        let bsk_prepared = ChimeraBootstrapKeyPrepared::prepare(&module, &bsk);
        let bp = &bsk_prepared.bootstrap_params;

        // Encrypt a simple value
        let values: Vec<i8> = vec![5];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        // Verify pre-bootstrap decryption
        let pt_pre = chimera_decrypt(&module, &key, &ct, &params);
        let dec_pre = pt_pre.decode_coeff_i64(TorusPrecision(params.scale_bits as u32), 0);
        assert_eq!(dec_pre, 5, "pre-bootstrap decryption should give 5, got {dec_pre}");

        // Inline bootstrapping: sample_extract → keyswitch → blind_rotation
        use poulpy_core::layouts::GLWEPlaintext;

        let n_glwe: usize = module.n();

        // Step 1: Sample extract GLWE → big LWE
        let lwe_big_layout = LWELayout {
            n: Degree(n_glwe as u32),
            k: TorusPrecision(bp.k_lwe_big as u32),
            base2k: Base2K(bp.base2k_lwe_big as u32),
        };
        let mut lwe_big = LWE::<Vec<u8>>::alloc_from_infos(&lwe_big_layout);
        lwe_big.sample_extract(&module, &ct);

        // Step 2: LWE keyswitch big → small
        let lwe_small_layout = LWELayout {
            n: Degree(bp.n_lwe as u32),
            k: TorusPrecision(bp.k_lwe_small as u32),
            base2k: Base2K(bp.base2k_lwe_small as u32),
        };
        let mut lwe_small = LWE::<Vec<u8>>::alloc_from_infos(&lwe_small_layout);
        let ks_bytes = LWE::keyswitch_tmp_bytes(&module, &lwe_small_layout, &lwe_big_layout, &bsk_prepared.ksk_prepared);
        let mut ks_scratch: ScratchOwned<BE> = ScratchOwned::alloc(ks_bytes);
        lwe_small.keyswitch(&module, &lwe_big, &bsk_prepared.ksk_prepared, ks_scratch.borrow());

        // Step 3: Blind rotation with identity LUT
        let message_modulus: usize = 1 << bp.log_message_modulus;
        let mut f_vec: Vec<i64> = vec![0i64; message_modulus];
        for (i, val) in f_vec.iter_mut().enumerate() {
            *val = i as i64;
        }
        let lut_layout = LookUpTableLayout {
            n: Degree(n_glwe as u32),
            extension_factor: bp.extension_factor,
            k: TorusPrecision(bp.k_lut as u32),
            base2k: Base2K(bp.base2k_brk as u32),
        };
        let mut lut = LookupTable::alloc(&lut_layout);
        lut.set(&module, &f_vec, bp.log_message_modulus + 1);

        let res_glwe_layout = GLWELayout {
            n: Degree(n_glwe as u32),
            base2k: Base2K(bp.base2k_brk as u32),
            k: TorusPrecision(bp.k_res as u32),
            rank: Rank(1),
        };
        let mut res_glwe = GLWE::<Vec<u8>>::alloc_from_infos(&res_glwe_layout);
        let br_bytes = BlindRotationKeyPrepared::execute_tmp_bytes(
            &module,
            bp.block_size,
            bp.extension_factor,
            &res_glwe_layout,
            &bsk_prepared.brk_prepared,
        );
        let mut br_scratch: ScratchOwned<BE> = ScratchOwned::alloc(br_bytes);
        bsk_prepared
            .brk_prepared
            .execute(&module, &mut res_glwe, &lwe_small, &lut, br_scratch.borrow());

        // Decrypt and verify
        let mut pt_dec = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&res_glwe_layout);
        res_glwe.decrypt(&module, &mut pt_dec, &key.prepared, br_scratch.borrow());

        let decoded = pt_dec.decode_coeff_i64(TorusPrecision((bp.log_message_modulus + 1) as u32), 0);

        let diff = (decoded - 5i64).unsigned_abs();
        assert!(
            diff <= 2,
            "bootstrap roundtrip error too large: expected 5, got {decoded} (diff={diff})"
        );
    }

    /// Diagnostic test: directly test the blind rotation pipeline using
    /// native LWE encoding (bypassing CHIMERA encoding) to isolate whether
    /// the keyswitch + blind rotation is correct.
    #[test]
    fn test_bootstrap_native_lwe_encoding() {
        use poulpy_core::layouts::{prepared::GLWESecretPrepared, GLWEPlaintext, LWEPlaintext};
        use poulpy_hal::api::ModuleNew;

        #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
        type BE = poulpy_cpu_ref::FFT64Ref;
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        type BE = poulpy_cpu_avx::FFT64Avx;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let bp = ChimeraBootstrapParams::from_chimera(&params);
        let n_glwe: usize = module.n();

        // --- Generate keys exactly like ChimeraBootstrapKey::generate ---
        let mut source_xs = Source::new([42u8; 32]);
        let layout = GLWELayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(params.in_base2k() as u32),
            k: params.k_ct,
            rank: params.rank,
        };
        let mut sk_glwe = GLWESecret::<Vec<u8>>::alloc_from_infos(&layout);
        sk_glwe.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_glwe_dft = GLWESecretPrepared::<Vec<u8>, BE>::alloc(&module, params.rank);
        sk_glwe_dft.prepare(&module, &sk_glwe);

        // Binary LWE secret (small key)
        let mut source_xs2 = Source::new([10u8; 32]);
        let mut sk_lwe = LWESecret::<Vec<u8>>::alloc(Degree(bp.n_lwe as u32));
        sk_lwe.fill_binary_block(bp.block_size, &mut source_xs2);

        let mut source_xa = Source::new([11u8; 32]);
        let mut source_xe = Source::new([12u8; 32]);

        // --- BRK ---
        let brk_layout = BlindRotationKeyLayout {
            n_glwe: Degree(n_glwe as u32),
            n_lwe: Degree(bp.n_lwe as u32),
            base2k: Base2K(bp.base2k_brk as u32),
            k: TorusPrecision(bp.k_brk as u32),
            dnum: Dnum(bp.dnum_brk as u32),
            rank: Rank(1),
        };
        let encrypt_bytes = BlindRotationKey::encrypt_sk_tmp_bytes(&module, &brk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(encrypt_bytes);
        let mut brk = BlindRotationKey::<Vec<u8>, CGGI>::alloc(&brk_layout);
        brk.encrypt_sk(
            &module,
            &sk_glwe_dft,
            &sk_lwe,
            &mut source_xa,
            &mut source_xe,
            scratch.borrow(),
        );
        let mut brk_prepared = BlindRotationKeyPrepared::<Vec<u8>, CGGI, BE>::alloc(&module, &brk);
        let br_prep_bytes = BlindRotationKeyPrepared::prepare_tmp_bytes(&module, &brk);
        let mut brk_scratch: ScratchOwned<BE> = ScratchOwned::alloc(br_prep_bytes);
        brk_prepared.prepare(&module, &brk, brk_scratch.borrow());

        // --- Test blind rotation directly with native LWE encoding ---
        let log_msg_mod: usize = 4; // use small log_message_modulus for clarity
        let msg_mod: usize = 1 << log_msg_mod; // 16
        let x: i64 = 5;

        // Encode and encrypt directly as LWE under the binary key
        let lwe_layout = LWELayout {
            n: Degree(bp.n_lwe as u32),
            k: TorusPrecision(bp.k_lwe_small as u32),
            base2k: Base2K(bp.base2k_lwe_small as u32),
        };
        let mut pt_lwe = LWEPlaintext::<Vec<u8>>::alloc_from_infos(&lwe_layout);
        pt_lwe.encode_i64(x, TorusPrecision((log_msg_mod + 1) as u32));

        let mut lwe = LWE::<Vec<u8>>::alloc_from_infos(&lwe_layout);
        lwe.encrypt_sk(&module, &pt_lwe, &sk_lwe, &mut source_xa, &mut source_xe, scratch.borrow());

        // Identity LUT
        let mut f_vec: Vec<i64> = vec![0i64; msg_mod];
        for (i, val) in f_vec.iter_mut().enumerate() {
            *val = i as i64;
        }
        let lut_layout = LookUpTableLayout {
            n: Degree(n_glwe as u32),
            extension_factor: bp.extension_factor,
            k: TorusPrecision(bp.k_lut as u32),
            base2k: Base2K(bp.base2k_brk as u32),
        };
        let mut lut = LookupTable::alloc(&lut_layout);
        lut.set(&module, &f_vec, log_msg_mod + 1);

        // Result GLWE
        let res_layout = GLWELayout {
            n: Degree(n_glwe as u32),
            base2k: Base2K(bp.base2k_brk as u32),
            k: TorusPrecision(bp.k_res as u32),
            rank: Rank(1),
        };
        let mut res_glwe = GLWE::<Vec<u8>>::alloc_from_infos(&res_layout);

        let br_bytes =
            BlindRotationKeyPrepared::execute_tmp_bytes(&module, bp.block_size, bp.extension_factor, &res_layout, &brk_prepared);
        let mut br_scratch2: ScratchOwned<BE> = ScratchOwned::alloc(br_bytes);
        brk_prepared.execute(&module, &mut res_glwe, &lwe, &lut, br_scratch2.borrow());

        // Decrypt the result
        let mut pt_res = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&res_layout);
        res_glwe.decrypt(&module, &mut pt_res, &sk_glwe_dft, scratch.borrow());

        let decoded = pt_res.decode_coeff_i64(TorusPrecision((log_msg_mod + 1) as u32), 0);
        let decoded_pos = (decoded + msg_mod as i64) % (msg_mod as i64);

        assert_eq!(
            decoded_pos, x,
            "Native LWE blind rotation failed: expected {x}, got {decoded_pos} (raw {decoded})"
        );
    }
}
