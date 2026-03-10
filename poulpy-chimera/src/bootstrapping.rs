//! Bootstrapping support for deep model inference.
//!
//! Bootstrapping refreshes the noise budget of a ciphertext, allowing
//! continued computation beyond the initial noise budget limit. For
//! CHIMERA's target workload (20B-40B parameter transformers with 32+
//! layers), a single noise budget may be insufficient, requiring one or
//! more bootstrapping operations during inference.
//!
//! ## Current Status
//!
//! This module provides:
//! - [`BootstrappingConfig`]: configuration for when and how to bootstrap
//! - [`needs_bootstrap`]: checks whether a ciphertext's estimated noise
//!   budget requires refreshing before further computation
//! - [`chimera_bootstrap`]: **stub** — panics with a diagnostic message
//!
//! Full bootstrapping implementation requires:
//! 1. A bootstrapping key (encrypted under the CHIMERA secret key)
//! 2. Blind rotation via TFHE-style functional bootstrapping adapted to
//!    CHIMERA's bivariate torus representation
//! 3. Key switching back to the GLWE domain
//!
//! ## Design Rationale
//!
//! CHIMERA's design prioritises **bootstrapping elimination** for shallow
//! models (≤32 layers at 128-bit security). The noise parameters are tuned
//! so that a full forward pass through a typical transformer block consumes
//! less than the available noise budget. For deeper models or multi-pass
//! inference, bootstrapping is needed at most once per forward pass.
//!
//! The expected bootstrapping approach is:
//! - **Programmable bootstrapping** (PBS) adapted from TFHE, operating on
//!   CHIMERA's bivariate limb decomposition rather than standard torus
//! - Bootstrap at the **boundary between transformer blocks** (after the
//!   residual connection), where the ciphertext is at its noisiest
//! - Optionally fold a nonlinear activation (e.g. GELU) into the
//!   bootstrapping step (functional bootstrapping), amortising the cost
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

use crate::noise::NoiseTracker;
use crate::params::ChimeraParams;

use poulpy_core::layouts::GLWE;
use poulpy_hal::layouts::Backend;

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
pub fn needs_bootstrap(
    tracker: &NoiseTracker,
    params: &ChimeraParams,
    config: &BootstrappingConfig,
) -> bool {
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

/// **STUB** — Bootstraps a CHIMERA ciphertext to refresh its noise budget.
///
/// # Status
///
/// This function is **not yet implemented**. It currently panics with a
/// diagnostic message including the estimated noise budget of the input
/// ciphertext.
///
/// # Future Implementation
///
/// When implemented, this function will:
/// 1. Extract LWE samples from the GLWE ciphertext
/// 2. Perform blind rotation using a bootstrapping key
/// 3. Repack the result into a fresh GLWE ciphertext with a reset noise budget
///
/// The output ciphertext will encrypt the same values as the input but
/// with noise variance reset to approximately `σ_fresh²`.
///
/// # Panics
///
/// Always panics with a message indicating that bootstrapping is not yet
/// implemented, along with the current noise state.
///
/// # Arguments
///
/// * `_module` - The backend module.
/// * `ct` - The ciphertext to bootstrap.
/// * `tracker` - Current noise state (for diagnostic output).
/// * `params` - CHIMERA parameter set.
/// * `_config` - Bootstrapping configuration.
pub fn chimera_bootstrap<BE: Backend>(
    _module: &Module<BE>,
    _ct: &GLWE<Vec<u8>>,
    tracker: &NoiseTracker,
    params: &ChimeraParams,
    _config: &BootstrappingConfig,
) -> GLWE<Vec<u8>> {
    let budget = tracker.budget_bits(params);
    let cost = estimate_bootstrap_cost(params);
    panic!(
        "chimera_bootstrap: not yet implemented.\n\
         Noise budget: {budget:.1} bits (depth: {depth}, ops: {ops})\n\
         Estimated bootstrap cost: {cost_ops} polynomial multiplications (~{cost_ms:.1} ms)\n\
         \n\
         To proceed without bootstrapping, either:\n\
         - Reduce model depth to <= {max_layers} layers (at {security:?} security), or\n\
         - Increase the security level (which increases the polynomial degree N and noise budget), or\n\
         - Use BootstrappingConfig::no_bootstrap() to suppress this check\n\
         \n\
         See docs/chimera_spec.md §3 (Single-Budget Inference) for details.",
        depth = tracker.depth,
        ops = tracker.num_ops,
        cost_ops = cost.total_poly_muls,
        cost_ms = cost.estimated_latency_ms,
        max_layers = params.max_layers_no_bootstrap(),
        security = params.security,
    );
}

use poulpy_hal::layouts::Module;

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
    #[should_panic(expected = "not yet implemented")]
    fn test_chimera_bootstrap_panics() {
        use poulpy_hal::api::ModuleNew;

        #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
        type BE = poulpy_cpu_ref::FFT64Ref;
        #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
        type BE = poulpy_cpu_avx::FFT64Avx;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let tracker = NoiseTracker::fresh();
        let config = BootstrappingConfig::for_deep_model(4.0, 1);

        // Create a dummy ciphertext
        let layout = poulpy_core::layouts::GLWELayout {
            n: params.degree,
            base2k: poulpy_core::layouts::Base2K(params.in_base2k() as u32),
            k: params.k_ct,
            rank: params.rank,
        };
        let ct = GLWE::<Vec<u8>>::alloc_from_infos(&layout);

        // This should panic with "not yet implemented"
        let _ = chimera_bootstrap(&module, &ct, &tracker, &params, &config);
    }
}
