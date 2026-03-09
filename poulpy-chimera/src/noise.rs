//! Noise tracking and budget estimation.
//!
//! Provides tools for tracking the noise budget consumed by CHIMERA
//! operations and estimating whether a given computation will stay
//! within the noise budget (avoiding the need for bootstrapping).
//!
//! ## Noise Model
//!
//! CHIMERA uses an approximate noise model based on the following:
//!
//! - Fresh encryption noise: σ_fresh = 3.2 (standard deviation)
//! - Addition: σ² = σ₁² + σ₂²
//! - Plaintext multiplication by constant c: σ² = |c|² · σ²_in
//! - Rescaling (bit-shift by base2k): σ² = σ²_in / 2^(2·base2k) + 2^(-2·base2k)/12
//! - Ciphertext multiplication (tensor product): σ² ≈ σ₁² · σ₂² · N
//!
//! The noise budget is exhausted when the noise magnitude exceeds
//! half the plaintext modulus (Δ/2), causing decryption errors.

use crate::params::ChimeraParams;

/// Fresh encryption noise standard deviation.
pub const SIGMA_FRESH: f64 = 3.2;

/// Noise state tracker for a CHIMERA ciphertext.
///
/// Tracks the estimated noise variance through a chain of operations.
/// This is a conservative upper bound; actual noise may be lower.
#[derive(Clone, Debug)]
pub struct NoiseTracker {
    /// Estimated noise variance (σ²).
    pub variance: f64,
    /// Number of operations applied.
    pub num_ops: usize,
    /// Multiplicative depth consumed.
    pub depth: usize,
    /// History of operations (for debugging).
    pub history: Vec<NoiseEvent>,
}

/// A single noise-affecting event.
#[derive(Clone, Debug)]
pub struct NoiseEvent {
    /// Operation name.
    pub op: String,
    /// Noise variance after this operation.
    pub variance_after: f64,
    /// Depth after this operation.
    pub depth_after: usize,
}

impl NoiseTracker {
    /// Creates a fresh noise tracker for a newly encrypted ciphertext.
    pub fn fresh() -> Self {
        NoiseTracker {
            variance: SIGMA_FRESH * SIGMA_FRESH,
            num_ops: 0,
            depth: 0,
            history: vec![NoiseEvent {
                op: "encrypt".to_string(),
                variance_after: SIGMA_FRESH * SIGMA_FRESH,
                depth_after: 0,
            }],
        }
    }

    /// Records an addition of two ciphertexts.
    pub fn add(&mut self, other: &NoiseTracker) {
        self.variance += other.variance;
        self.num_ops += 1;
        self.depth = self.depth.max(other.depth);
        self.history.push(NoiseEvent {
            op: "add".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records a subtraction of two ciphertexts.
    pub fn sub(&mut self, other: &NoiseTracker) {
        self.variance += other.variance;
        self.num_ops += 1;
        self.depth = self.depth.max(other.depth);
        self.history.push(NoiseEvent {
            op: "sub".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records a plaintext multiplication by a constant with given L2 norm.
    pub fn mul_const(&mut self, const_l2_norm: f64) {
        self.variance *= const_l2_norm * const_l2_norm;
        self.num_ops += 1;
        self.depth += 1;
        self.history.push(NoiseEvent {
            op: format!("mul_const(||c||={const_l2_norm:.2})"),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records a ciphertext-ciphertext multiplication (tensor product).
    ///
    /// This is the most expensive operation in terms of noise growth.
    /// σ²_out ≈ σ²_a · σ²_b · N + relinearisation noise
    ///
    /// The relinearisation noise depends on the tensor key parameters (dnum, dsize, base2k).
    /// We approximate it as: N * dnum * σ²_fresh / 2^(2 * tsk_base2k)
    /// This captures the key-switching noise from the tensor key.
    pub fn mul_ct(&mut self, other: &NoiseTracker, ring_degree: usize) {
        // Approximate relinearisation noise based on typical CHIMERA parameters:
        // dnum ≈ k / tsk_base2k ≈ 54 / 14 ≈ 4, tsk_base2k = 14
        // relin_noise ≈ N * dnum * σ²_fresh / 2^(2 * tsk_base2k)
        let dnum_approx = 4.0;
        let tsk_base2k = 14.0;
        let relin_noise = ring_degree as f64 * dnum_approx * (SIGMA_FRESH * SIGMA_FRESH)
            / (2.0_f64).powf(2.0 * tsk_base2k);
        self.variance = self.variance * other.variance * ring_degree as f64 + relin_noise;
        self.num_ops += 1;
        self.depth = self.depth.max(other.depth) + 1;
        self.history.push(NoiseEvent {
            op: "mul_ct".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records a rescaling operation (bit-shift by base2k bits).
    pub fn rescale(&mut self, base2k: u32) {
        let divisor = (1u64 << (2 * base2k)) as f64;
        let rounding_noise = 1.0 / (12.0 * divisor);
        self.variance = self.variance / divisor + rounding_noise;
        self.num_ops += 1;
        self.history.push(NoiseEvent {
            op: format!("rescale({base2k})"),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Returns the noise standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Returns the noise budget in bits.
    ///
    /// noise_budget = log₂(Δ / (2 · σ)) where Δ = 2^scale_bits.
    pub fn budget_bits(&self, params: &ChimeraParams) -> f64 {
        let delta_bits = params.scale_bits as f64;
        let noise_bits = self.std_dev().log2();
        delta_bits - noise_bits - 1.0
    }

    /// Returns whether the noise budget is still positive (decryption should succeed).
    pub fn is_valid(&self, params: &ChimeraParams) -> bool {
        self.budget_bits(params) > 0.0
    }
}

/// Estimates total noise growth for a sequence of transformer operations.
pub fn estimate_transformer_layer_noise(
    params: &ChimeraParams,
    d_model: usize,
) -> NoiseTracker {
    let mut tracker = NoiseTracker::fresh();

    // QKV projection: matmul with d_model accumulations
    // Each accumulation multiplies by a weight (~1) and adds
    let weight_norm = (d_model as f64).sqrt(); // L2 norm of a random weight column
    tracker.mul_const(weight_norm);
    tracker.rescale(params.base2k.0);

    // Attention scores: matmul Q · K^T with d_head accumulations
    let _d_head = 128; // Typical
    let mut k_tracker = NoiseTracker::fresh();
    k_tracker.mul_const(weight_norm);
    k_tracker.rescale(params.base2k.0);
    tracker.mul_ct(&k_tracker, params.degree.0 as usize);
    tracker.rescale(params.base2k.0);

    // Softmax approximation: degree-4 polynomial
    // Each degree increases depth by 1
    for _ in 0..3 {
        let self_copy = tracker.clone();
        tracker.mul_ct(&self_copy, params.degree.0 as usize);
        tracker.rescale(params.base2k.0);
    }

    // Context = attn · V: matmul with d_head accumulations
    let mut v_tracker = NoiseTracker::fresh();
    v_tracker.mul_const(weight_norm);
    v_tracker.rescale(params.base2k.0);
    tracker.mul_ct(&v_tracker, params.degree.0 as usize);
    tracker.rescale(params.base2k.0);

    // Output projection
    tracker.mul_const(weight_norm);
    tracker.rescale(params.base2k.0);

    // FFN layer 1
    tracker.mul_const(weight_norm);
    tracker.rescale(params.base2k.0);

    // Activation (degree-3)
    for _ in 0..2 {
        let self_copy = tracker.clone();
        tracker.mul_ct(&self_copy, params.degree.0 as usize);
        tracker.rescale(params.base2k.0);
    }

    // FFN layer 2
    tracker.mul_const(weight_norm);
    tracker.rescale(params.base2k.0);

    tracker
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{ChimeraParams, Precision, SecurityLevel};

    #[test]
    fn test_fresh_noise() {
        let tracker = NoiseTracker::fresh();
        assert!((tracker.std_dev() - SIGMA_FRESH).abs() < 1e-10);
        assert_eq!(tracker.depth, 0);
        assert_eq!(tracker.num_ops, 0);
    }

    #[test]
    fn test_add_noise() {
        let mut a = NoiseTracker::fresh();
        let b = NoiseTracker::fresh();
        a.add(&b);

        let expected_var = 2.0 * SIGMA_FRESH * SIGMA_FRESH;
        assert!((a.variance - expected_var).abs() < 1e-10);
        assert_eq!(a.num_ops, 1);
    }

    #[test]
    fn test_mul_const_noise() {
        let mut tracker = NoiseTracker::fresh();
        tracker.mul_const(10.0);

        let expected_var = SIGMA_FRESH * SIGMA_FRESH * 100.0;
        assert!((tracker.variance - expected_var).abs() < 1e-6);
        assert_eq!(tracker.depth, 1);
    }

    #[test]
    fn test_rescale_reduces_noise() {
        let mut tracker = NoiseTracker::fresh();
        tracker.mul_const(100.0);
        let before = tracker.variance;
        tracker.rescale(14);
        assert!(tracker.variance < before);
    }

    #[test]
    fn test_noise_budget() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let tracker = NoiseTracker::fresh();
        let budget = tracker.budget_bits(&params);
        // Fresh ciphertext should have positive budget
        assert!(budget > 0.0, "budget = {budget}");
    }

    #[test]
    fn test_noise_history() {
        let mut tracker = NoiseTracker::fresh();
        tracker.mul_const(5.0);
        tracker.rescale(14);

        assert_eq!(tracker.history.len(), 3); // encrypt + mul_const + rescale
        assert_eq!(tracker.history[0].op, "encrypt");
    }

    #[test]
    fn test_transformer_layer_noise() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let tracker = estimate_transformer_layer_noise(&params, 4096);

        // After a full layer, depth should be significant
        assert!(tracker.depth > 0);
        assert!(tracker.num_ops > 0);
        // Noise will be very large due to ct-ct multiplications
        // This demonstrates why bootstrapping is needed for deep models
    }
}
