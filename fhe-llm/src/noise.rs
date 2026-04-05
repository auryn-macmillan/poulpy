//! Noise tracking and budget estimation.
//!
//! Provides tools for tracking the noise budget consumed by FHE_LLM
//! operations and estimating whether a given computation will stay
//! within the noise budget (avoiding the need for bootstrapping).
//!
//! ## Noise Model
//!
//! FHE_LLM uses an approximate noise model based on the following:
//!
//! - Fresh encryption noise: σ_fresh = 3.2 (standard deviation)
//! - Addition: σ² = σ₁² + σ₂²
//! - Plaintext multiplication by constant c: σ² = |c|² · σ²_in
//! - Rescaling (bit-shift by base2k): σ² = σ²_in / 2^(2·base2k) + 2^(-2·base2k)/12
//! - Ciphertext multiplication (tensor product): σ² ≈ σ₁² · σ₂² · N
//!
//! The noise budget is exhausted when the noise magnitude exceeds
//! half the plaintext modulus (Δ/2), causing decryption errors.
//!
//! ## Noise Budget Analysis
//!
//! This module tracks noise growth at each transformer sublayer to identify
//! where signal is lost. The goal is to achieve hidden state L-inf < 0.015
//! for correct token prediction.
//!
//! ### Operations to Track
//!
//! - QKV projection: matrix multiplication with INT8 weights
//! - Attention scores: Q · K^T multiplication
//! - Attention output: attn · V multiplication
//! - Output projection: attention output × weight matrix
//! - FFN gate/up: matrix multiplication
//! - SiLU activation: elementwise multiplication
//! - FFN down: matrix multiplication
//! - RMSNorm: inv_sqrt polynomial + scaling
//! - Bootstrap: quantization to log_message_modulus levels

use crate::params::FHE_LLMParams;

/// Fresh encryption noise standard deviation.
pub const SIGMA_FRESH: f64 = 3.2;

/// Noise state tracker for a FHE_LLM ciphertext.
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
        // Approximate relinearisation noise based on typical FHE_LLM parameters:
        // dnum ≈ k / tsk_base2k ≈ 54 / 14 ≈ 4, tsk_base2k = 14
        // relin_noise ≈ N * dnum * σ²_fresh / 2^(2 * tsk_base2k)
        let dnum_approx = 4.0;
        let tsk_base2k = 14.0;
        let relin_noise = ring_degree as f64 * dnum_approx * (SIGMA_FRESH * SIGMA_FRESH) / (2.0_f64).powf(2.0 * tsk_base2k);
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
    pub fn budget_bits(&self, params: &FHE_LLMParams) -> f64 {
        let delta_bits = params.scale_bits as f64;
        let noise_bits = self.std_dev().log2();
        delta_bits - noise_bits - 1.0
    }

    /// Returns whether the noise budget is still positive (decryption should succeed).
    pub fn is_valid(&self, params: &FHE_LLMParams) -> bool {
        self.budget_bits(params) > 0.0
    }

    /// Resets the noise tracker to its fresh state after a bootstrapping operation.
    ///
    /// Bootstrapping refreshes the noise budget by re-encrypting the plaintext
    /// value through a blind rotation. The resulting ciphertext has noise
    /// comparable to a fresh encryption, so this resets the variance, depth,
    /// and operation count accordingly. A "bootstrap_reset" event is recorded
    /// in the history.
    pub fn bootstrap_reset(&mut self) {
        self.variance = SIGMA_FRESH * SIGMA_FRESH;
        self.depth = 0;
        self.num_ops = 0;
        self.history.push(NoiseEvent {
            op: "bootstrap_reset".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records QKV projection noise growth.
    ///
    /// QKV projection performs matrix multiplication with INT8 weights.
    /// Noise grows proportionally to the L2 norm of the weight matrix.
    pub fn qkv_projection(&mut self, weight_l2_norm: f64) {
        self.mul_const(weight_l2_norm);
        self.history.push(NoiseEvent {
            op: "qkv_projection".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records attention scores noise growth (Q · K^T).
    pub fn attention_scores(&mut self, other: &NoiseTracker, ring_degree: usize) {
        let before = self.variance;
        self.mul_ct(other, ring_degree);
        self.history.push(NoiseEvent {
            op: "attention_scores".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
            // Note: depth_after may need adjustment based on actual operation
        });
    }

    /// Records attention output noise growth (attn · V).
    pub fn attention_output(&mut self, other: &NoiseTracker, ring_degree: usize) {
        let before = self.variance;
        self.mul_ct(other, ring_degree);
        self.history.push(NoiseEvent {
            op: "attention_output".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records output projection noise growth.
    pub fn output_projection(&mut self, weight_l2_norm: f64) {
        let before = self.variance;
        self.mul_const(weight_l2_norm);
        self.history.push(NoiseEvent {
            op: "output_projection".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records FFN gate/up projection noise growth.
    pub fn ffn_gate_up(&mut self, weight_l2_norm: f64) {
        let before = self.variance;
        self.mul_const(weight_l2_norm);
        self.history.push(NoiseEvent {
            op: "ffn_gate_up".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records SiLU activation noise growth (elementwise ct × ct).
    pub fn silu_activation(&mut self, other: &NoiseTracker, ring_degree: usize) {
        let before = self.variance;
        self.mul_ct(other, ring_degree);
        self.history.push(NoiseEvent {
            op: "silu_activation".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records FFN down projection noise growth.
    pub fn ffn_down(&mut self, weight_l2_norm: f64) {
        let before = self.variance;
        self.mul_const(weight_l2_norm);
        self.history.push(NoiseEvent {
            op: "ffn_down".to_string(),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records RMSNorm noise growth (inv_sqrt polynomial + scaling).
    pub fn rms_norm(&mut self, polynomial_order: usize, scale_factor: f64) {
        // Polynomial inv_sqrt: degree-3 polynomial, each degree adds depth
        for _ in 0..polynomial_order {
            let self_copy = self.clone();
            self.mul_ct(&self_copy, 8192); // Typical ring degree
            self.rescale(26); // RMS output scale
        }
        // Scaling by inv_rms value
        self.mul_const(scale_factor);
        self.history.push(NoiseEvent {
            op: format!("rms_norm(deg={polynomial_order}, scale={scale_factor:.4})"),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Records bootstrap quantization noise.
    ///
    /// Bootstrap quantizes the ciphertext to log_message_modulus levels.
    /// The quantization error is approximately ±0.5 * message_step.
    pub fn bootstrap_quantization(&mut self, log_message_modulus: u32, scale_bits: u32) {
        // Quantization error: ±0.5 * 2^(scale_bits - log_message_modulus)
        let levels = 1u64 << log_message_modulus;
        let step = (1u64 << scale_bits) as f64 / levels as f64;
        let quant_noise = (step * 0.5).powi(2) / 12.0; // Uniform distribution variance
        self.variance += quant_noise;
        self.depth = 0; // Bootstrap resets multiplicative depth
        self.num_ops = 0;
        self.history.push(NoiseEvent {
            op: format!("bootstrap_quant(log_mod={log_message_modulus}, scale={scale_bits})"),
            variance_after: self.variance,
            depth_after: self.depth,
        });
    }

    /// Returns noise standard deviation in the original message scale.
    ///
    /// Converts variance to L-inf bound for comparison with hidden state values.
    /// Assumes Gaussian noise distribution: L-inf ≈ 3 * std_dev (99.7% confidence).
    pub fn linf_bound(&self) -> f64 {
        3.0 * self.std_dev()
    }
}

/// Estimates total noise growth for a sequence of transformer operations.
pub fn estimate_transformer_layer_noise(params: &FHE_LLMParams, d_model: usize) -> NoiseTracker {
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
    use crate::params::{FHE_LLMParams, Precision, SecurityLevel};

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
        let params = FHE_LLMParams::new(SecurityLevel::Bits128, Precision::Int8);
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
        let params = FHE_LLMParams::new(SecurityLevel::Bits128, Precision::Int8);
        let tracker = estimate_transformer_layer_noise(&params, 4096);

        // After a full layer, depth should be significant
        assert!(tracker.depth > 0);
        assert!(tracker.num_ops > 0);
        // Noise will be very large due to ct-ct multiplications
        // This demonstrates why bootstrapping is needed for deep models
    }
}
