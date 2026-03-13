//! LUT-based nonlinearity evaluation via blind rotation.
//!
//! For cases where polynomial approximations are insufficiently accurate,
//! CHIMERA can evaluate arbitrary fixed-precision nonlinearities by encoding
//! them as lookup tables and using Poulpy's TFHE-style blind rotation
//! infrastructure.
//!
//! ## How it works
//!
//! 1. A nonlinear function f(x) is discretised over the input range and
//!    encoded into a GLWE lookup table.
//! 2. For each slot of the input ciphertext, an LWE sample is extracted.
//! 3. The LWE sample drives a blind rotation that selects the correct LUT
//!    entry, producing a GLWE ciphertext encrypting f(x).
//! 4. Results are repacked into a multi-slot GLWE ciphertext.
//!
//! This is more expensive than polynomial evaluation (~1000x per slot) but
//! provides exact results for the discretised input range and arbitrary
//! function support.

use crate::params::ChimeraParams;

/// A discretised lookup table for a nonlinear function.
///
/// Maps input values in [lo, hi] discretised to `num_entries` points
/// to output values.
#[derive(Clone, Debug)]
pub struct NonlinearLUT {
    /// Function name (for debugging).
    pub name: String,
    /// Input range lower bound.
    pub lo: f64,
    /// Input range upper bound.
    pub hi: f64,
    /// Number of LUT entries (must be a power of 2 and ≤ N).
    pub num_entries: usize,
    /// Output values: lut[i] = f(lo + i * (hi - lo) / num_entries).
    pub entries: Vec<f64>,
    /// Plaintext precision bits for the output encoding.
    pub output_bits: u32,
}

impl NonlinearLUT {
    /// Creates a LUT by sampling a function at `num_entries` equally-spaced points.
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable function name.
    /// * `f` - The function to tabulate.
    /// * `lo`, `hi` - Input range.
    /// * `num_entries` - Number of table entries (should be power of 2).
    /// * `output_bits` - Precision bits for output encoding.
    pub fn from_fn(name: &str, f: impl Fn(f64) -> f64, lo: f64, hi: f64, num_entries: usize, output_bits: u32) -> Self {
        assert!(num_entries > 0);
        assert!(hi > lo);

        let step = (hi - lo) / num_entries as f64;
        let entries: Vec<f64> = (0..num_entries).map(|i| f(lo + (i as f64 + 0.5) * step)).collect();

        NonlinearLUT {
            name: name.to_string(),
            lo,
            hi,
            num_entries,
            entries,
            output_bits,
        }
    }

    /// Creates a GELU LUT for the range [-8, 8] with 256 entries.
    pub fn gelu(output_bits: u32) -> Self {
        Self::from_fn(
            "gelu",
            |x| {
                // GELU(x) = x · Φ(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
                let inner = (2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            },
            -8.0,
            8.0,
            256,
            output_bits,
        )
    }

    /// Creates a SiLU/Swish LUT for the range [-8, 8] with 256 entries.
    pub fn silu(output_bits: u32) -> Self {
        Self::from_fn("silu", |x| x / (1.0 + (-x).exp()), -8.0, 8.0, 256, output_bits)
    }

    /// Creates integer LUT entries for SiLU over a symmetric signed message domain.
    ///
    /// The returned vector has length `2^log_message_modulus`, where index `i`
    /// is interpreted as the signed integer `(i as i64)` for the positive half
    /// and `(i as i64 - 2^log_message_modulus)` for the negative half.
    pub fn silu_message_lut(log_message_modulus: usize) -> Vec<i64> {
        let message_modulus = 1usize << log_message_modulus;
        let half_mod = message_modulus / 2;
        (0..message_modulus)
            .map(|i| {
                let signed = if i < half_mod {
                    i as i64
                } else {
                    i as i64 - message_modulus as i64
                };
                let x = signed as f64;
                (x / (1.0 + (-x).exp())).round() as i64
            })
            .collect()
    }

    /// Creates an exp(x) LUT for the range [-8, 0] with 256 entries.
    ///
    /// Used for softmax computation.
    pub fn exp(output_bits: u32) -> Self {
        Self::from_fn("exp", |x| x.exp(), -8.0, 0.0, 256, output_bits)
    }

    /// Creates a 1/√x LUT for the range [0.01, 10] with 256 entries.
    ///
    /// Used for LayerNorm inverse square root.
    pub fn inv_sqrt(output_bits: u32) -> Self {
        Self::from_fn("inv_sqrt", |x| 1.0 / x.sqrt(), 0.01, 10.0, 256, output_bits)
    }

    /// Creates a 1/x LUT for the range [0.01, 10] with 256 entries.
    ///
    /// Used for softmax normalisation.
    pub fn reciprocal(output_bits: u32) -> Self {
        Self::from_fn("reciprocal", |x| 1.0 / x, 0.01, 10.0, 256, output_bits)
    }

    /// Evaluates the LUT at a given input by nearest-entry lookup.
    ///
    /// For testing/verification only (no FHE involved).
    pub fn eval(&self, x: f64) -> f64 {
        let x_clamped = x.clamp(self.lo, self.hi);
        let step = (self.hi - self.lo) / self.num_entries as f64;
        let idx = ((x_clamped - self.lo) / step).floor() as usize;
        let idx = idx.min(self.num_entries - 1);
        self.entries[idx]
    }

    /// Converts entries to scaled i64 values for encoding into a Poulpy LUT polynomial.
    ///
    /// The output values are scaled to the torus: entry * 2^(64 - output_bits).
    ///
    /// # Panics
    ///
    /// Panics if `output_bits` is 0 or greater than 62 (to avoid overflow).
    pub fn to_i64_entries(&self) -> Vec<i64> {
        assert!(
            self.output_bits > 0 && self.output_bits <= 62,
            "output_bits must be in [1, 62], got {}",
            self.output_bits,
        );
        let shift = 64 - self.output_bits;
        self.entries
            .iter()
            .map(|&v| {
                let scale = (1u64 << self.output_bits) as f64;
                let quantised = (v * scale).round().clamp(i64::MIN as f64, i64::MAX as f64) as i64;
                // Use wrapping_shl to avoid UB if shift is exactly 64
                // (which can't happen given the assert, but is defensive)
                quantised.wrapping_shl(shift)
            })
            .collect()
    }
}

/// Estimates the cost (in GLWE operations) of evaluating a LUT for a single slot.
///
/// The cost is dominated by blind rotation, which requires `n_lwe` external products
/// where n_lwe is the LWE dimension (= N for extracted LWE from GLWE).
///
/// # Arguments
///
/// * `params` - CHIMERA parameter set.
/// * `lut` - The lookup table.
///
/// # Returns
///
/// Approximate number of polynomial multiplications.
pub fn lut_eval_cost(params: &ChimeraParams, _lut: &NonlinearLUT) -> usize {
    // Blind rotation cost: N external products (each is ~1 poly mul)
    // Plus extraction + repacking overhead
    let blind_rotation_cost = params.degree.0 as usize;
    let extraction_cost = 1;
    let repacking_cost = 1;
    blind_rotation_cost + extraction_cost + repacking_cost
}

/// Estimates the total cost for evaluating a LUT across all active slots.
pub fn lut_eval_total_cost(params: &ChimeraParams, lut: &NonlinearLUT, num_slots: usize) -> usize {
    num_slots * lut_eval_cost(params, lut)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_lut() {
        let lut = NonlinearLUT::gelu(8);
        assert_eq!(lut.num_entries, 256);

        // GELU(0) ≈ 0
        let at_zero = lut.eval(0.0);
        assert!(at_zero.abs() < 0.05, "GELU LUT(0) = {at_zero}");

        // GELU(2) ≈ 1.9545
        let at_two = lut.eval(2.0);
        assert!((at_two - 1.9545).abs() < 0.1, "GELU LUT(2) = {at_two}");
    }

    #[test]
    fn test_silu_lut() {
        let lut = NonlinearLUT::silu(8);
        let at_zero = lut.eval(0.0);
        assert!(at_zero.abs() < 0.05, "SiLU LUT(0) = {at_zero}");
    }

    #[test]
    fn test_exp_lut() {
        let lut = NonlinearLUT::exp(8);
        let at_zero = lut.eval(-0.01); // Near 0
        assert!((at_zero - 1.0).abs() < 0.1, "exp LUT(~0) = {at_zero}");

        let at_neg_one = lut.eval(-1.0);
        assert!((at_neg_one - 0.3679).abs() < 0.05, "exp LUT(-1) = {at_neg_one}");
    }

    #[test]
    fn test_inv_sqrt_lut() {
        let lut = NonlinearLUT::inv_sqrt(8);
        let at_one = lut.eval(1.0);
        assert!((at_one - 1.0).abs() < 0.1, "inv_sqrt LUT(1) = {at_one}");

        let at_four = lut.eval(4.0);
        assert!((at_four - 0.5).abs() < 0.1, "inv_sqrt LUT(4) = {at_four}");
    }

    #[test]
    fn test_reciprocal_lut() {
        let lut = NonlinearLUT::reciprocal(8);
        let at_one = lut.eval(1.0);
        assert!((at_one - 1.0).abs() < 0.1, "reciprocal LUT(1) = {at_one}");
    }

    #[test]
    fn test_lut_to_i64() {
        let lut = NonlinearLUT::from_fn("test", |x| x, 0.0, 1.0, 4, 8);
        let entries = lut.to_i64_entries();
        assert_eq!(entries.len(), 4);
        // All entries should be non-zero (values are in (0,1))
        for &e in &entries {
            assert_ne!(e, 0);
        }
    }

    #[test]
    fn test_lut_cost_estimate() {
        let params = crate::params::ChimeraParams::new(crate::params::SecurityLevel::Bits80, crate::params::Precision::Int8);
        let lut = NonlinearLUT::gelu(8);
        let cost = lut_eval_cost(&params, &lut);
        assert!(cost > 0);
        assert_eq!(cost, 4096 + 2); // N + extraction + repacking
    }
}
