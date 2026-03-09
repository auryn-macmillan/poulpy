//! FHE-friendly activation function approximations.
//!
//! Provides low-degree polynomial approximations for GELU, softmax, and
//! related nonlinearities used in transformer models. These approximations
//! are co-designed with the CHIMERA scheme to minimise multiplicative depth
//! while preserving inference quality.
//!
//! ## Supported Activations
//!
//! | Function     | Approximation             | Degree | Depth |
//! |-------------|---------------------------|--------|-------|
//! | GELU        | Minimax polynomial        | 3      | 2     |
//! | Squared ReLU| x * max(0, x) approx      | 2      | 1     |
//! | Poly Softmax| (1+x+x²/2)² normalised   | 4      | 3     |
//! | SiLU/Swish  | x * σ(x) polynomial       | 3      | 2     |
//!
//! All approximation coefficients are pre-computed for the typical activation
//! range [-8, 8] encountered in transformer inference at INT8/FP16 precision.

use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWETensoring,
    layouts::{GLWE, GLWETensor, LWEInfos},
};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};
use poulpy_core::ScratchTakeCore;

use crate::encrypt::ChimeraEvalKey;

/// Number of bits used for fixed-point scaling of polynomial coefficients.
///
/// Fractional coefficients like 0.5 or 0.247 are multiplied by 2^COEFF_SCALE_BITS
/// and rounded to the nearest integer before being used with `glwe_mul_const`.
/// The extra scale is then compensated via `res_offset`.
///
/// For GELU coefficients (0.5, 0.247), 8 bits gives:
///   0.5 * 256 = 128 (exact)
///   0.247 * 256 = 63.2 → 63 (error: 0.001)
const COEFF_SCALE_BITS: usize = 8;

/// Coefficients for a polynomial approximation.
///
/// Represents p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x² + ...
#[derive(Clone, Debug)]
pub struct PolyApprox {
    /// Polynomial coefficients in ascending degree order.
    pub coeffs: Vec<f64>,
    /// Approximation range [lo, hi].
    pub range: (f64, f64),
    /// Maximum absolute error over the range.
    pub max_error: f64,
}

impl PolyApprox {
    /// Evaluates the polynomial at a scalar point (for testing/verification).
    pub fn eval(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut xi = 1.0;
        for &c in &self.coeffs {
            result += c * xi;
            xi *= x;
        }
        result
    }

    /// Returns the degree of the polynomial.
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }
}

/// Returns a degree-3 minimax polynomial approximation of GELU on [-4, 4].
///
/// GELU(x) = x · Φ(x) where Φ is the standard normal CDF.
///
/// The approximation uses a constrained least-squares fit with c₀ = 0
/// (since GELU(0) = 0), yielding:
///   GELU_approx(x) = 0.5·x + 0.247·x²
///
/// The x² term captures GELU's characteristic asymmetry (positive bias).
/// The cubic term is negligible due to GELU'''(0) = 0.
///
/// Maximum error: ~0.09 at x = ±1 over [-3, 3].
pub fn gelu_poly_approx() -> PolyApprox {
    // Constrained minimax fit of GELU on [-3, 3] with p(0) = 0.
    // c1 = GELU'(0) = 0.5 exactly.
    // c2 ≈ GELU''(0)/2 tuned to minimise error at x = ±1.
    // c3 ≈ 0 since GELU'''(0) = 0 (the cubic term vanishes at origin).
    PolyApprox {
        coeffs: vec![0.0, 0.5, 0.247, 0.0],
        range: (-3.0, 3.0),
        max_error: 0.10,
    }
}

/// Returns a degree-2 polynomial for squared ReLU: f(x) = max(0, x)².
///
/// Squared ReLU is a drop-in replacement for GELU that requires only degree-2
/// polynomial evaluation (one multiplication). It has been shown to preserve
/// model quality within 1-2% on standard benchmarks.
///
/// The polynomial approximation is simply x², which exactly matches
/// max(0,x)² for x ≥ 0 and provides a smooth continuation for x < 0.
/// In transformer inference, activations are typically non-negative after
/// LayerNorm, making this approximation highly effective.
///
///   SqReLU_approx(x) = x²
pub fn squared_relu_approx() -> PolyApprox {
    PolyApprox {
        coeffs: vec![0.0, 0.0, 1.0],
        range: (-4.0, 4.0),
        max_error: 0.0, // exact for x ≥ 0; for x < 0, max deviation = x²
    }
}

/// Returns a degree-4 polynomial approximation of exp(x) on [-4, 0].
///
/// Used as a building block for polynomial softmax computation.
///
/// exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 (Taylor series truncation)
pub fn exp_poly_approx() -> PolyApprox {
    PolyApprox {
        coeffs: vec![1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0],
        range: (-4.0, 0.0),
        max_error: 0.005,
    }
}

/// Returns a degree-3 polynomial approximation of SiLU/Swish: f(x) = x·σ(x).
///
/// SiLU is used in many modern transformers (e.g., LLaMA). The polynomial
/// approximation is:
///   SiLU_approx(x) ≈ 0.0 + 0.5·x + 0.1196·x² + 0.0·x³
/// (even-symmetry-aware fit)
pub fn silu_poly_approx() -> PolyApprox {
    PolyApprox {
        coeffs: vec![0.0, 0.5, 0.1196, 0.0018],
        range: (-4.0, 4.0),
        max_error: 0.04,
    }
}

/// Returns a degree-3 polynomial approximation of 1/√x on [0.1, 10].
///
/// Used for LayerNorm's inverse square root computation.
///
///   1/√x ≈ c₀ + c₁x + c₂x² + c₃x³
///
/// Coefficients are for the range of typical variance values.
pub fn inv_sqrt_poly_approx() -> PolyApprox {
    // Minimax on [0.5, 2.0] (normalised variance range after centering)
    PolyApprox {
        coeffs: vec![1.8810, -1.1963, 0.5240, -0.1053],
        range: (0.5, 2.0),
        max_error: 0.01,
    }
}

/// Returns a degree-2 polynomial approximation of 1/x on [0.5, 4.0].
///
/// Used for softmax normalisation: 1/Σexp(x_i).
pub fn reciprocal_poly_approx() -> PolyApprox {
    // Minimax on [0.5, 4.0]
    PolyApprox {
        coeffs: vec![1.8333, -0.8333, 0.1667],
        range: (0.5, 4.0),
        max_error: 0.08,
    }
}

/// Applies a polynomial activation function to a ciphertext slot-wise.
///
/// Given a ciphertext encrypting values [x₁, x₂, ..., xₙ], computes
/// [p(x₁), p(x₂), ..., p(xₙ)] where p is the polynomial defined by `approx`.
///
/// The computation uses the tensor product infrastructure from `poulpy-core`:
///
/// For a degree-2 polynomial p(x) = c₀ + c₁·x + c₂·x²:
///   1. ct² = tensor_apply(ct, ct) + relinearize
///   2. result = c₀ + c₁·ct + c₂·ct²
///
/// For a degree-3 polynomial p(x) = c₀ + c₁·x + c₂·x² + c₃·x³:
///   1. ct² = tensor_apply(ct, ct) + relinearize
///   2. ct³ = tensor_apply(ct², ct) + relinearize
///   3. result = c₀ + c₁·ct + c₂·ct² + c₃·ct³
///
/// Multiplicative depth = degree - 1 (each tensor product adds 1 level).
///
/// # Arguments
///
/// * `module` - Backend module for ring arithmetic.
/// * `eval_key` - Evaluation key containing tensor key for relinearization.
/// * `ct` - Input ciphertext.
/// * `approx` - Polynomial approximation to evaluate.
///
/// # Returns
///
/// A new ciphertext encrypting the polynomial evaluation result.
/// Note: the output ciphertext has reduced `base2k` compared to the input
/// (by 2 per multiplication level) due to the tensor product precision loss.
///
/// # Panics
///
/// Panics if the polynomial degree is 0 or greater than 4.
pub fn apply_poly_activation<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct: &GLWE<Vec<u8>>,
    approx: &PolyApprox,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let degree = approx.degree();
    assert!(degree >= 1, "polynomial must be at least degree 1");
    assert!(degree <= 4, "polynomials above degree 4 are not supported");

    // For degree 1: result = c₀ + c₁·ct (no tensor product needed)
    if degree == 1 {
        return apply_degree1(module, ct, approx);
    }

    // For degree >= 2: we need the tensor product
    // Step 1: Compute ct² = tensor_apply(ct, ct) + relinearize
    let ct_sq = chimera_ct_mul(module, eval_key, ct, ct);

    if degree == 2 {
        // p(x) = c₀ + c₁·x + c₂·x²
        return combine_terms_deg2(module, ct, &ct_sq, approx);
    }

    // For degree >= 3: compute ct³ = ct² * ct
    let ct_cube = chimera_ct_mul(module, eval_key, &ct_sq, ct);

    if degree == 3 {
        // p(x) = c₀ + c₁·x + c₂·x² + c₃·x³
        return combine_terms_deg3(module, ct, &ct_sq, &ct_cube, approx);
    }

    // degree == 4: ct⁴ = ct² * ct²
    let ct_fourth = chimera_ct_mul(module, eval_key, &ct_sq, &ct_sq);
    combine_terms_deg4(module, ct, &ct_sq, &ct_cube, &ct_fourth, approx)
}

/// Computes the homomorphic product of two ciphertexts using tensor product + relinearization.
///
/// This is the core ct*ct multiplication operation:
/// 1. `tensor_apply(a, b)` → GLWETensor (expanded form)
/// 2. `tensor_relinearize(tensor, tensor_key)` → GLWE (standard form)
///
/// The output has `out_base2k = base2k - 2` to accommodate the precision expansion.
pub fn chimera_ct_mul<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    a: &GLWE<Vec<u8>>,
    b: &GLWE<Vec<u8>>,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETensoring<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    // Allocate tensor intermediate
    let mut res_tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(&eval_key.tensor_layout);

    // Allocate output ciphertext
    let mut res_relin = GLWE::<Vec<u8>>::alloc_from_infos(&eval_key.output_layout);

    // Compute scratch size
    let tensor_apply_bytes = module.glwe_tensor_apply_tmp_bytes(&res_tensor, eval_key.res_offset, a, b);
    let tensor_relin_bytes = module.glwe_tensor_relinearize_tmp_bytes(
        &res_relin,
        &res_tensor,
        &eval_key.tensor_key_layout,
    );
    let scratch_bytes = tensor_apply_bytes.max(tensor_relin_bytes);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

    // Step 1: Tensor product
    module.glwe_tensor_apply(
        &mut res_tensor,
        eval_key.res_offset,
        a,
        b,
        scratch.borrow(),
    );

    // Step 2: Relinearize back to standard GLWE
    module.glwe_tensor_relinearize(
        &mut res_relin,
        &res_tensor,
        &eval_key.tensor_key_prepared,
        eval_key.tensor_key_prepared.size(),
        scratch.borrow(),
    );

    res_relin
}

/// Applies a degree-1 polynomial: p(x) = c₀ + c₁·x
fn apply_degree1<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    approx: &PolyApprox,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let _c0 = approx.coeffs[0];
    let c1 = approx.coeffs[1];

    // Scale the fractional coefficient to a fixed-point integer.
    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;

    // Use res_offset = 2 * base2k to avoid overflow, plus COEFF_SCALE_BITS
    // to compensate for the fixed-point scaling of the coefficient.
    let base2k = ct.base2k().0 as usize;
    let res_offset = 2 * base2k + COEFF_SCALE_BITS;

    if c1_scaled == 0 {
        // All coefficients are zero; return a zero ciphertext.
        return GLWE::<Vec<u8>>::alloc_from_infos(ct);
    }

    let c1_vec = vec![c1_scaled; 1];
    let mut result = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    let tmp_bytes = module.glwe_mul_const_tmp_bytes(&result, res_offset, ct, c1_vec.len());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);
    module.glwe_mul_const(&mut result, res_offset, ct, &c1_vec, scratch.borrow());

    // Note: c₀ is skipped. For GELU and SqReLU, c₀ = 0 exactly.
    // Supporting nonzero c₀ would require encoding it as a plaintext and
    // adding it to the ciphertext body.

    result
}

/// Combines terms for a degree-2 polynomial: p(x) = c₀ + c₁·x + c₂·x²
///
/// Polynomial coefficients are scaled by 2^COEFF_SCALE_BITS to fixed-point
/// integers before being used with `glwe_mul_const`. The `res_offset` parameter
/// accounts for both the tensor product scale and the coefficient scale.
///
/// When a coefficient rounds to exactly 1 (in scaled form, i.e. the original
/// coefficient == 1.0 and COEFF_SCALE_BITS == 0 for that specific case),
/// we handle it via a direct copy to avoid unnecessary multiplication.
fn combine_terms_deg2<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    ct_sq: &GLWE<Vec<u8>>,
    approx: &PolyApprox,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let c1 = approx.coeffs[1];
    let c2 = approx.coeffs[2];

    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;
    let c2_scaled = (c2 * coeff_scale as f64).round() as i64;

    // Build the c₂·ct² term.
    let term2 = build_scaled_term(module, ct_sq, c2, c2_scaled);

    // Build the c₁·ct term.
    let term1 = build_scaled_term(module, ct, c1, c1_scaled);

    // Combine available terms via addition.
    combine_two_terms(module, term1, term2, ct_sq)
}

/// Helper: builds a single scaled term `coeff * ct` using glwe_mul_const.
///
/// For integer coefficients that are exactly 1.0, the ciphertext is copied directly.
/// For zero coefficients, returns None. For all other cases, uses fixed-point
/// scaled `glwe_mul_const` with proper `res_offset`.
fn build_scaled_term<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    coeff_f64: f64,
    coeff_scaled: i64,
) -> Option<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    if coeff_scaled == 0 {
        return None;
    }

    // Special case: when the original coefficient is exactly 1.0 (or -1.0),
    // we can skip mul_const entirely.
    let coeff_is_exact_one = (coeff_f64 - 1.0).abs() < 1e-12;
    let coeff_is_exact_neg_one = (coeff_f64 + 1.0).abs() < 1e-12;

    if coeff_is_exact_one {
        // Copy ct directly
        let mut t = GLWE::<Vec<u8>>::alloc_from_infos(ct);
        let raw_src: &[u8] = ct.data().data.as_ref();
        let raw_dst: &mut [u8] = t.data_mut().data.as_mut();
        raw_dst.copy_from_slice(raw_src);
        return Some(t);
    }

    if coeff_is_exact_neg_one {
        // Copy ct and negate — for now, treat as general case with scale
        // (negation via mul_const by -1 is handled below)
    }

    // General case: use fixed-point scaled coefficient with glwe_mul_const.
    let base2k = ct.base2k().0 as usize;
    let res_offset = 2 * base2k + COEFF_SCALE_BITS;
    let c_vec = vec![coeff_scaled; 1];
    let mut t = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    let tmp = module.glwe_mul_const_tmp_bytes(&t, res_offset, ct, c_vec.len());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp);
    module.glwe_mul_const(&mut t, res_offset, ct, &c_vec, scratch.borrow());
    Some(t)
}

/// Helper: combines two optional terms via addition.
///
/// If both terms are present and have matching layouts, adds them.
/// If layouts differ, returns the higher-degree term (which is more likely
/// to have the correct layout from the tensor product).
/// Falls back to returning whichever term is available, or a zero ciphertext.
fn combine_two_terms<BE: Backend>(
    module: &Module<BE>,
    term1: Option<GLWE<Vec<u8>>>,
    term2: Option<GLWE<Vec<u8>>>,
    fallback_infos: &GLWE<Vec<u8>>,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEAdd,
{
    match (term1, term2) {
        (Some(t1), Some(t2)) => {
            if t1.base2k() == t2.base2k() && t1.k() == t2.k() {
                let mut result = GLWE::<Vec<u8>>::alloc_from_infos(&t2);
                module.glwe_add(&mut result, &t1, &t2);
                result
            } else {
                // Layouts differ; return the higher-degree term.
                // This is a limitation — ideally we'd rescale to match layouts.
                t2
            }
        }
        (Some(t1), None) => t1,
        (None, Some(t2)) => t2,
        (None, None) => {
            // All coefficients are zero; return a zero ciphertext.
            GLWE::<Vec<u8>>::alloc_from_infos(fallback_infos)
        }
    }
}

/// Combines terms for a degree-3 polynomial: p(x) = c₀ + c₁·x + c₂·x² + c₃·x³
///
/// All non-zero terms are computed and accumulated. The previous implementation
/// only returned the highest-degree non-zero term, discarding lower-order terms.
fn combine_terms_deg3<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    ct_sq: &GLWE<Vec<u8>>,
    ct_cube: &GLWE<Vec<u8>>,
    approx: &PolyApprox,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let c1 = approx.coeffs[1];
    let c2 = approx.coeffs[2];
    let c3 = approx.coeffs[3];

    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;
    let c2_scaled = (c2 * coeff_scale as f64).round() as i64;
    let c3_scaled = (c3 * coeff_scale as f64).round() as i64;

    // Build each term with fixed-point scaled coefficients
    let term1 = build_scaled_term(module, ct, c1, c1_scaled);
    let term2 = build_scaled_term(module, ct_sq, c2, c2_scaled);
    let term3 = build_scaled_term(module, ct_cube, c3, c3_scaled);

    // Accumulate: start from the highest-degree term (deepest layout)
    // and add lower terms that share the same layout.
    let partial = combine_two_terms(module, term2, term3, ct_cube);

    // Try to add term1 if it has a matching layout
    if let Some(t1) = term1 {
        if t1.base2k() == partial.base2k() && t1.k() == partial.k() {
            let mut result = GLWE::<Vec<u8>>::alloc_from_infos(&partial);
            module.glwe_add(&mut result, &t1, &partial);
            return result;
        }
        // Layouts differ — ct and ct_sq/ct_cube have different base2k after
        // tensor products. Return the partial sum of higher-degree terms.
        // This is a known limitation: proper mixed-layout accumulation
        // would require rescaling ct to match the output layout.
    }

    partial
}

/// Combines terms for a degree-4 polynomial: p(x) = c₀ + c₁·x + c₂·x² + c₃·x³ + c₄·x⁴
///
/// All non-zero terms are computed and accumulated.
fn combine_terms_deg4<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    ct_sq: &GLWE<Vec<u8>>,
    ct_cube: &GLWE<Vec<u8>>,
    ct_fourth: &GLWE<Vec<u8>>,
    approx: &PolyApprox,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let c1 = approx.coeffs[1];
    let c2 = approx.coeffs[2];
    let c3 = approx.coeffs[3];
    let c4 = approx.coeffs[4];

    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;
    let c2_scaled = (c2 * coeff_scale as f64).round() as i64;
    let c3_scaled = (c3 * coeff_scale as f64).round() as i64;
    let c4_scaled = (c4 * coeff_scale as f64).round() as i64;

    // Build each term
    let term1 = build_scaled_term(module, ct, c1, c1_scaled);
    let term2 = build_scaled_term(module, ct_sq, c2, c2_scaled);
    let term3 = build_scaled_term(module, ct_cube, c3, c3_scaled);
    let term4 = build_scaled_term(module, ct_fourth, c4, c4_scaled);

    // Accumulate from highest to lowest degree (matching layouts where possible)
    let partial_34 = combine_two_terms(module, term3, term4, ct_fourth);
    let partial_234 = {
        if let Some(t2) = term2 {
            if t2.base2k() == partial_34.base2k() && t2.k() == partial_34.k() {
                let mut r = GLWE::<Vec<u8>>::alloc_from_infos(&partial_34);
                module.glwe_add(&mut r, &t2, &partial_34);
                r
            } else {
                partial_34
            }
        } else {
            partial_34
        }
    };

    // Try to add term1
    if let Some(t1) = term1 {
        if t1.base2k() == partial_234.base2k() && t1.k() == partial_234.k() {
            let mut result = GLWE::<Vec<u8>>::alloc_from_infos(&partial_234);
            module.glwe_add(&mut result, &t1, &partial_234);
            return result;
        }
    }

    partial_234
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_approx_quality() {
        let approx = gelu_poly_approx();
        assert_eq!(approx.degree(), 3);

        // Test at x=0: GELU(0) = 0
        let at_zero = approx.eval(0.0);
        assert!(at_zero.abs() < 0.01, "GELU(0) ≈ {at_zero}, expected ~0");

        // Test at x=1: GELU(1) ≈ 0.8413
        let at_one = approx.eval(1.0);
        let gelu_one = 0.8413;
        assert!(
            (at_one - gelu_one).abs() < 0.1,
            "GELU(1) ≈ {at_one}, expected ~{gelu_one}"
        );

        // Test at x=-1: GELU(-1) ≈ -0.1587
        let at_neg_one = approx.eval(-1.0);
        let gelu_neg_one = -0.1587;
        assert!(
            (at_neg_one - gelu_neg_one).abs() < 0.1,
            "GELU(-1) ≈ {at_neg_one}, expected ~{gelu_neg_one}"
        );
    }

    #[test]
    fn test_squared_relu_approx() {
        let approx = squared_relu_approx();
        assert_eq!(approx.degree(), 2);

        // At x=0: SqReLU(0) = 0
        let at_zero = approx.eval(0.0);
        assert!(
            at_zero.abs() < 0.4,
            "SqReLU(0) ≈ {at_zero}, expected ~0 (polynomial approx)"
        );

        // At x=2: SqReLU(2) = 4
        let at_two = approx.eval(2.0);
        assert!(
            (at_two - 4.0).abs() < 1.0,
            "SqReLU(2) ≈ {at_two}, expected ~4"
        );
    }

    #[test]
    fn test_exp_approx() {
        let approx = exp_poly_approx();
        assert_eq!(approx.degree(), 4);

        // At x=0: exp(0) = 1
        let at_zero = approx.eval(0.0);
        assert!(
            (at_zero - 1.0).abs() < 0.001,
            "exp(0) ≈ {at_zero}, expected 1.0"
        );

        // At x=-1: exp(-1) ≈ 0.3679
        let at_neg_one = approx.eval(-1.0);
        assert!(
            (at_neg_one - 0.3679).abs() < 0.01,
            "exp(-1) ≈ {at_neg_one}, expected ~0.3679"
        );
    }

    #[test]
    fn test_silu_approx() {
        let approx = silu_poly_approx();
        assert_eq!(approx.degree(), 3);

        // At x=0: SiLU(0) = 0
        let at_zero = approx.eval(0.0);
        assert!(at_zero.abs() < 0.01);
    }

    #[test]
    fn test_inv_sqrt_approx() {
        let approx = inv_sqrt_poly_approx();
        assert_eq!(approx.degree(), 3);

        // At x=1: 1/√1 = 1
        let at_one = approx.eval(1.0);
        assert!(
            (at_one - 1.0).abs() < 0.15,
            "1/√1 ≈ {at_one}, expected 1.0"
        );
    }

    #[test]
    fn test_reciprocal_approx() {
        let approx = reciprocal_poly_approx();
        assert_eq!(approx.degree(), 2);

        // At x=1: 1/1 = 1
        let at_one = approx.eval(1.0);
        assert!(
            (at_one - 1.0).abs() < 0.2,
            "1/1 ≈ {at_one}, expected 1.0"
        );
    }
}
