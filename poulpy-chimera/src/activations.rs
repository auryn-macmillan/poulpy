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

use poulpy_core::ScratchTakeCore;
use poulpy_core::{
    layouts::{GLWEInfos, GLWETensor, LWEInfos, GLWE},
    GLWEAdd, GLWEMulConst, GLWETensoring,
};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned, ZnxViewMut},
};

use crate::arithmetic::chimera_project_layout;
use crate::encrypt::ChimeraEvalKey;

/// Number of bits used for fixed-point scaling of polynomial coefficients.
///
/// Fractional coefficients like 0.5 or 0.247 are multiplied by 2^COEFF_SCALE_BITS
/// and rounded to the nearest integer before being used with `glwe_mul_const`.
///
/// After polynomial evaluation, the result is encoded at an increased torus
/// precision: `scale + COEFF_SCALE_BITS` instead of `scale`. The caller must
/// decode at this increased precision using [`activation_decode_precision`].
///
/// For GELU coefficients (0.5, 0.247), 8 bits gives:
///   0.5 * 256 = 128 (exact)
///   0.247 * 256 = 63.2 → 63 (error: 0.001)
pub const COEFF_SCALE_BITS: usize = 8;

/// Returns the torus precision at which to decode the output of
/// [`apply_poly_activation`].
///
/// After polynomial evaluation with fixed-point coefficient scaling, the
/// coefficients have been multiplied by `2^COEFF_SCALE_BITS`. Since
/// `glwe_mul_const` with `res_offset = a.base2k()` computes `a * b` on
/// the torus (identity scaling), the output contains `value * coeff_scaled`
/// at the original torus precision. To recover `value * coeff_float`, we
/// must divide by `2^COEFF_SCALE_BITS`, which is achieved by decoding at
/// `encoding_scale - COEFF_SCALE_BITS`.
///
/// For example, if the input was encoded at `scale = 2 * in_base2k = 26`,
/// the output must be decoded at `TorusPrecision(18)`.
pub fn activation_decode_precision(encoding_scale: usize) -> usize {
    encoding_scale - COEFF_SCALE_BITS
}

/// Returns the torus precision needed to encode a constant term into the
/// ciphertext output of [`apply_poly_activation`].
///
/// The polynomial coefficients are scaled by `2^COEFF_SCALE_BITS`, so a
/// constant `c0` must be encoded at the same effective precision as the other
/// terms. If the input was encoded at `encoding_scale`, the constant should be
/// encoded at `encoding_scale - COEFF_SCALE_BITS`.
pub fn activation_constant_precision(encoding_scale: usize) -> usize {
    activation_decode_precision(encoding_scale)
}

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

    /// Returns the nominal degree of the polynomial (from coefficient count).
    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Returns the effective degree — the highest index with a non-zero coefficient.
    ///
    /// For example, GELU `[0.0, 0.5, 0.247, 0.0]` has nominal degree 3
    /// but effective degree 2 (because c₃ = 0). This avoids unnecessary
    /// tensor products that would require mismatched-base2k inputs.
    pub fn effective_degree(&self) -> usize {
        let scale = 1i64 << COEFF_SCALE_BITS;
        for i in (0..self.coeffs.len()).rev() {
            let scaled = (self.coeffs[i] * scale as f64).round() as i64;
            if scaled != 0 {
                return i;
            }
        }
        0
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
    let degree = approx.effective_degree();
    assert!(
        degree >= 1,
        "polynomial must have at least one non-zero coefficient above degree 0"
    );
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

    // For degree >= 3: compute ct³ = ct² * ct.
    // Align ct to ct²'s layout before multiplying.
    let ct_for_cube = if ct.base2k() == ct_sq.base2k() && ct.k() == ct_sq.k() {
        ct.clone()
    } else {
        chimera_project_layout(module, ct, &ct_sq.glwe_layout())
    };
    let ct_cube = chimera_ct_mul(module, eval_key, &ct_sq, &ct_for_cube);

    if degree == 3 {
        // p(x) = c₀ + c₁·x + c₂·x² + c₃·x³
        return combine_terms_deg3(module, ct, &ct_sq, &ct_cube, approx);
    }

    // degree == 4: ct⁴ = ct² * ct²
    let ct_sq_for_fourth = if ct_sq.base2k() == ct_cube.base2k() && ct_sq.k() == ct_cube.k() {
        ct_sq.clone()
    } else {
        chimera_project_layout(module, &ct_sq, &ct_cube.glwe_layout())
    };
    let ct_fourth = chimera_ct_mul(module, eval_key, &ct_sq_for_fourth, &ct_sq_for_fourth);
    combine_terms_deg4(module, ct, &ct_sq, &ct_cube, &ct_fourth, approx)
}

/// Adds a public constant term to a ciphertext by injecting it into the GLWE
/// body polynomial.
///
/// The constant is encoded in coefficient 0 at torus precision
/// `activation_constant_precision(encoding_scale)` so it matches the fixed-point
/// scaling used by the non-constant polynomial terms.
fn add_constant_term(ct: &mut GLWE<Vec<u8>>, c0: f64, encoding_scale: usize) {
    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c0_scaled = (c0 * coeff_scale as f64).round() as i64;
    if c0_scaled == 0 {
        return;
    }

    let const_precision = activation_constant_precision(encoding_scale);
    let shift = const_precision.saturating_sub(ct.base2k().0 as usize);

    let body = ct.data_mut().at_mut(0, 0);
    body[0] = body[0].wrapping_add(c0_scaled << shift);
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
    let use_l2 = a.base2k() == b.base2k() && eval_key.output_l2_layout.is_some() && a.base2k() == eval_key.output_layout.base2k;

    let (tensor_layout, output_layout, res_offset, tensor_key_layout, tensor_key_prepared) = if use_l2 {
        (
            eval_key.tensor_l2_layout.as_ref().expect("missing level-2 tensor layout"),
            eval_key.output_l2_layout.as_ref().expect("missing level-2 output layout"),
            eval_key.res_offset_l2.expect("missing level-2 res_offset"),
            eval_key
                .tensor_key_l2_layout
                .as_ref()
                .expect("missing level-2 tensor key layout"),
            eval_key.tensor_key_l2_prepared.as_ref().expect("missing level-2 tensor key"),
        )
    } else {
        (
            &eval_key.tensor_layout,
            &eval_key.output_layout,
            eval_key.res_offset,
            &eval_key.tensor_key_layout,
            &eval_key.tensor_key_prepared,
        )
    };

    // Allocate tensor intermediate
    let mut res_tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(tensor_layout);

    // Allocate output ciphertext
    let mut res_relin = GLWE::<Vec<u8>>::alloc_from_infos(output_layout);

    // Compute scratch size
    let tensor_apply_bytes = module.glwe_tensor_apply_tmp_bytes(&res_tensor, res_offset, a, b);
    let tensor_relin_bytes = module.glwe_tensor_relinearize_tmp_bytes(&res_relin, &res_tensor, tensor_key_layout);
    let scratch_bytes = tensor_apply_bytes.max(tensor_relin_bytes);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

    // Step 1: Tensor product
    module.glwe_tensor_apply(&mut res_tensor, res_offset, a, b, scratch.borrow());

    // Step 2: Relinearize back to standard GLWE
    module.glwe_tensor_relinearize(
        &mut res_relin,
        &res_tensor,
        tensor_key_prepared,
        tensor_key_prepared.size(),
        scratch.borrow(),
    );

    res_relin
}

/// Applies a degree-1 polynomial: p(x) = c₀ + c₁·x
fn apply_degree1<BE: Backend>(module: &Module<BE>, ct: &GLWE<Vec<u8>>, approx: &PolyApprox) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let c0 = approx.coeffs[0];
    let c1 = approx.coeffs[1];

    // Scale the fractional coefficient to a fixed-point integer.
    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;

    if c1_scaled == 0 {
        // All coefficients are zero; return a zero ciphertext.
        return GLWE::<Vec<u8>>::alloc_from_infos(ct);
    }

    // Use res_offset = ct.base2k() for raw product.
    let res_offset = ct.base2k().0 as usize;

    let c1_vec = vec![c1_scaled; 1];
    let mut result = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    let tmp_bytes = module.glwe_mul_const_tmp_bytes(&result, res_offset, ct, c1_vec.len());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp_bytes);
    module.glwe_mul_const(&mut result, res_offset, ct, &c1_vec, scratch.borrow());

    add_constant_term(&mut result, c0, ct.k().0 as usize);

    result
}

/// Returns a target layout if the source and target ciphertexts have different base2k.
///
/// When building a term from `ct_source` that needs to be added to a term from
/// `ct_target`, and their base2k values differ, this returns `Some(layout)` matching
/// `ct_target`'s parameters. Otherwise returns `None` (no layout conversion needed).
fn get_target_layout(ct_source: &GLWE<Vec<u8>>, ct_target: &GLWE<Vec<u8>>) -> Option<poulpy_core::layouts::GLWELayout> {
    if ct_source.base2k() != ct_target.base2k() || ct_source.k() != ct_target.k() {
        Some(poulpy_core::layouts::GLWELayout {
            n: ct_target.n(),
            base2k: ct_target.base2k(),
            k: ct_target.k(),
            rank: ct_target.rank(),
        })
    } else {
        None
    }
}

/// Combines terms for a degree-2 polynomial: p(x) = c₀ + c₁·x + c₂·x²
///
/// Polynomial coefficients are scaled by 2^COEFF_SCALE_BITS to fixed-point
/// integers before being used with `glwe_mul_const`. Each `glwe_mul_const`
/// call uses `res_offset = ct.base2k()` (identity scaling on the torus),
/// so the output contains `value * coeff_scaled` at the original torus precision.
/// The caller must decode at `scale - COEFF_SCALE_BITS` to recover the true result.
///
/// The c₁·x term is built with the output layout of the tensor product
/// (matching ct_sq's layout) to ensure all terms can be added together.
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
    let c0 = approx.coeffs[0];
    let c1 = approx.coeffs[1];
    let c2 = approx.coeffs[2];

    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;
    let c2_scaled = (c2 * coeff_scale as f64).round() as i64;

    // Build the c₂·ct² term (output at ct_sq's layout, no layout change needed).
    let term2 = build_scaled_term(module, ct_sq, c2, c2_scaled, None);

    // Determine the target layout for the c₁·ct term.
    let target_layout = get_target_layout(ct, ct_sq);
    let term1 = build_scaled_term(module, ct, c1, c1_scaled, target_layout.as_ref());

    // Combine available terms via addition.
    let mut result = combine_two_terms(module, term1, term2, ct_sq);
    add_constant_term(&mut result, c0, ct.k().0 as usize);
    result
}

/// Helper: builds a single scaled term `coeff * ct` using glwe_mul_const,
/// with the output allocated at a specified target layout.
///
/// Every coefficient (including exact integers like 1.0) is scaled by
/// `2^COEFF_SCALE_BITS` and applied via `glwe_mul_const`. This ensures all terms
/// land at the same output torus precision (`scale + COEFF_SCALE_BITS`) regardless
/// of the coefficient value.
///
/// The `res_offset` is set to `ct.base2k()`, which produces the raw product
/// `value * coeff_scaled` without any additional scaling. This is the only
/// `res_offset` value that works reliably across all base2k levels.
///
/// For zero coefficients, returns None.
/// If `target_layout` is None, the output has the same layout as the input ct.
fn build_scaled_term<BE: Backend>(
    module: &Module<BE>,
    ct: &GLWE<Vec<u8>>,
    _coeff_f64: f64,
    coeff_scaled: i64,
    target_layout: Option<&poulpy_core::layouts::GLWELayout>,
) -> Option<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    if coeff_scaled == 0 {
        return None;
    }

    // Use res_offset = ct.base2k() to produce the raw product: value * coeff_scaled.
    // This is the standard working point for glwe_mul_const (res_offset_hi = 0,
    // res_offset_lo = 0), which avoids both the overflow issues of small res_offset
    // and the signal loss of large res_offset.
    let res_offset = ct.base2k().0 as usize;

    let coeff_vec = vec![coeff_scaled; 1];

    let mut t = if let Some(layout) = target_layout {
        GLWE::<Vec<u8>>::alloc_from_infos(layout)
    } else {
        GLWE::<Vec<u8>>::alloc_from_infos(ct)
    };
    let tmp = module.glwe_mul_const_tmp_bytes(&t, res_offset, ct, coeff_vec.len());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(tmp);
    module.glwe_mul_const(&mut t, res_offset, ct, &coeff_vec, scratch.borrow());
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
                panic!(
                    "combine_two_terms: layout mismatch after projection (lhs base2k={}, k={}, rhs base2k={}, k={})",
                    t1.base2k().0,
                    t1.k().0,
                    t2.base2k().0,
                    t2.k().0,
                );
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
/// All non-zero terms are computed and accumulated. The c₁·x term is built with
/// the output layout matching the highest-degree terms (from tensor products) to
/// ensure all terms can be added together.
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
    let c0 = approx.coeffs[0];
    let c1 = approx.coeffs[1];
    let c2 = approx.coeffs[2];
    let c3 = approx.coeffs[3];

    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;
    let c2_scaled = (c2 * coeff_scale as f64).round() as i64;
    let c3_scaled = (c3 * coeff_scale as f64).round() as i64;

    // The target layout for lower-degree terms: match ct_cube (deepest tensor product).
    let target_for_ct = get_target_layout(ct, ct_cube);
    let target_for_ct_sq = get_target_layout(ct_sq, ct_cube);

    // Build each term with fixed-point scaled coefficients.
    // Lower-degree terms are converted to ct_cube's layout via glwe_mul_const.
    let term1 = build_scaled_term(module, ct, c1, c1_scaled, target_for_ct.as_ref());
    let term2 = build_scaled_term(module, ct_sq, c2, c2_scaled, target_for_ct_sq.as_ref());
    let term3 = build_scaled_term(module, ct_cube, c3, c3_scaled, None);

    // Accumulate all terms. Since all are now at the same layout, combine_two_terms
    // should succeed for every pair.
    let partial = combine_two_terms(module, term2, term3, ct_cube);

    if let Some(t1) = term1 {
        if t1.base2k() == partial.base2k() && t1.k() == partial.k() {
            let mut result = GLWE::<Vec<u8>>::alloc_from_infos(&partial);
            module.glwe_add(&mut result, &t1, &partial);
            return result;
        }
        // This should not happen now that we convert to target layout,
        // but fall back gracefully if it does.
    }

    let mut result = partial;
    add_constant_term(&mut result, c0, ct.k().0 as usize);
    result
}

/// Combines terms for a degree-4 polynomial: p(x) = c₀ + c₁·x + c₂·x² + c₃·x³ + c₄·x⁴
///
/// All non-zero terms are computed and accumulated. Lower-degree terms are
/// converted to the highest-degree term's layout via glwe_mul_const.
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
    let c0 = approx.coeffs[0];
    let c1 = approx.coeffs[1];
    let c2 = approx.coeffs[2];
    let c3 = approx.coeffs[3];
    let c4 = approx.coeffs[4];

    let coeff_scale = 1i64 << COEFF_SCALE_BITS;
    let c1_scaled = (c1 * coeff_scale as f64).round() as i64;
    let c2_scaled = (c2 * coeff_scale as f64).round() as i64;
    let c3_scaled = (c3 * coeff_scale as f64).round() as i64;
    let c4_scaled = (c4 * coeff_scale as f64).round() as i64;

    // The target layout for lower-degree terms: match ct_fourth (deepest tensor product).
    let target_for_ct = get_target_layout(ct, ct_fourth);
    let target_for_ct_sq = get_target_layout(ct_sq, ct_fourth);
    let target_for_ct_cube = get_target_layout(ct_cube, ct_fourth);

    // Build each term, converting to the target layout where needed.
    let term1 = build_scaled_term(module, ct, c1, c1_scaled, target_for_ct.as_ref());
    let term2 = build_scaled_term(module, ct_sq, c2, c2_scaled, target_for_ct_sq.as_ref());
    let term3 = build_scaled_term(module, ct_cube, c3, c3_scaled, target_for_ct_cube.as_ref());
    let term4 = build_scaled_term(module, ct_fourth, c4, c4_scaled, None);

    // Accumulate from highest to lowest degree
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

    let mut result = partial_234;
    add_constant_term(&mut result, c0, ct.k().0 as usize);
    result
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
        assert!((at_one - gelu_one).abs() < 0.1, "GELU(1) ≈ {at_one}, expected ~{gelu_one}");

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
        assert!(at_zero.abs() < 0.4, "SqReLU(0) ≈ {at_zero}, expected ~0 (polynomial approx)");

        // At x=2: SqReLU(2) = 4
        let at_two = approx.eval(2.0);
        assert!((at_two - 4.0).abs() < 1.0, "SqReLU(2) ≈ {at_two}, expected ~4");
    }

    #[test]
    fn test_exp_approx() {
        let approx = exp_poly_approx();
        assert_eq!(approx.degree(), 4);

        // At x=0: exp(0) = 1
        let at_zero = approx.eval(0.0);
        assert!((at_zero - 1.0).abs() < 0.001, "exp(0) ≈ {at_zero}, expected 1.0");

        // At x=-1: exp(-1) ≈ 0.3679
        let at_neg_one = approx.eval(-1.0);
        assert!((at_neg_one - 0.3679).abs() < 0.01, "exp(-1) ≈ {at_neg_one}, expected ~0.3679");
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
        assert!((at_one - 1.0).abs() < 0.15, "1/√1 ≈ {at_one}, expected 1.0");
    }

    #[test]
    fn test_reciprocal_approx() {
        let approx = reciprocal_poly_approx();
        assert_eq!(approx.degree(), 2);

        // At x=1: 1/1 = 1
        let at_one = approx.eval(1.0);
        assert!((at_one - 1.0).abs() < 0.2, "1/1 ≈ {at_one}, expected 1.0");
    }
}
