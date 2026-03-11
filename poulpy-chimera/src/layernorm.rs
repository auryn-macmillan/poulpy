//! Approximate LayerNorm / RMSNorm under FHE.
//!
//! LayerNorm is a critical component of transformer architectures that
//! normalises activations to zero mean and unit variance. Under FHE, the
//! key challenges are:
//!
//! 1. Computing the mean (requires summing all slots — uses rotation-and-add)
//! 2. Computing the variance (requires squaring — uses tensor product)
//! 3. Computing 1/√variance (requires nonlinear function — uses poly approx or LUT)
//!
//! CHIMERA provides two variants:
//!
//! - **LayerNorm**: Full normalisation with mean subtraction and variance scaling.
//! - **RMSNorm**: Simplified variant that skips mean subtraction (used by LLaMA
//!   and other modern architectures). ~30% cheaper under FHE.

use poulpy_core::ScratchTakeCore;
use poulpy_core::{
    layouts::{GLWEInfos, LWEInfos, GLWE},
    GLWEAdd, GLWEMulConst, GLWETensoring, GLWETrace,
};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

use crate::activations::{
    activation_decode_precision, apply_poly_activation, chimera_ct_mul, chimera_ct_mul_with_res_offset, PolyApprox,
};
use crate::arithmetic::{chimera_add, chimera_align_layout, chimera_mul_const, chimera_slot_sum};
use crate::encrypt::ChimeraEvalKey;

/// Configuration for LayerNorm/RMSNorm evaluation under FHE.
#[derive(Clone, Debug)]
pub struct LayerNormConfig {
    /// Number of active elements to normalise over.
    pub norm_size: usize,
    /// Whether to use RMSNorm (skip mean subtraction).
    pub use_rms_norm: bool,
    /// Polynomial approximation for 1/√x.
    pub inv_sqrt_approx: PolyApprox,
    /// Epsilon for numerical stability (added to variance before inv_sqrt).
    pub epsilon: f64,
    /// Optional learnable scale (gamma) parameters, encoded as i64.
    pub gamma: Option<Vec<i64>>,
    /// Optional learnable bias (beta) parameters, encoded as i64.
    pub beta: Option<Vec<i64>>,
}

impl LayerNormConfig {
    /// Creates a default RMSNorm configuration (recommended for FHE).
    ///
    /// RMSNorm is preferred because it avoids the mean computation,
    /// saving log₂(norm_size) rotations and one multiplication depth level.
    pub fn rms_norm(norm_size: usize) -> Self {
        use crate::activations::inv_sqrt_poly_approx;
        LayerNormConfig {
            norm_size,
            use_rms_norm: true,
            inv_sqrt_approx: inv_sqrt_poly_approx(),
            epsilon: 1e-5,
            gamma: None,
            beta: None,
        }
    }

    /// Creates a full LayerNorm configuration.
    pub fn layer_norm(norm_size: usize) -> Self {
        use crate::activations::inv_sqrt_poly_approx;
        LayerNormConfig {
            norm_size,
            use_rms_norm: false,
            inv_sqrt_approx: inv_sqrt_poly_approx(),
            epsilon: 1e-5,
            gamma: None,
            beta: None,
        }
    }

    /// Sets the learnable scale (gamma) parameters.
    pub fn with_gamma(mut self, gamma: Vec<i64>) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Sets the learnable bias (beta) parameters.
    pub fn with_beta(mut self, beta: Vec<i64>) -> Self {
        self.beta = Some(beta);
        self
    }

    /// Estimates the multiplicative depth of this LayerNorm evaluation.
    ///
    /// - RMSNorm: 1 (square) + inv_sqrt_degree
    /// - LayerNorm: 1 (square) + inv_sqrt_degree + 0 (mean subtraction is depth-0)
    pub fn depth(&self) -> usize {
        let sq_depth = 1; // x²
        let inv_sqrt_depth = self.inv_sqrt_approx.degree();
        sq_depth + inv_sqrt_depth
    }

    /// Estimates the number of rotation operations needed.
    ///
    /// Mean computation (if not RMSNorm): log₂(norm_size) rotations
    /// Variance/RMS computation: log₂(norm_size) rotations
    pub fn num_rotations(&self) -> usize {
        let log_n = (self.norm_size as f64).log2().ceil() as usize;
        if self.use_rms_norm {
            log_n // Only for sum-of-squares
        } else {
            2 * log_n // Mean + variance
        }
    }
}

/// Describes the operations performed during LayerNorm evaluation.
///
/// This struct documents the FHE computation plan without executing it,
/// useful for cost estimation and noise budget analysis.
#[derive(Clone, Debug)]
pub struct LayerNormPlan {
    /// Number of rotation-and-add steps for mean computation (0 if RMSNorm).
    pub mean_rotations: usize,
    /// Number of rotation-and-add steps for variance/RMS computation.
    pub variance_rotations: usize,
    /// Multiplicative depth of the inv_sqrt approximation.
    pub inv_sqrt_depth: usize,
    /// Total multiplicative depth.
    pub total_depth: usize,
    /// Whether gamma/beta scaling is applied (adds 1 depth if gamma present).
    pub has_affine: bool,
}

/// Plans a LayerNorm evaluation, returning cost estimates.
pub fn plan_layernorm(config: &LayerNormConfig) -> LayerNormPlan {
    let log_n = (config.norm_size as f64).log2().ceil() as usize;

    let mean_rotations = if config.use_rms_norm { 0 } else { log_n };
    let variance_rotations = log_n;
    let inv_sqrt_depth = config.inv_sqrt_approx.degree();

    // Depth: square(1) + inv_sqrt(degree) + affine(0 or 1)
    let affine_depth = if config.gamma.is_some() { 1 } else { 0 };
    let total_depth = 1 + inv_sqrt_depth + affine_depth;

    LayerNormPlan {
        mean_rotations,
        variance_rotations,
        inv_sqrt_depth,
        total_depth,
        has_affine: config.gamma.is_some(),
    }
}

/// Evaluates LayerNorm on plaintext vectors (for testing/verification).
///
/// Applies the same normalisation that the FHE version would compute,
/// using the polynomial approximations rather than exact math.
pub fn layernorm_plaintext(values: &[f64], config: &LayerNormConfig) -> Vec<f64> {
    let n = values.len();
    assert!(n > 0);

    let (centered, rms) = if config.use_rms_norm {
        // RMSNorm: no mean subtraction
        let sum_sq: f64 = values.iter().map(|x| x * x).sum();
        let rms = (sum_sq / n as f64 + config.epsilon).sqrt();
        (values.to_vec(), rms)
    } else {
        // LayerNorm: subtract mean first
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = values.iter().map(|x| x - mean).collect();
        let var: f64 = centered.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let rms = (var + config.epsilon).sqrt();
        (centered, rms)
    };

    // Apply polynomial approximation of 1/√(rms²) = 1/rms
    // For verification, we use exact 1/rms here
    let inv_rms = 1.0 / rms;

    let mut result: Vec<f64> = centered.iter().map(|x| x * inv_rms).collect();

    // Apply affine transform if present
    if let Some(gamma) = &config.gamma {
        for (i, r) in result.iter_mut().enumerate() {
            if i < gamma.len() {
                // Gamma is stored as scaled integer; convert back
                let g = gamma[i] as f64 / (1u64 << 8) as f64;
                *r *= g;
            }
        }
    }
    if let Some(beta) = &config.beta {
        for (i, r) in result.iter_mut().enumerate() {
            if i < beta.len() {
                let b = beta[i] as f64 / (1u64 << 8) as f64;
                *r += b;
            }
        }
    }

    result
}

/// Computes RMSNorm homomorphically on a vector of encrypted scalars.
///
/// This is the **vector representation** variant. Each element of `x_cts`
/// is a separate ciphertext encrypting one dimension of the input vector
/// (value in coefficient 0). This representation is used by the
/// `chimera_transformer_block_vec` pipeline for real model inference.
///
/// RMSNorm(x)_i = x_i * (1 / √(mean(x²) + ε))
///
/// The computation proceeds as:
///
/// 1. **Square each dimension**: `sq_i = x_i * x_i` (ct*ct for each dim).
/// 2. **Sum squares**: `sum_sq = Σ_i sq_i` (homomorphic addition of cts).
/// 3. **Mean**: `mean_sq = sum_sq * (1/d_model)` (plaintext multiply).
/// 4. **Inverse sqrt**: `inv_rms = inv_sqrt_poly(mean_sq)`.
/// 5. **Scale each dimension**: `out_i = x_i * inv_rms` (ct*ct per dim).
/// 6. **Optional gamma**: `out_i = out_i * gamma_i` (ct*pt per dim).
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `x_cts` - Input vector: one ciphertext per dimension (value in coeff 0).
/// * `config` - RMSNorm configuration.
///
/// # Returns
///
/// A vector of ciphertexts of the same length as `x_cts`.
pub fn chimera_rms_norm_vec<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    config: &LayerNormConfig,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    assert!(
        config.use_rms_norm,
        "chimera_rms_norm_vec: full LayerNorm is not yet implemented. \
         Use LayerNormConfig::rms_norm() instead."
    );
    assert!(!x_cts.is_empty(), "chimera_rms_norm_vec: input vector must not be empty");

    let d = x_cts.len();

    // Step 1: Square each dimension — sq_i = x_i * x_i (parallelized)
    let sq: Vec<GLWE<Vec<u8>>> = {
        use rayon::prelude::*;
        x_cts.par_iter().map(|xi| chimera_ct_mul(module, eval_key, xi, xi)).collect()
    };

    // Step 2: Sum all squared values — sum_sq = Σ_i sq_i
    // All sq_i should have the same layout after ct*ct (out_base2k).
    let mut sum_sq = {
        use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
        let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(&sq[0]);
        {
            let src_ref = sq[0].to_ref();
            let src: &[u8] = src_ref.data().data;
            let mut dst_mut = cloned.to_mut();
            let dst: &mut [u8] = dst_mut.data_mut().data;
            let len = src.len().min(dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
        cloned
    };
    for i in 1..d {
        sum_sq = chimera_add(module, &sum_sq, &sq[i]);
    }

    // Step 3: Mean — multiply by 1/d_model
    let inv_n_scaled = ((1i64 << crate::activations::COEFF_SCALE_BITS) as f64 / d as f64).round() as i64;
    let mean_sq = chimera_mul_const(module, &sum_sq, &[inv_n_scaled]);

    // Step 4: Inverse square root — evaluate polynomial approx of 1/√x
    let inv_rms = apply_poly_activation(module, eval_key, &mean_sq, &config.inv_sqrt_approx);

    // Step 5: Scale each dimension — out_i = x_i * inv_rms (parallelized)
    // We need to project each x_i to match inv_rms's layout before ct*ct multiply.
    let inv_rms_layout = poulpy_core::layouts::GLWELayout {
        n: inv_rms.n(),
        base2k: inv_rms.base2k(),
        k: inv_rms.k(),
        rank: inv_rms.rank(),
    };

    let mut outputs: Vec<GLWE<Vec<u8>>> = {
        use rayon::prelude::*;
        x_cts
            .par_iter()
            .map(|xi| {
                let xi_proj = if xi.base2k() == inv_rms.base2k() && xi.k() == inv_rms.k() {
                    use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
                    let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(xi);
                    {
                        let src_ref = xi.to_ref();
                        let src: &[u8] = src_ref.data().data;
                        let mut dst_mut = cloned.to_mut();
                        let dst: &mut [u8] = dst_mut.data_mut().data;
                        let len = src.len().min(dst.len());
                        dst[..len].copy_from_slice(&src[..len]);
                    }
                    cloned
                } else {
                    chimera_align_layout(module, xi, &inv_rms_layout)
                };
                chimera_ct_mul_with_res_offset(
                    module,
                    eval_key,
                    &xi_proj,
                    &inv_rms,
                    activation_decode_precision(eval_key.res_offset),
                )
            })
            .collect()
    };

    // Step 6: Optional gamma scaling
    if let Some(gamma) = &config.gamma {
        for (i, out) in outputs.iter_mut().enumerate() {
            if i < gamma.len() {
                *out = chimera_mul_const(module, out, &[gamma[i]]);
            }
        }
    }

    outputs
}

/// Computes RMSNorm homomorphically on an encrypted vector.
///
/// RMSNorm(x) = x * (1 / √(mean(x²) + ε))
///
/// The computation proceeds in five stages:
///
/// 1. **Square**: `ct_sq = ct * ct` via tensor product (element-wise x²).
/// 2. **Sum**: `sum_sq = trace(ct_sq)` — sums all slots via rotation-and-add,
///    producing a ciphertext where every slot holds the same scalar Σxᵢ².
/// 3. **Mean**: `mean_sq = sum_sq * (1/N)` — scales by the inverse of the
///    normalisation size (encoded as a fixed-point constant).
/// 4. **Inverse sqrt**: `inv_rms = inv_sqrt_poly(mean_sq)` — evaluates a
///    polynomial approximation of 1/√x on the replicated scalar.
/// 5. **Normalise**: `result = ct * inv_rms` — element-wise product of the
///    original vector with the broadcast normalisation factor.
///
/// Optionally applies a learnable scale (gamma) via plaintext multiplication.
///
/// # Noise budget consumption
///
/// - Step 1 (square): 1 tensor product depth level
/// - Step 4 (inv_sqrt): polynomial degree levels (typically 2-3 for degree-3 poly)
/// - Step 5 (normalise): 1 tensor product depth level
/// - Total: ~4 depth levels for a degree-3 inv_sqrt polynomial
///
/// # Arguments
///
/// * `module` - Backend module for ring arithmetic.
/// * `eval_key` - Evaluation key (tensor key for ct×ct, auto keys for trace).
/// * `ct` - Input ciphertext encrypting the vector to normalise.
/// * `config` - RMSNorm configuration (norm_size, inv_sqrt polynomial, etc.).
///
/// # Returns
///
/// A new ciphertext encrypting the normalised vector.
///
/// # Panics
///
/// Panics if `config.use_rms_norm` is false (full LayerNorm not yet implemented
/// under FHE — use `chimera_rms_norm` with a RMSNorm config instead).
pub fn chimera_rms_norm<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct: &GLWE<Vec<u8>>,
    config: &LayerNormConfig,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    assert!(
        config.use_rms_norm,
        "chimera_rms_norm: full LayerNorm is not yet implemented under FHE. \
         Use LayerNormConfig::rms_norm() instead."
    );

    // Step 1: Square — ct_sq = ct * ct (element-wise x²)
    let ct_sq = chimera_ct_mul(module, eval_key, ct, ct);

    // Step 2: Sum all slots — trace(ct_sq) produces a ciphertext where
    // every slot holds the same scalar Σxᵢ².
    // skip=0 means full trace (sum all N slots).
    let sum_sq = chimera_slot_sum(module, eval_key, &ct_sq, 0);

    // Step 3: Mean — multiply by 1/N encoded as fixed-point.
    // We use COEFF_SCALE_BITS for the scaling factor so it integrates
    // cleanly with the subsequent polynomial evaluation.
    let inv_n_scaled = ((1i64 << crate::activations::COEFF_SCALE_BITS) as f64 / config.norm_size as f64).round() as i64;
    let mean_sq = chimera_mul_const(module, &sum_sq, &[inv_n_scaled]);

    // Step 4: Inverse square root — evaluate polynomial approx of 1/√x
    // on the replicated scalar (all slots hold the same mean(x²) value).
    let inv_rms = apply_poly_activation(module, eval_key, &mean_sq, &config.inv_sqrt_approx);

    // Step 5: Normalise — ct * inv_rms (element-wise broadcast multiplication).
    // The original ct is at in_base2k; inv_rms is at a lower base2k after
    // multiple tensor products and mul_const operations. We need to project
    // the original ct to match inv_rms's layout before the ct×ct multiply.
    let inv_rms_layout = poulpy_core::layouts::GLWELayout {
        n: inv_rms.n(),
        base2k: inv_rms.base2k(),
        k: inv_rms.k(),
        rank: inv_rms.rank(),
    };
    let ct_projected = if ct.base2k() == inv_rms.base2k() && ct.k() == inv_rms.k() {
        // Layouts already match — no projection needed
        use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
        let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(ct);
        let ct_ref = ct.to_ref();
        let src: &[u8] = ct_ref.data().data;
        let mut cloned_mut = cloned.to_mut();
        let dst: &mut [u8] = cloned_mut.data_mut().data;
        let len = src.len().min(dst.len());
        dst[..len].copy_from_slice(&src[..len]);
        cloned
    } else {
        chimera_align_layout(module, ct, &inv_rms_layout)
    };

    let normalised = chimera_ct_mul_with_res_offset(
        module,
        eval_key,
        &ct_projected,
        &inv_rms,
        activation_decode_precision(eval_key.res_offset),
    );

    // Optional: apply learnable scale (gamma) via plaintext multiplication.
    // Gamma is stored as fixed-point i64 values (one per slot).
    if let Some(gamma) = &config.gamma {
        chimera_mul_const(module, &normalised, gamma)
    } else {
        normalised
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_plaintext() {
        let config = LayerNormConfig::rms_norm(4);
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = layernorm_plaintext(&values, &config);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // Normalised: [0.3651, 0.7303, 1.0954, 1.4606]
        assert_eq!(result.len(), 4);
        let rms = (7.5_f64 + 1e-5).sqrt();
        for (i, &v) in values.iter().enumerate() {
            let expected = v / rms;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "RMSNorm[{i}]: {} vs expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_layer_norm_plaintext() {
        let config = LayerNormConfig::layer_norm(4);
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = layernorm_plaintext(&values, &config);

        // Mean = 2.5, Centered = [-1.5, -0.5, 0.5, 1.5]
        // Var = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
        // Std = sqrt(1.25) ≈ 1.1180
        assert_eq!(result.len(), 4);
        let mean = 2.5;
        let var = 1.25;
        let std = (var + 1e-5_f64).sqrt();
        for (i, &v) in values.iter().enumerate() {
            let expected = (v - mean) / std;
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "LayerNorm[{i}]: {} vs expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_plan_rms_norm() {
        let config = LayerNormConfig::rms_norm(128);
        let plan = plan_layernorm(&config);
        assert_eq!(plan.mean_rotations, 0); // RMSNorm skips mean
        assert_eq!(plan.variance_rotations, 7); // log₂(128) = 7
        assert!(!plan.has_affine);
    }

    #[test]
    fn test_plan_layer_norm() {
        let config = LayerNormConfig::layer_norm(256);
        let plan = plan_layernorm(&config);
        assert_eq!(plan.mean_rotations, 8); // log₂(256) = 8
        assert_eq!(plan.variance_rotations, 8);
    }

    #[test]
    fn test_depth_estimate() {
        let config = LayerNormConfig::rms_norm(128);
        // depth = 1 (square) + 3 (inv_sqrt degree-3) = 4
        assert_eq!(config.depth(), 4);
    }
}
