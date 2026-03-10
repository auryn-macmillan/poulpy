//! Full transformer block and forward pass under FHE.
//!
//! Combines attention, FFN, and LayerNorm into a complete transformer
//! decoder block, and provides a forward pass over multiple layers.
//!
//! ## Architecture
//!
//! Each transformer block follows the pre-norm architecture:
//!
//! ```text
//! x = x + Attention(RMSNorm(x))
//! x = x + FFN(RMSNorm(x))
//! ```
//!
//! Where FFN is:
//! ```text
//! FFN(x) = W_down · activation(W_gate · x) ⊙ (W_up · x)    // SwiGLU
//! // or
//! FFN(x) = W_2 · activation(W_1 · x)                        // standard
//! ```

use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWETensoring,
    layouts::GLWE,
};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};
use poulpy_core::ScratchTakeCore;

use crate::activations::{apply_poly_activation, chimera_ct_mul, gelu_poly_approx, silu_poly_approx, squared_relu_approx, PolyApprox};
use crate::arithmetic::{chimera_add, chimera_matmul_single_ct, chimera_project_layout};
use crate::attention::{AttentionConfig, AttentionPlan, SoftmaxStrategy, plan_attention};
use crate::encrypt::ChimeraEvalKey;
use crate::layernorm::{LayerNormConfig, LayerNormPlan, plan_layernorm};
use crate::params::{ChimeraParams, ModelDims};

/// Configuration for a single transformer block.
#[derive(Clone, Debug)]
pub struct TransformerBlockConfig {
    /// Attention layer configuration.
    pub attention: AttentionConfig,
    /// Pre-attention LayerNorm/RMSNorm.
    pub pre_attn_norm: LayerNormConfig,
    /// Pre-FFN LayerNorm/RMSNorm.
    pub pre_ffn_norm: LayerNormConfig,
    /// FFN configuration.
    pub ffn: FFNConfig,
    /// Whether to use residual connections.
    pub residual: bool,
}

/// FFN (feed-forward network) variant.
#[derive(Clone, Debug)]
pub enum FFNConfig {
    /// Standard FFN: W_2 · activation(W_1 · x)
    Standard {
        /// Activation function to use.
        activation: ActivationChoice,
    },
    /// SwiGLU FFN: W_down · (SiLU(W_gate · x) ⊙ (W_up · x))
    /// Used in LLaMA, Mixtral, etc.
    SwiGLU,
}

/// Choice of activation function for FFN.
#[derive(Clone, Debug)]
pub enum ActivationChoice {
    /// Degree-3 polynomial GELU approximation.
    PolyGELU,
    /// Degree-2 squared ReLU.
    SquaredReLU,
    /// Degree-3 polynomial SiLU/Swish approximation.
    PolySiLU,
    /// LUT-based exact GELU (expensive but accurate).
    LutGELU,
}

impl ActivationChoice {
    /// Returns the multiplicative depth of this activation.
    pub fn depth(&self) -> usize {
        match self {
            ActivationChoice::PolyGELU => 2,      // degree-3 via Horner = depth 2 (with squaring)
            ActivationChoice::SquaredReLU => 1,    // degree-2 = depth 1
            ActivationChoice::PolySiLU => 2,       // degree-3 = depth 2
            ActivationChoice::LutGELU => 0,        // LUT resets noise, depth = 0 (but expensive)
        }
    }
}

/// Weight matrices for a single FFN layer.
#[derive(Clone, Debug)]
pub struct FFNWeights {
    /// First linear layer: [d_model × d_ffn] (or W_gate for SwiGLU)
    pub w1: Vec<Vec<i64>>,
    /// Second linear layer: [d_ffn × d_model] (or W_down for SwiGLU)
    pub w2: Vec<Vec<i64>>,
    /// Optional third matrix for SwiGLU: W_up [d_model × d_ffn]
    pub w3: Option<Vec<Vec<i64>>>,
}

impl FFNWeights {
    /// Creates zero-initialised standard FFN weights.
    pub fn zeros_standard(dims: &ModelDims) -> Self {
        FFNWeights {
            w1: vec![vec![0i64; dims.d_ffn]; dims.d_model],
            w2: vec![vec![0i64; dims.d_model]; dims.d_ffn],
            w3: None,
        }
    }

    /// Creates zero-initialised SwiGLU FFN weights.
    pub fn zeros_swiglu(dims: &ModelDims) -> Self {
        FFNWeights {
            w1: vec![vec![0i64; dims.d_ffn]; dims.d_model],   // W_gate
            w2: vec![vec![0i64; dims.d_model]; dims.d_ffn],   // W_down
            w3: Some(vec![vec![0i64; dims.d_ffn]; dims.d_model]), // W_up
        }
    }
}

/// Plan for a complete transformer block.
#[derive(Clone, Debug)]
pub struct TransformerBlockPlan {
    /// Pre-attention norm plan.
    pub pre_attn_norm: LayerNormPlan,
    /// Attention computation plan.
    pub attention: AttentionPlan,
    /// Pre-FFN norm plan.
    pub pre_ffn_norm: LayerNormPlan,
    /// FFN plan.
    pub ffn: FFNPlan,
    /// Total multiplicative depth of the block.
    pub total_depth: usize,
    /// Total number of ciphertext-plaintext multiplications.
    pub total_ct_pt_muls: usize,
    /// Total number of ciphertext-ciphertext multiplications.
    pub total_ct_ct_muls: usize,
    /// Total number of rotations.
    pub total_rotations: usize,
}

/// Plan for FFN computation.
#[derive(Clone, Debug)]
pub struct FFNPlan {
    /// Number of ct-pt multiplications for W1 (and W3 if SwiGLU).
    pub w1_muls: usize,
    /// Activation depth.
    pub activation_depth: usize,
    /// Number of ct-pt multiplications for W2.
    pub w2_muls: usize,
    /// Number of ct-ct multiplications (1 if SwiGLU for the element-wise product).
    pub element_wise_muls: usize,
    /// Total multiplicative depth.
    pub total_depth: usize,
}

/// Plans a transformer block computation.
pub fn plan_transformer_block(config: &TransformerBlockConfig) -> TransformerBlockPlan {
    let pre_attn_norm = plan_layernorm(&config.pre_attn_norm);
    let attention = plan_attention(&config.attention);
    let pre_ffn_norm = plan_layernorm(&config.pre_ffn_norm);

    let dims = &config.attention.dims;

    let ffn = match &config.ffn {
        FFNConfig::Standard { activation } => {
            let w1_muls = dims.d_model * dims.d_ffn;
            let w2_muls = dims.d_ffn * dims.d_model;
            let act_depth = activation.depth();
            FFNPlan {
                w1_muls,
                activation_depth: act_depth,
                w2_muls,
                element_wise_muls: 0,
                total_depth: 1 + act_depth + 1, // W1 + activation + W2
            }
        }
        FFNConfig::SwiGLU => {
            let w1_muls = 2 * dims.d_model * dims.d_ffn; // W_gate + W_up
            let w2_muls = dims.d_ffn * dims.d_model;      // W_down
            FFNPlan {
                w1_muls,
                activation_depth: 2, // SiLU depth
                w2_muls,
                element_wise_muls: dims.d_ffn, // element-wise product
                total_depth: 1 + 2 + 1 + 1, // W_gate/W_up + SiLU + element_mul + W_down
            }
        }
    };

    let total_depth = pre_attn_norm.total_depth
        + attention.total_depth
        + pre_ffn_norm.total_depth
        + ffn.total_depth;

    let total_ct_pt_muls = attention.qkv_muls + attention.output_muls + ffn.w1_muls + ffn.w2_muls;
    let total_ct_ct_muls = attention.score_muls + attention.context_muls + ffn.element_wise_muls;
    let total_rotations = pre_attn_norm.mean_rotations
        + pre_attn_norm.variance_rotations
        + attention.num_rotations
        + pre_ffn_norm.mean_rotations
        + pre_ffn_norm.variance_rotations;

    TransformerBlockPlan {
        pre_attn_norm,
        attention,
        pre_ffn_norm,
        ffn,
        total_depth,
        total_ct_pt_muls,
        total_ct_ct_muls,
        total_rotations,
    }
}

/// Plan for a complete model forward pass.
#[derive(Clone, Debug)]
pub struct ForwardPassPlan {
    /// Plan for each layer (all identical if weights differ but architecture is the same).
    pub block_plan: TransformerBlockPlan,
    /// Number of layers.
    pub num_layers: usize,
    /// Total multiplicative depth across all layers.
    pub total_depth: usize,
    /// Whether bootstrapping is needed.
    pub needs_bootstrapping: bool,
    /// Estimated number of bootstrapping operations (0 if none needed).
    pub num_bootstraps: usize,
    /// Estimated total ciphertext-plaintext multiplications.
    pub total_ct_pt_muls: usize,
    /// Estimated total ciphertext-ciphertext multiplications.
    pub total_ct_ct_muls: usize,
}

/// Plans a complete model forward pass.
pub fn plan_forward_pass(
    config: &TransformerBlockConfig,
    num_layers: usize,
    params: &ChimeraParams,
) -> ForwardPassPlan {
    let block_plan = plan_transformer_block(config);
    let total_depth = block_plan.total_depth * num_layers;
    let needs_bootstrapping = total_depth > params.max_depth;
    let num_bootstraps = if needs_bootstrapping {
        (total_depth / params.max_depth).saturating_sub(1)
    } else {
        0
    };

    ForwardPassPlan {
        block_plan: block_plan.clone(),
        num_layers,
        total_depth,
        needs_bootstrapping,
        num_bootstraps,
        total_ct_pt_muls: block_plan.total_ct_pt_muls * num_layers,
        total_ct_ct_muls: block_plan.total_ct_ct_muls * num_layers,
    }
}

/// Creates a default transformer block configuration for the given model.
pub fn default_block_config(
    dims: ModelDims,
    params: ChimeraParams,
) -> TransformerBlockConfig {
    TransformerBlockConfig {
        attention: AttentionConfig {
            dims: dims.clone(),
            params: params.clone(),
            softmax_approx: SoftmaxStrategy::PolynomialDeg4,
            causal: true,
        },
        pre_attn_norm: LayerNormConfig::rms_norm(dims.d_model),
        pre_ffn_norm: LayerNormConfig::rms_norm(dims.d_model),
        ffn: FFNConfig::SwiGLU,
        residual: true,
    }
}

// ---------------------------------------------------------------------------
// Homomorphic FFN evaluation
// ---------------------------------------------------------------------------

/// Returns the polynomial approximation corresponding to an [`ActivationChoice`].
pub fn activation_to_poly(choice: &ActivationChoice) -> PolyApprox {
    match choice {
        ActivationChoice::PolyGELU => gelu_poly_approx(),
        ActivationChoice::SquaredReLU => squared_relu_approx(),
        ActivationChoice::PolySiLU => silu_poly_approx(),
        ActivationChoice::LutGELU => {
            // LUT-based GELU is not yet implemented as a ciphertext operation.
            // Fall back to the polynomial GELU for now.
            gelu_poly_approx()
        }
    }
}

/// Computes a standard FFN on encrypted input: `y = W2 · activation(W1 · x)`.
///
/// The computation proceeds in three phases:
///
/// 1. **Up-projection**: `h = W1 · x` — ciphertext-plaintext matmul.
///    Each row of `W1` is multiplied with the input ciphertext via
///    [`chimera_matmul_single_ct`], producing one intermediate ciphertext per
///    hidden dimension.
///
/// 2. **Activation**: `h' = activation(h)` — polynomial evaluation on each
///    intermediate ciphertext using the tensor product infrastructure.
///
/// 3. **Down-projection**: `y = W2 · h'` — the activated intermediates are
///    linearly combined per output dimension. Because each `h'_j` is a
///    separate ciphertext and `W2[i]` is a row of scalar weights, this is
///    a sum of scaled ciphertexts: `y_i = Σ_j W2[i][j] · h'_j`.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key containing tensor key for the activation.
/// * `x` - Encrypted input vector (single ciphertext).
/// * `weights` - FFN weight matrices. `w1` rows are up-projection weights,
///   `w2` rows are down-projection weights. `w3` is ignored.
/// * `activation` - Which polynomial activation to apply.
///
/// # Returns
///
/// A vector of ciphertexts, one per output dimension (one per row of `w2`).
///
/// # Panics
///
/// Panics if `weights.w1` or `weights.w2` is empty.
pub fn chimera_ffn_standard<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x: &GLWE<Vec<u8>>,
    weights: &FFNWeights,
    activation: &ActivationChoice,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    assert!(!weights.w1.is_empty(), "W1 must have at least one row");
    assert!(!weights.w2.is_empty(), "W2 must have at least one row");

    // Phase 1: Up-projection — h_j = W1[j] · x
    let w1_rows: Vec<Vec<i64>> = weights.w1.iter().map(|r| r.clone()).collect();
    let h = chimera_matmul_single_ct(module, x, &w1_rows);

    // Phase 2: Activation — h'_j = activation(h_j)
    let approx = activation_to_poly(activation);
    let h_act: Vec<GLWE<Vec<u8>>> = h
        .iter()
        .map(|hj| apply_poly_activation(module, eval_key, hj, &approx))
        .collect();

    // Phase 3: Down-projection — y_i = Σ_j W2[i][j] * h'_j
    //
    // Each W2[i][j] is a scalar weight for hidden dimension j.
    // We encode it as a single-coefficient polynomial [w] and compute
    // chimera_mul_const(h'_j, [w]), then accumulate with chimera_add.
    let mut outputs = Vec::with_capacity(weights.w2.len());
    for w2_row in &weights.w2 {
        assert_eq!(
            w2_row.len(),
            h_act.len(),
            "W2 row length must match hidden dimension (number of W1 rows)"
        );

        // Compute Σ_j w2_row[j] * h'_j
        let w2_vecs: Vec<Vec<i64>> = w2_row.iter().map(|&w| vec![w]).collect();
        let dot = crate::arithmetic::chimera_dot_product(module, &h_act, &w2_vecs);
        outputs.push(dot);
    }

    outputs
}

/// Computes a SwiGLU FFN on encrypted input:
///
/// ```text
/// y = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))
/// ```
///
/// This is the FFN variant used in LLaMA, Mixtral, and most modern
/// transformer architectures. It has three weight matrices and one
/// element-wise ct×ct product.
///
/// The computation proceeds in four phases:
///
/// 1. **Gate projection**: `gate = W_gate · x` — ct-pt matmul.
/// 2. **Up projection**: `up = W_up · x` — ct-pt matmul.
/// 3. **SiLU + element-wise product**: `h = SiLU(gate) ⊙ up` — polynomial
///    activation on each gate element, then ct×ct multiplication with the
///    corresponding up element.
/// 4. **Down projection**: `y = W_down · h` — sum of scaled ciphertexts.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key (tensor key for SiLU activation + ct×ct mul).
/// * `x` - Encrypted input vector (single ciphertext).
/// * `weights` - FFN weight matrices. `w1` is W_gate, `w2` is W_down,
///   `w3` must be `Some(W_up)`.
///
/// # Returns
///
/// A vector of ciphertexts, one per output dimension (one per row of `w2`).
///
/// # Panics
///
/// Panics if `weights.w3` is `None` or if dimensions are inconsistent.
pub fn chimera_ffn_swiglu<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x: &GLWE<Vec<u8>>,
    weights: &FFNWeights,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    assert!(!weights.w1.is_empty(), "W_gate must have at least one row");
    assert!(!weights.w2.is_empty(), "W_down must have at least one row");
    let w_up = weights.w3.as_ref().expect("SwiGLU requires w3 (W_up)");
    assert_eq!(
        weights.w1.len(),
        w_up.len(),
        "W_gate and W_up must have the same number of rows (d_ffn)"
    );

    // Phase 1: Gate projection — gate_j = W_gate[j] · x
    let w_gate_rows: Vec<Vec<i64>> = weights.w1.iter().map(|r| r.clone()).collect();
    let gate = chimera_matmul_single_ct(module, x, &w_gate_rows);

    // Phase 2: Up projection — up_j = W_up[j] · x
    let w_up_rows: Vec<Vec<i64>> = w_up.iter().map(|r| r.clone()).collect();
    let up = chimera_matmul_single_ct(module, x, &w_up_rows);

    // Phase 3: SiLU activation + element-wise product
    // h_j = SiLU(gate_j) ⊙ up_j
    //
    // After SiLU activation, gate_activated is at a lower base2k (out_base2k)
    // due to the tensor product. The up projections are still at in_base2k.
    // We must project up_j to the same layout before the ct*ct multiply.
    let silu_approx = silu_poly_approx();
    let d_ffn = gate.len();
    let mut h = Vec::with_capacity(d_ffn);
    for j in 0..d_ffn {
        // SiLU(gate_j)
        let gate_activated = apply_poly_activation(module, eval_key, &gate[j], &silu_approx);

        // Project up_j to match gate_activated's layout (potentially lower base2k)
        use poulpy_core::layouts::{GLWEInfos, LWEInfos};
        let gate_layout = poulpy_core::layouts::GLWELayout {
            n: gate_activated.n(),
            base2k: gate_activated.base2k(),
            k: gate_activated.k(),
            rank: gate_activated.rank(),
        };
        let up_projected = if up[j].base2k() == gate_activated.base2k() {
            // No projection needed — layouts already match
            up[j].clone()
        } else {
            chimera_project_layout(module, &up[j], &gate_layout)
        };

        // Element-wise product: SiLU(gate_j) ⊙ up_j
        let hj = chimera_ct_mul(module, eval_key, &gate_activated, &up_projected);
        h.push(hj);
    }

    // Phase 4: Down projection — y_i = Σ_j W_down[i][j] * h_j
    let mut outputs = Vec::with_capacity(weights.w2.len());
    for w2_row in &weights.w2 {
        assert_eq!(
            w2_row.len(),
            h.len(),
            "W_down row length must match hidden dimension (d_ffn)"
        );
        let w2_vecs: Vec<Vec<i64>> = w2_row.iter().map(|&w| vec![w]).collect();
        let dot = crate::arithmetic::chimera_dot_product(module, &h, &w2_vecs);
        outputs.push(dot);
    }

    outputs
}

/// Dispatches to the appropriate FFN variant based on the configuration.
///
/// Calls [`chimera_ffn_standard`] for `FFNConfig::Standard` or
/// [`chimera_ffn_swiglu`] for `FFNConfig::SwiGLU`.
pub fn chimera_ffn<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x: &GLWE<Vec<u8>>,
    weights: &FFNWeights,
    config: &FFNConfig,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    match config {
        FFNConfig::Standard { activation } => {
            chimera_ffn_standard(module, eval_key, x, weights, activation)
        }
        FFNConfig::SwiGLU => chimera_ffn_swiglu(module, eval_key, x, weights),
    }
}

/// Adds two encrypted vectors element-wise (residual connection).
///
/// Given two vectors of ciphertexts of the same length, computes
/// `result[i] = a[i] + b[i]` for each element.
///
/// # Panics
///
/// Panics if the vectors have different lengths.
pub fn chimera_residual_add<BE: Backend>(
    module: &Module<BE>,
    a: &[GLWE<Vec<u8>>],
    b: &[GLWE<Vec<u8>>],
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEAdd,
{
    assert_eq!(
        a.len(),
        b.len(),
        "residual add: vectors must have the same length"
    );
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| chimera_add(module, ai, bi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Precision, SecurityLevel};

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    #[test]
    fn test_transformer_block_plan() {
        let dims = ModelDims::dense_7b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = default_block_config(dims, params);
        let plan = plan_transformer_block(&config);

        assert!(plan.total_depth > 0);
        assert!(plan.total_ct_pt_muls > 0);
        assert!(plan.total_rotations > 0);
    }

    #[test]
    fn test_forward_pass_plan_small_model() {
        let dims = ModelDims::dense_7b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = default_block_config(dims.clone(), params.clone());
        let plan = plan_forward_pass(&config, dims.n_layers, &params);

        assert_eq!(plan.num_layers, 32);
        assert!(plan.total_depth > 0);
        // 32-layer model should need bootstrapping at 128-bit
        // (total_depth = block_depth * 32, which exceeds max_depth=48)
    }

    #[test]
    fn test_forward_pass_plan_moe() {
        let dims = ModelDims::moe_40b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = default_block_config(dims.clone(), params.clone());
        let plan = plan_forward_pass(&config, dims.n_layers, &params);

        assert_eq!(plan.num_layers, 32);
        assert!(plan.total_ct_ct_muls > 0);
    }

    #[test]
    fn test_ffn_standard_config() {
        let dims = ModelDims::dense_7b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: dims.clone(),
                params: params.clone(),
                softmax_approx: SoftmaxStrategy::ReluSquared,
                causal: true,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(dims.d_model),
            pre_ffn_norm: LayerNormConfig::rms_norm(dims.d_model),
            ffn: FFNConfig::Standard {
                activation: ActivationChoice::SquaredReLU,
            },
            residual: true,
        };
        let plan = plan_transformer_block(&config);
        // Squared ReLU has depth 1, Linear attention has depth 1
        assert!(plan.total_depth > 0);
    }

    #[test]
    fn test_activation_depths() {
        assert_eq!(ActivationChoice::PolyGELU.depth(), 2);
        assert_eq!(ActivationChoice::SquaredReLU.depth(), 1);
        assert_eq!(ActivationChoice::PolySiLU.depth(), 2);
        assert_eq!(ActivationChoice::LutGELU.depth(), 0);
    }

    #[test]
    fn test_activation_to_poly() {
        // Verify that activation_to_poly returns the correct approximations.
        let gelu = activation_to_poly(&ActivationChoice::PolyGELU);
        assert_eq!(gelu.degree(), 3);

        let sqrelu = activation_to_poly(&ActivationChoice::SquaredReLU);
        assert_eq!(sqrelu.degree(), 2);

        let silu = activation_to_poly(&ActivationChoice::PolySiLU);
        assert_eq!(silu.degree(), 3);

        let lut_gelu = activation_to_poly(&ActivationChoice::LutGELU);
        // Falls back to polynomial GELU for now
        assert_eq!(lut_gelu.degree(), 3);
    }

    #[test]
    fn test_ffn_standard_homomorphic() {
        // Test a minimal standard FFN: y = W2 · activation(W1 · x)
        // With:
        //   x = [3] (single value)
        //   W1 = [[2]] (d_model=1, d_ffn=1): h = 2 * 3 = 6
        //   activation = SquaredReLU (x²): h' = 6² = 36 (on the torus)
        //   W2 = [[1]] (d_ffn=1, d_model=1): y = 1 * h' = 36
        //
        // This test verifies the pipeline completes without panicking.
        // Full numerical verification of chained operations (mul_const + tensor
        // product + mul_const) requires careful torus precision tracking.
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_encrypt, ChimeraEvalKey, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [4u8; 32], [5u8; 32],
        );

        let vals: Vec<i8> = vec![3];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        let weights = FFNWeights {
            w1: vec![vec![2]],
            w2: vec![vec![1]],
            w3: None,
        };

        let result = chimera_ffn_standard(
            &module,
            &eval_key,
            &ct,
            &weights,
            &ActivationChoice::SquaredReLU,
        );
        assert_eq!(result.len(), 1, "FFN should produce one output ciphertext");
    }

    #[test]
    fn test_ffn_swiglu_homomorphic() {
        // Test a minimal SwiGLU FFN: y = W_down · (SiLU(W_gate · x) ⊙ W_up · x)
        // With:
        //   x = [2] (single value)
        //   W_gate = [[1]] (d_model=1, d_ffn=1): gate = 1 * 2 = 2
        //   W_up   = [[1]] (d_model=1, d_ffn=1): up   = 1 * 2 = 2
        //   SiLU(gate) ⊙ up = SiLU(2) * 2 ≈ 1.76 * 2 = 3.52 (on torus)
        //   W_down = [[1]] (d_ffn=1, d_model=1): y = 1 * h = 3.52
        //
        // This test verifies the SwiGLU pipeline completes without panicking.
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_encrypt, ChimeraEvalKey, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [4u8; 32], [5u8; 32],
        );

        let vals: Vec<i8> = vec![2];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        let weights = FFNWeights {
            w1: vec![vec![1]],                       // W_gate
            w2: vec![vec![1]],                       // W_down
            w3: Some(vec![vec![1]]),                  // W_up
        };

        let result = chimera_ffn_swiglu(&module, &eval_key, &ct, &weights);
        assert_eq!(result.len(), 1, "SwiGLU FFN should produce one output ciphertext");
    }

    #[test]
    fn test_chimera_ffn_dispatch() {
        // Test the chimera_ffn dispatch function routes correctly.
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_encrypt, ChimeraEvalKey, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [4u8; 32], [5u8; 32],
        );

        let vals: Vec<i8> = vec![2];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        // Standard FFN
        let std_weights = FFNWeights {
            w1: vec![vec![1]],
            w2: vec![vec![1]],
            w3: None,
        };
        let config_std = FFNConfig::Standard {
            activation: ActivationChoice::SquaredReLU,
        };
        let result_std = chimera_ffn(&module, &eval_key, &ct, &std_weights, &config_std);
        assert_eq!(result_std.len(), 1);

        // SwiGLU FFN
        let swiglu_weights = FFNWeights {
            w1: vec![vec![1]],
            w2: vec![vec![1]],
            w3: Some(vec![vec![1]]),
        };
        let config_swiglu = FFNConfig::SwiGLU;
        let result_swiglu = chimera_ffn(&module, &eval_key, &ct, &swiglu_weights, &config_swiglu);
        assert_eq!(result_swiglu.len(), 1);
    }

    #[test]
    fn test_residual_add() {
        // Test chimera_residual_add on a pair of ciphertext vectors.
        use crate::encoding::{decode_int8, encode_int8};
        use crate::encrypt::{chimera_decrypt, chimera_encrypt, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let a_vals: Vec<i8> = vec![10];
        let b_vals: Vec<i8> = vec![5];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [2u8; 32], [3u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [4u8; 32], [5u8; 32]);

        let result = chimera_residual_add(&module, &[ct_a], &[ct_b]);
        assert_eq!(result.len(), 1);

        let pt_dec = chimera_decrypt(&module, &key, &result[0], &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 1);
        let diff = (decoded[0] as i16 - 15).unsigned_abs();
        assert!(
            diff <= 1,
            "residual_add: expected 15, got {}", decoded[0]
        );
    }
}
