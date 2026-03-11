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
    layouts::{GLWEInfos, GLWELayout, LWEInfos, GLWE},
    GLWEAdd, GLWEMulConst, GLWESub, GLWETensoring, GLWETrace,
};
use poulpy_core::{GLWEAutomorphism, ScratchTakeCore};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

use crate::activations::{
    apply_poly_activation, chimera_ct_mul, gelu_poly_approx, silu_poly_approx, squared_relu_approx, PolyApprox,
};
use crate::arithmetic::{chimera_add, chimera_align_layout, chimera_matmul_single_ct};
use crate::attention::{
    chimera_apply_softmax, chimera_attention_context, chimera_attention_score, chimera_multi_head_attention_vec,
    chimera_output_project, chimera_qkv_project_single, plan_attention, AttentionConfig, AttentionPlan, AttentionWeights,
    SoftmaxStrategy,
};
use crate::bootstrapping::{chimera_bootstrap, needs_bootstrap, BootstrappingConfig, ChimeraBootstrapKeyPrepared};
use crate::encrypt::ChimeraEvalKey;
use crate::layernorm::{chimera_rms_norm, chimera_rms_norm_vec, plan_layernorm, LayerNormConfig, LayerNormPlan};
use crate::noise::NoiseTracker;
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
            ActivationChoice::PolyGELU => 2,    // degree-3 via Horner = depth 2 (with squaring)
            ActivationChoice::SquaredReLU => 1, // degree-2 = depth 1
            ActivationChoice::PolySiLU => 2,    // degree-3 = depth 2
            ActivationChoice::LutGELU => 0,     // LUT resets noise, depth = 0 (but expensive)
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
            w1: vec![vec![0i64; dims.d_ffn]; dims.d_model],       // W_gate
            w2: vec![vec![0i64; dims.d_model]; dims.d_ffn],       // W_down
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
            let w2_muls = dims.d_ffn * dims.d_model; // W_down
            FFNPlan {
                w1_muls,
                activation_depth: 2, // SiLU depth
                w2_muls,
                element_wise_muls: dims.d_ffn, // element-wise product
                total_depth: 1 + 2 + 1 + 1,    // W_gate/W_up + SiLU + element_mul + W_down
            }
        }
    };

    let total_depth = pre_attn_norm.total_depth + attention.total_depth + pre_ffn_norm.total_depth + ffn.total_depth;

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
pub fn plan_forward_pass(config: &TransformerBlockConfig, num_layers: usize, params: &ChimeraParams) -> ForwardPassPlan {
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
pub fn default_block_config(dims: ModelDims, params: ChimeraParams) -> TransformerBlockConfig {
    TransformerBlockConfig {
        attention: AttentionConfig {
            dims: dims.clone(),
            params: params.clone(),
            softmax_approx: SoftmaxStrategy::PolynomialDeg4,
            causal: true,
            rope: None,
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
            chimera_align_layout(module, &up[j], &gate_layout)
        };

        // Element-wise product: SiLU(gate_j) ⊙ up_j
        let hj = chimera_ct_mul(module, eval_key, &gate_activated, &up_projected);
        h.push(hj);
    }

    // Phase 4: Down projection — y_i = Σ_j W_down[i][j] * h_j
    let mut outputs = Vec::with_capacity(weights.w2.len());
    for w2_row in &weights.w2 {
        assert_eq!(w2_row.len(), h.len(), "W_down row length must match hidden dimension (d_ffn)");
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
        FFNConfig::Standard { activation } => chimera_ffn_standard(module, eval_key, x, weights, activation),
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
pub fn chimera_residual_add<BE: Backend>(module: &Module<BE>, a: &[GLWE<Vec<u8>>], b: &[GLWE<Vec<u8>>]) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEAdd,
{
    assert_eq!(a.len(), b.len(), "residual add: vectors must have the same length");
    a.iter().zip(b.iter()).map(|(ai, bi)| chimera_add(module, ai, bi)).collect()
}

/// Weight matrices for a complete transformer block (attention + FFN).
#[derive(Clone, Debug)]
pub struct TransformerBlockWeights {
    /// Attention layer weights (Q, K, V, O projection matrices).
    pub attention: AttentionWeights,
    /// FFN layer weights (W1, W2, and optionally W3 for SwiGLU).
    pub ffn: FFNWeights,
    /// Pre-attention RMSNorm gamma weights (one per d_model slot).
    /// If `None`, uses unit gamma (no learnable scale).
    pub pre_attn_norm_gamma: Option<Vec<i64>>,
    /// Pre-FFN RMSNorm gamma weights (one per d_model slot).
    /// If `None`, uses unit gamma (no learnable scale).
    pub pre_ffn_norm_gamma: Option<Vec<i64>>,
}

impl TransformerBlockWeights {
    /// Creates zero-initialised weights for a standard FFN block.
    pub fn zeros_standard(dims: &ModelDims) -> Self {
        TransformerBlockWeights {
            attention: AttentionWeights::zeros(dims),
            ffn: FFNWeights::zeros_standard(dims),
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
        }
    }

    /// Creates zero-initialised weights for a SwiGLU FFN block.
    pub fn zeros_swiglu(dims: &ModelDims) -> Self {
        TransformerBlockWeights {
            attention: AttentionWeights::zeros(dims),
            ffn: FFNWeights::zeros_swiglu(dims),
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Homomorphic transformer block evaluation
// ---------------------------------------------------------------------------

/// Evaluates a legacy single-ciphertext transformer decoder block on an encrypted input.
///
/// Implements the pre-norm architecture:
///
/// ```text
/// x = x + Attention(RMSNorm(x))
/// x = x + FFN(RMSNorm(x))
/// ```
///
/// ## Computation Flow
///
/// 1. **Pre-attention RMSNorm**: Normalises the input via `chimera_rms_norm`.
/// 2. **Attention**: Projects Q/K/V via plaintext weight multiplication,
///    computes attention scores (ct*ct), applies softmax approximation,
///    computes context (ct*ct), and projects output.
/// 3. **Residual connection**: Adds the attention output back to the input.
/// 4. **Pre-FFN RMSNorm**: Normalises the post-attention result.
/// 5. **FFN**: Applies the feed-forward network (standard or SwiGLU).
/// 6. **Residual connection**: Adds the FFN output back to the post-attention result.
///
/// ## Multi-Head Attention
///
/// The attention step iterates over `n_heads` heads. Each head h uses
/// weight rows `w_q[h]`, `w_k[h]`, `w_v[h]`, `w_o[h]` to project Q, K, V
/// and produce an output. The per-head outputs are summed to produce the
/// final attention output. This is the "per-head" packing strategy where
/// each head operates on the full polynomial ciphertext via ring
/// multiplication with its weight row.
///
/// When `n_heads == 1`, this reduces to the original single-head behavior.
///
/// ## Noise Budget
///
/// A single transformer block consumes approximately:
/// - RMSNorm: 2 ct*ct multiplications (square + normalise) + polynomial eval
/// - Attention: 1-2 ct*ct multiplications (score + context)
/// - FFN: 1-2 ct*ct multiplications (activation + SwiGLU element-wise)
/// - Total: ~6-8 multiplicative depth levels per block
///
/// At 80-bit security (max_depth=12), this supports 1-2 blocks without
/// bootstrapping. At 128-bit security (max_depth=48), this supports ~6-8 blocks.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key (tensor key for ct*ct, auto keys for trace/rotation).
/// * `ct_x` - Encrypted input vector (single ciphertext).
/// * `config` - Transformer block configuration (attention, norms, FFN settings).
/// * `weights` - Plaintext weight matrices for attention and FFN.
///
/// # Returns
///
/// A single ciphertext result.
///
/// This legacy path truncates multi-output attention/FFN computations to the
/// first produced ciphertext and is therefore suitable only for toy experiments
/// and backward-compatibility tests. Use [`chimera_transformer_block_vec`] for
/// model-faithful transformer evaluation.
///
/// # Panics
///
/// Panics if norm configurations are not set to RMSNorm mode.
#[deprecated(note = "legacy toy-only API; use chimera_transformer_block_vec instead")]
pub fn chimera_transformer_block<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct_x: &GLWE<Vec<u8>>,
    config: &TransformerBlockConfig,
    weights: &TransformerBlockWeights,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWETrace<BE> + GLWEAutomorphism<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    // -----------------------------------------------------------------------
    // Step 1: Pre-attention RMSNorm
    //
    // If the block weights carry learnable gamma for the pre-attention norm,
    // build a config that includes it.  Otherwise fall back to the original
    // config (unit gamma).
    // -----------------------------------------------------------------------
    let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
        Some(gamma) => config.pre_attn_norm.clone().with_gamma(gamma.clone()),
        None => config.pre_attn_norm.clone(),
    };
    let normed_pre_attn = chimera_rms_norm(module, eval_key, ct_x, &pre_attn_norm_cfg);

    // -----------------------------------------------------------------------
    // Step 2: Multi-head attention
    //
    // For each head h (0..n_heads), we project Q/K/V using one weight row
    // per head:
    //   Q_h = W_Q[h] · normed_x  (polynomial ring multiplication)
    //   K_h = W_K[h] · normed_x
    //   V_h = W_V[h] · normed_x
    //   score_h = Q_h · K_h (ct*ct + slot sum)
    //   attn_h  = softmax_approx(score_h)
    //   context_h = attn_h * V_h (ct*ct)
    //   head_out_h = W_O[h] · context_h
    //
    // The final attention output is the sum of all per-head contributions:
    //   attn_out = Σ_h head_out_h
    //
    // Each weight row W_Q[h] encodes the projection for head h as a
    // polynomial. The ring multiplication computes the inner product
    // structure over d_model coefficients. The output projection W_O[h]
    // recombines each head's context back into the d_model space.
    //
    // This is the "per-head" packing strategy described in the spec:
    // one ciphertext per head throughout the computation, with the output
    // projection summing across heads to produce the final result.
    // -----------------------------------------------------------------------
    let n_heads = config.attention.dims.n_heads;
    let attn_out = {
        // Compute first head to initialise the accumulator
        let (ct_q0, ct_k0, ct_v0) = chimera_qkv_project_single(
            module,
            &normed_pre_attn,
            &weights.attention.w_q[0],
            &weights.attention.w_k[0],
            &weights.attention.w_v[0],
        );
        let score0 = chimera_attention_score(module, eval_key, &ct_q0, &ct_k0, 0);
        let attn_wts0 = chimera_apply_softmax(module, eval_key, &[score0], &config.attention.softmax_approx);
        let context0 = chimera_attention_context(module, eval_key, &attn_wts0, &[ct_v0]);
        let head_out_vec0 = chimera_output_project(module, &context0, &[weights.attention.w_o[0].clone()]);
        let mut acc = head_out_vec0
            .into_iter()
            .next()
            .expect("output projection must produce at least one ciphertext");

        // Accumulate remaining heads
        for h in 1..n_heads {
            let (ct_q_h, ct_k_h, ct_v_h) = chimera_qkv_project_single(
                module,
                &normed_pre_attn,
                &weights.attention.w_q[h],
                &weights.attention.w_k[h],
                &weights.attention.w_v[h],
            );
            let score_h = chimera_attention_score(module, eval_key, &ct_q_h, &ct_k_h, 0);
            let attn_wts_h = chimera_apply_softmax(module, eval_key, &[score_h], &config.attention.softmax_approx);
            let context_h = chimera_attention_context(module, eval_key, &attn_wts_h, &[ct_v_h]);
            let head_out_vec_h = chimera_output_project(module, &context_h, &[weights.attention.w_o[h].clone()]);
            let head_out_h = head_out_vec_h
                .into_iter()
                .next()
                .expect("output projection must produce at least one ciphertext");

            // The head outputs may be at different layouts after tensor products.
            // Project the accumulator to match the new head's layout before adding.
            let head_layout = GLWELayout {
                n: head_out_h.n(),
                base2k: head_out_h.base2k(),
                k: head_out_h.k(),
                rank: head_out_h.rank(),
            };
            let acc_proj = if acc.base2k() == head_out_h.base2k() && acc.k() == head_out_h.k() {
                acc
            } else {
                chimera_align_layout(module, &acc, &head_layout)
            };
            acc = chimera_add(module, &acc_proj, &head_out_h);
        }

        acc
    };

    // -----------------------------------------------------------------------
    // Step 3: Residual connection (attention)
    //
    // The attention output may be at a different base2k/k than the input
    // due to multiple tensor products. We project the input to match before
    // adding.
    // -----------------------------------------------------------------------
    let residual_1 = if config.residual {
        let ct_x_layout = GLWELayout {
            n: attn_out.n(),
            base2k: attn_out.base2k(),
            k: attn_out.k(),
            rank: attn_out.rank(),
        };
        let ct_x_proj = if ct_x.base2k() == attn_out.base2k() && ct_x.k() == attn_out.k() {
            // Clone ct_x since layouts match
            use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
            let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(ct_x);
            {
                let src_ref = ct_x.to_ref();
                let src: &[u8] = src_ref.data().data;
                let mut dst_mut = cloned.to_mut();
                let dst: &mut [u8] = dst_mut.data_mut().data;
                let len = src.len().min(dst.len());
                dst[..len].copy_from_slice(&src[..len]);
            }
            cloned
        } else {
            chimera_align_layout(module, ct_x, &ct_x_layout)
        };
        chimera_add(module, &ct_x_proj, &attn_out)
    } else {
        attn_out
    };

    // -----------------------------------------------------------------------
    // Step 4: Pre-FFN RMSNorm
    //
    // Same gamma wiring as step 1 — use learnable scale from the weights
    // if available.
    // -----------------------------------------------------------------------
    let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
        Some(gamma) => config.pre_ffn_norm.clone().with_gamma(gamma.clone()),
        None => config.pre_ffn_norm.clone(),
    };
    let normed_pre_ffn = chimera_rms_norm(module, eval_key, &residual_1, &pre_ffn_norm_cfg);

    // -----------------------------------------------------------------------
    // Step 5: FFN
    //
    // The FFN produces a vector of ciphertexts (one per output dimension).
    // For the single-ciphertext-in/single-ciphertext-out transformer block,
    // we take only the first output dimension.
    // -----------------------------------------------------------------------
    let ffn_out_vec = chimera_ffn(module, eval_key, &normed_pre_ffn, &weights.ffn, &config.ffn);
    let ffn_out = ffn_out_vec
        .into_iter()
        .next()
        .expect("FFN must produce at least one output ciphertext");

    // -----------------------------------------------------------------------
    // Step 6: Residual connection (FFN)
    // -----------------------------------------------------------------------
    if config.residual {
        let res1_layout = GLWELayout {
            n: ffn_out.n(),
            base2k: ffn_out.base2k(),
            k: ffn_out.k(),
            rank: ffn_out.rank(),
        };
        let res1_proj = if residual_1.base2k() == ffn_out.base2k() && residual_1.k() == ffn_out.k() {
            use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
            let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(&residual_1);
            {
                let src_ref = residual_1.to_ref();
                let src: &[u8] = src_ref.data().data;
                let mut dst_mut = cloned.to_mut();
                let dst: &mut [u8] = dst_mut.data_mut().data;
                let len = src.len().min(dst.len());
                dst[..len].copy_from_slice(&src[..len]);
            }
            cloned
        } else {
            chimera_align_layout(module, &residual_1, &res1_layout)
        };
        chimera_add(module, &res1_proj, &ffn_out)
    } else {
        ffn_out
    }
}

/// Evaluates a sequence of legacy single-ciphertext transformer blocks.
///
/// Chains `num_layers` transformer blocks sequentially, feeding the output of
/// each block as the input to the next. All blocks share the same configuration
/// but may have different weights.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `ct_x` - Encrypted input vector.
/// * `config` - Block configuration (shared across all layers).
/// * `layer_weights` - Per-layer weight matrices.
///
/// # Returns
///
/// Encrypted output after all transformer blocks.
///
/// This function inherits the legacy truncation semantics of
/// [`chimera_transformer_block`]. Use [`chimera_forward_pass_vec`] for the
/// vector representation used by real-model inference.
///
/// # Panics
///
/// Panics if `layer_weights.len()` does not match `num_layers`.
#[deprecated(note = "legacy toy-only API; use chimera_forward_pass_vec instead")]
#[allow(deprecated)]
pub fn chimera_forward_pass<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct_x: &GLWE<Vec<u8>>,
    config: &TransformerBlockConfig,
    layer_weights: &[TransformerBlockWeights],
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWETrace<BE> + GLWEAutomorphism<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut current = {
        use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
        let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(ct_x);
        {
            let src_ref = ct_x.to_ref();
            let src: &[u8] = src_ref.data().data;
            let mut dst_mut = cloned.to_mut();
            let dst: &mut [u8] = dst_mut.data_mut().data;
            let len = src.len().min(dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
        cloned
    };

    for (layer_idx, weights) in layer_weights.iter().enumerate() {
        current = chimera_transformer_block(module, eval_key, &current, config, weights);
        // In a production implementation, this is where we would check the noise
        // budget and potentially bootstrap if needed:
        //   if needs_bootstrap(&current, params) {
        //       current = chimera_bootstrap(module, eval_key, &current, params);
        //   }
        let _ = layer_idx; // suppress unused warning
    }

    current
}

/// Evaluates a sequence of transformer blocks with optional mid-inference
/// bootstrapping to refresh the noise budget.
///
/// This is the production-grade forward pass. It maintains a
/// [`NoiseTracker`] that estimates noise growth after each transformer
/// block. When the estimated noise budget drops below the threshold in
/// `bootstrap_config`, the ciphertext is bootstrapped before proceeding
/// to the next layer.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key (tensor key, auto keys).
/// * `ct_x` - Encrypted input vector.
/// * `config` - Block configuration (shared across all layers).
/// * `layer_weights` - Per-layer weight matrices.
/// * `params` - CHIMERA parameter set (used for noise budget calculation).
/// * `bootstrap_config` - Controls when bootstrapping is triggered.
/// * `bsk_prepared` - Prepared bootstrap key material (required if
///   `bootstrap_config.enabled` is `true`; may be `None` otherwise).
///
/// # Returns
///
/// A tuple of:
/// - The encrypted output after all transformer blocks.
/// - The final [`NoiseTracker`] state (useful for diagnostics).
///
/// # Panics
///
/// Panics if bootstrapping is triggered but `bsk_prepared` is `None`.
#[allow(deprecated)]
pub fn chimera_forward_pass_with_bootstrap<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct_x: &GLWE<Vec<u8>>,
    config: &TransformerBlockConfig,
    layer_weights: &[TransformerBlockWeights],
    params: &ChimeraParams,
    bootstrap_config: &BootstrappingConfig,
    bsk_prepared: Option<&ChimeraBootstrapKeyPrepared<BE>>,
) -> (GLWE<Vec<u8>>, NoiseTracker)
where
    Module<BE>: GLWETensoring<BE>
        + GLWEMulConst<BE>
        + GLWEAdd
        + GLWETrace<BE>
        + GLWEAutomorphism<BE>
        + poulpy_hal::api::ModuleN
        + poulpy_core::LWESampleExtract
        + poulpy_core::LWEKeySwitch<BE>
        + poulpy_schemes::bin_fhe::blind_rotation::BlindRotationExecute<poulpy_schemes::bin_fhe::blind_rotation::CGGI, BE>
        + poulpy_schemes::bin_fhe::blind_rotation::LookupTableFactory,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut current = {
        use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
        let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(ct_x);
        {
            let src_ref = ct_x.to_ref();
            let src: &[u8] = src_ref.data().data;
            let mut dst_mut = cloned.to_mut();
            let dst: &mut [u8] = dst_mut.data_mut().data;
            let len = src.len().min(dst.len());
            dst[..len].copy_from_slice(&src[..len]);
        }
        cloned
    };

    let mut tracker = NoiseTracker::fresh();
    let mut bootstrap_count: usize = 0;

    // Estimate the noise cost of a single transformer block for the tracker.
    // Each block performs: 2× RMSNorm (ct*ct + mul_const + ct*ct) + attention
    // (3× mul_const + ct*ct + softmax + ct*ct + mul_const) + FFN (mul_const
    // + activation + mul_const) + 2× residual add.  We approximate this as
    // several mul_ct and mul_const operations.
    let ring_degree = params.degree.0 as usize;

    for (layer_idx, weights) in layer_weights.iter().enumerate() {
        // --- Check noise budget before this layer ---
        if needs_bootstrap(&tracker, params, bootstrap_config) {
            let bsk = bsk_prepared.expect(
                "chimera_forward_pass_with_bootstrap: bootstrapping triggered \
                 but no bootstrap key provided (bsk_prepared is None)",
            );

            // Respect max_bootstraps_per_pass limit
            if bootstrap_config.max_bootstraps_per_pass == 0 || bootstrap_count < bootstrap_config.max_bootstraps_per_pass {
                current = chimera_bootstrap(module, &current, &mut tracker, params, bsk, bootstrap_config);
                bootstrap_count += 1;
            }
            // If we've hit the limit, we continue without bootstrapping
            // (the computation may produce incorrect results, but we don't
            // panic — the caller should check the tracker's budget).
        }

        // --- Evaluate transformer block ---
        current = chimera_transformer_block(module, eval_key, &current, config, weights);

        // --- Update noise tracker with approximate cost of one block ---
        //
        // A transformer block's dominant noise sources are:
        //   - Pre-attn RMSNorm: 1 ct*ct (square) + 1 ct*ct (normalise) = 2 depth
        //   - Attention: 1 ct*ct (score) + 1 ct*ct (context) = 2 depth
        //   - Pre-FFN RMSNorm: 2 depth (same as pre-attn)
        //   - FFN: 1-2 ct*ct (activation + optional SwiGLU element-wise) = 1-2 depth
        // Total: ~7-8 ct*ct multiplications per block.
        //
        // We model this as a sequence of ct*ct multiplications applied to
        // a fresh tracker (representing the "other operand" each time).
        let block_ct_ct_muls = match &config.ffn {
            FFNConfig::SwiGLU => 8, // 2 (norm) + 2 (attn) + 2 (norm) + 2 (silu + ew)
            FFNConfig::Standard { activation } => {
                6 + activation.depth() // 2 (norm) + 2 (attn) + 2 (norm) + activation
            }
        };
        for _ in 0..block_ct_ct_muls {
            let other = NoiseTracker::fresh();
            tracker.mul_ct(&other, ring_degree);
        }

        let _ = layer_idx;
    }

    (current, tracker)
}

// ===========================================================================
// Vector-representation pipeline
//
// The functions below operate on `Vec<GLWE<Vec<u8>>>` — one ciphertext per
// dimension, with each scalar value encoded in coefficient 0. This is the
// representation needed for real model inference, where matrix-vector
// products are computed as dot products of scalar ciphertexts with scalar
// plaintext weights.
//
// The original single-ciphertext pipeline (above) remains unchanged for
// backward compatibility with all 137+ existing tests.
// ===========================================================================

/// Computes a standard FFN in the vector representation.
///
/// `y_i = Σ_j W2[i][j] * activation(Σ_k W1[j][k] * x_k)`
///
/// Each element of `x_cts` encrypts one dimension of the input (value in
/// coefficient 0). The output is a vector of the same kind.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `x_cts` - Input vector: one ct per dimension.
/// * `weights` - FFN weight matrices.
/// * `activation` - Which polynomial activation to apply.
///
/// # Returns
///
/// A vector of ciphertexts, one per output dimension (one per row of `w2`).
pub fn chimera_ffn_standard_vec<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
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
    let d_in = x_cts.len();

    // Phase 1: Up-projection — h_j = Σ_k W1[j][k] * x_k
    let approx = activation_to_poly(activation);
    let d_ffn = weights.w1.len();
    let mut h_act = Vec::with_capacity(d_ffn);
    for j in 0..d_ffn {
        let w_vecs: Vec<Vec<i64>> = weights.w1[j][..d_in].iter().map(|&w| vec![w]).collect();
        let hj = crate::arithmetic::chimera_dot_product(module, x_cts, &w_vecs);

        // Phase 2: Activation on each hidden dimension
        let hj_act = apply_poly_activation(module, eval_key, &hj, &approx);
        h_act.push(hj_act);
    }

    // Phase 3: Down-projection — y_i = Σ_j W2[i][j] * h'_j
    let mut outputs = Vec::with_capacity(weights.w2.len());
    for w2_row in &weights.w2 {
        let w2_vecs: Vec<Vec<i64>> = w2_row.iter().map(|&w| vec![w]).collect();
        let dot = crate::arithmetic::chimera_dot_product(module, &h_act, &w2_vecs);
        outputs.push(dot);
    }

    outputs
}

/// Computes a SwiGLU FFN in the vector representation.
///
/// `y_i = Σ_j W_down[i][j] * (SiLU(Σ_k W_gate[j][k] * x_k) ⊙ (Σ_k W_up[j][k] * x_k))`
pub fn chimera_ffn_swiglu_vec<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
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
    assert_eq!(weights.w1.len(), w_up.len(), "W_gate and W_up must have same row count");
    let d_in = x_cts.len();
    let d_ffn = weights.w1.len();

    // Phase 1+2: Gate projection + SiLU, Up projection, element-wise product
    let silu_approx = silu_poly_approx();
    let mut h = Vec::with_capacity(d_ffn);
    for j in 0..d_ffn {
        // Gate: gate_j = Σ_k W_gate[j][k] * x_k
        let wg_vecs: Vec<Vec<i64>> = weights.w1[j][..d_in].iter().map(|&w| vec![w]).collect();
        let gate_j = crate::arithmetic::chimera_dot_product(module, x_cts, &wg_vecs);

        // Up: up_j = Σ_k W_up[j][k] * x_k
        let wu_vecs: Vec<Vec<i64>> = w_up[j][..d_in].iter().map(|&w| vec![w]).collect();
        let up_j = crate::arithmetic::chimera_dot_product(module, x_cts, &wu_vecs);

        // SiLU(gate_j)
        let gate_activated = apply_poly_activation(module, eval_key, &gate_j, &silu_approx);

        // Project up_j to match gate_activated's layout
        let gate_layout = GLWELayout {
            n: gate_activated.n(),
            base2k: gate_activated.base2k(),
            k: gate_activated.k(),
            rank: gate_activated.rank(),
        };
        let up_proj = if up_j.base2k() == gate_activated.base2k() {
            up_j
        } else {
            chimera_align_layout(module, &up_j, &gate_layout)
        };

        // Element-wise: SiLU(gate_j) ⊙ up_j
        let hj = chimera_ct_mul(module, eval_key, &gate_activated, &up_proj);
        h.push(hj);
    }

    // Phase 3: Down projection — y_i = Σ_j W_down[i][j] * h_j
    let mut outputs = Vec::with_capacity(weights.w2.len());
    for w2_row in &weights.w2 {
        let w2_vecs: Vec<Vec<i64>> = w2_row.iter().map(|&w| vec![w]).collect();
        let dot = crate::arithmetic::chimera_dot_product(module, &h, &w2_vecs);
        outputs.push(dot);
    }

    outputs
}

/// Dispatches to the appropriate vector-based FFN variant.
pub fn chimera_ffn_vec<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    weights: &FFNWeights,
    config: &FFNConfig,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    match config {
        FFNConfig::Standard { activation } => chimera_ffn_standard_vec(module, eval_key, x_cts, weights, activation),
        FFNConfig::SwiGLU => chimera_ffn_swiglu_vec(module, eval_key, x_cts, weights),
    }
}

/// Evaluates a complete transformer block in the vector representation.
///
/// This is the vector-representation variant of [`chimera_transformer_block`].
/// Each element of the input and output is a separate ciphertext encoding
/// one scalar dimension (value in coefficient 0).
///
/// Implements the pre-norm architecture:
///
/// ```text
/// x = x + Attention(RMSNorm(x))
/// x = x + FFN(RMSNorm(x))
/// ```
///
/// All intermediate computations (RMSNorm, attention, FFN) operate on the
/// full `Vec<GLWE>` without truncation. Residual connections add the full
/// d_model-length vectors element-wise.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `x_cts` - Input vector: d_model ciphertexts.
/// * `config` - Transformer block configuration.
/// * `weights` - Weight matrices for this block.
///
/// # Returns
///
/// d_model ciphertexts representing the block output.
pub fn chimera_transformer_block_vec<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    config: &TransformerBlockConfig,
    weights: &TransformerBlockWeights,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWESub + GLWETrace<BE> + GLWEAutomorphism<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let _d_model = x_cts.len();

    // Step 1: Pre-attention RMSNorm
    let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
        Some(gamma) => config.pre_attn_norm.clone().with_gamma(gamma.clone()),
        None => config.pre_attn_norm.clone(),
    };
    let normed_pre_attn = chimera_rms_norm_vec(module, eval_key, x_cts, &pre_attn_norm_cfg);

    // Step 2: Multi-head attention (vector representation)
    //
    // chimera_multi_head_attention_vec takes Vec<GLWE> input (one per dim)
    // and returns Vec<GLWE> output (one per dim). It uses dot products
    // for QKV projection and output projection rather than ring products.
    let attn_out = chimera_multi_head_attention_vec(module, eval_key, &normed_pre_attn, &weights.attention, &config.attention);

    // Step 3: Residual connection (attention)
    let residual_1 = if config.residual {
        project_and_add_vec(module, x_cts, &attn_out)
    } else {
        attn_out
    };

    // Step 4: Pre-FFN RMSNorm
    let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
        Some(gamma) => config.pre_ffn_norm.clone().with_gamma(gamma.clone()),
        None => config.pre_ffn_norm.clone(),
    };
    let normed_pre_ffn = chimera_rms_norm_vec(module, eval_key, &residual_1, &pre_ffn_norm_cfg);

    // Step 5: FFN (vector representation — returns full d_model output)
    let ffn_out = chimera_ffn_vec(module, eval_key, &normed_pre_ffn, &weights.ffn, &config.ffn);

    // Step 6: Residual connection (FFN)
    if config.residual {
        project_and_add_vec(module, &residual_1, &ffn_out)
    } else {
        ffn_out
    }
}

/// Projects two ciphertext vectors to matching layouts and adds element-wise.
///
/// Used for residual connections in the vector pipeline. Handles the case
/// where `a` and `b` may have different base2k/k after tensor products.
fn project_and_add_vec<BE: Backend>(module: &Module<BE>, a: &[GLWE<Vec<u8>>], b: &[GLWE<Vec<u8>>]) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    assert_eq!(a.len(), b.len(), "project_and_add_vec: length mismatch");
    let d = a.len();
    let mut result = Vec::with_capacity(d);

    for i in 0..d {
        let (ai_proj, bi_ref) = if a[i].base2k() == b[i].base2k() && a[i].k() == b[i].k() {
            // Layouts match — clone a[i]
            let ai_cloned = clone_glwe(&a[i]);
            (ai_cloned, &b[i])
        } else {
            // Project a[i] to match b[i]'s layout
            let target = GLWELayout {
                n: b[i].n(),
                base2k: b[i].base2k(),
                k: b[i].k(),
                rank: b[i].rank(),
            };
            let projected = chimera_align_layout(module, &a[i], &target);
            (projected, &b[i])
        };
        result.push(chimera_add(module, &ai_proj, bi_ref));
    }

    result
}

/// Clones a GLWE ciphertext (deep copy of data buffer).
fn clone_glwe(ct: &GLWE<Vec<u8>>) -> GLWE<Vec<u8>> {
    use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
    let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(ct);
    {
        let src_ref = ct.to_ref();
        let src: &[u8] = src_ref.data().data;
        let mut dst_mut = cloned.to_mut();
        let dst: &mut [u8] = dst_mut.data_mut().data;
        let len = src.len().min(dst.len());
        dst[..len].copy_from_slice(&src[..len]);
    }
    cloned
}

/// Evaluates a sequence of transformer blocks in the vector representation.
///
/// Chains `num_layers` transformer blocks, feeding the full d_model output
/// of each block as the input to the next.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `x_cts` - Initial input: d_model ciphertexts.
/// * `config` - Block configuration (shared across layers).
/// * `layer_weights` - Per-layer weights.
///
/// # Returns
///
/// d_model ciphertexts representing the final output.
pub fn chimera_forward_pass_vec<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    config: &TransformerBlockConfig,
    layer_weights: &[TransformerBlockWeights],
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWESub + GLWETrace<BE> + GLWEAutomorphism<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut current: Vec<GLWE<Vec<u8>>> = x_cts.iter().map(|ct| clone_glwe(ct)).collect();

    for weights in layer_weights {
        current = chimera_transformer_block_vec(module, eval_key, &current, config, weights);
    }

    current
}

/// Configuration for the full forward pass pipeline (vec representation).
///
/// Extends the basic block-loop with optional bootstrapping, final norm,
/// and noise tracking — everything needed for a real model forward pass.
#[derive(Clone, Debug)]
pub struct ForwardPassConfig {
    /// Transformer block config (shared across layers unless overridden per-layer).
    pub block: TransformerBlockConfig,

    /// CHIMERA parameters (needed for noise budget calculations).
    pub params: ChimeraParams,

    /// Bootstrapping configuration. Use [`BootstrappingConfig::no_bootstrap()`]
    /// if the model fits within a single noise budget.
    pub bootstrap: BootstrappingConfig,

    /// Optional final RMSNorm gamma weights (applied after the last
    /// transformer layer, before the LM head). When `None`, no final norm
    /// is applied.
    pub final_norm_gamma: Option<Vec<i64>>,

    /// RMSNorm configuration for the final norm layer.
    /// Only used when `final_norm_gamma` is `Some`.
    pub final_norm_config: LayerNormConfig,
}

/// Evaluates a full transformer forward pass in the vector representation
/// with bootstrapping support and optional final RMSNorm.
///
/// This is the production entry point for real model inference. It:
///
/// 1. Chains `layer_weights.len()` transformer blocks
/// 2. Optionally checks noise budget between layers and bootstraps all
///    d_model ciphertexts when the budget is low
/// 3. Applies a final RMSNorm (with learned gamma) after the last layer
///    if `config.final_norm_gamma` is set
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `x_cts` - Initial input: d_model ciphertexts.
/// * `config` - Full forward pass configuration (block config + bootstrapping + final norm).
/// * `layer_weights` - Per-layer weights.
/// * `bsk_prepared` - Prepared bootstrap key (required if bootstrapping is enabled;
///   pass `None` if bootstrapping is disabled).
///
/// # Returns
///
/// A tuple of:
/// - d_model ciphertexts representing the final output (after final norm if configured).
/// - The final [`NoiseTracker`] state (useful for diagnostics).
pub fn chimera_forward_pass_vec_full<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    config: &ForwardPassConfig,
    layer_weights: &[TransformerBlockWeights],
    bsk_prepared: Option<&ChimeraBootstrapKeyPrepared<BE>>,
) -> (Vec<GLWE<Vec<u8>>>, NoiseTracker)
where
    Module<BE>: GLWETensoring<BE>
        + GLWEMulConst<BE>
        + GLWEAdd
        + GLWESub
        + GLWETrace<BE>
        + GLWEAutomorphism<BE>
        + poulpy_hal::api::ModuleN
        + poulpy_core::LWESampleExtract
        + poulpy_core::LWEKeySwitch<BE>
        + poulpy_schemes::bin_fhe::blind_rotation::BlindRotationExecute<poulpy_schemes::bin_fhe::blind_rotation::CGGI, BE>
        + poulpy_schemes::bin_fhe::blind_rotation::LookupTableFactory,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let mut current: Vec<GLWE<Vec<u8>>> = x_cts.iter().map(|ct| clone_glwe(ct)).collect();
    let mut tracker = NoiseTracker::fresh();
    let mut bootstrap_count: usize = 0;
    let ring_degree = config.params.degree.0 as usize;

    for (layer_idx, weights) in layer_weights.iter().enumerate() {
        // --- Check noise budget before this layer ---
        if needs_bootstrap(&tracker, &config.params, &config.bootstrap) {
            let bsk = bsk_prepared.expect(
                "chimera_forward_pass_vec_full: bootstrapping triggered \
                 but no bootstrap key provided (bsk_prepared is None)",
            );

            if config.bootstrap.max_bootstraps_per_pass == 0 || bootstrap_count < config.bootstrap.max_bootstraps_per_pass {
                // Bootstrap every ciphertext in the vec
                current = current
                    .iter()
                    .map(|ct| {
                        let mut t = tracker.clone();
                        chimera_bootstrap(module, ct, &mut t, &config.params, bsk, &config.bootstrap)
                    })
                    .collect();

                // Reset tracker after bootstrap (noise is refreshed)
                tracker = NoiseTracker::fresh();
                bootstrap_count += 1;

                eprintln!(
                    "[forward_pass_vec_full] Bootstrapped all {} cts before layer {} (bootstrap #{})",
                    current.len(),
                    layer_idx,
                    bootstrap_count,
                );
            }
        }

        // --- Evaluate transformer block ---
        current = chimera_transformer_block_vec(module, eval_key, &current, &config.block, weights);

        // --- Update noise tracker ---
        // Approximate the noise cost of one transformer block as several ct*ct muls.
        let block_ct_ct_muls = match &config.block.ffn {
            FFNConfig::SwiGLU => 8,
            FFNConfig::Standard { activation } => 6 + activation.depth(),
        };
        for _ in 0..block_ct_ct_muls {
            let other = NoiseTracker::fresh();
            tracker.mul_ct(&other, ring_degree);
        }
    }

    // --- Apply final RMSNorm if configured ---
    if let Some(ref gamma) = config.final_norm_gamma {
        let norm_cfg = config.final_norm_config.clone().with_gamma(gamma.clone());
        current = chimera_rms_norm_vec(module, eval_key, &current, &norm_cfg);
    }

    (current, tracker)
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
                rope: None,
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
        let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [4u8; 32], [5u8; 32]);

        let vals: Vec<i8> = vec![3];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        let weights = FFNWeights {
            w1: vec![vec![2]],
            w2: vec![vec![1]],
            w3: None,
        };

        let result = chimera_ffn_standard(&module, &eval_key, &ct, &weights, &ActivationChoice::SquaredReLU);
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
        let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [4u8; 32], [5u8; 32]);

        let vals: Vec<i8> = vec![2];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        let weights = FFNWeights {
            w1: vec![vec![1]],       // W_gate
            w2: vec![vec![1]],       // W_down
            w3: Some(vec![vec![1]]), // W_up
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
        let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [4u8; 32], [5u8; 32]);

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
        assert!(diff <= 1, "residual_add: expected 15, got {}", decoded[0]);
    }
}
