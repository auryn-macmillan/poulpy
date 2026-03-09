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

use crate::attention::{AttentionConfig, AttentionPlan, SoftmaxStrategy, plan_attention};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Precision, SecurityLevel};

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
}
