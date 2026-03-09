//! Transformer attention mechanism under FHE.
//!
//! Implements the multi-head self-attention computation of a transformer
//! decoder layer, with all operations expressed in terms of CHIMERA's
//! homomorphic primitives.
//!
//! ## Computation Flow
//!
//! ```text
//! Input: x [d_model]
//!
//! 1. Q, K, V projections:
//!    Q = x · W_Q    (matmul: encrypted x, cleartext W_Q)
//!    K = x · W_K
//!    V = x · W_V
//!
//! 2. Attention scores:
//!    scores = Q · K^T / √d_k   (encrypted × encrypted, needs tensor product)
//!
//! 3. Softmax approximation:
//!    attn = poly_softmax(scores)
//!
//! 4. Output:
//!    context = attn · V
//!    output = context · W_O
//! ```
//!
//! ## Packing Strategy
//!
//! For attention computation, values are packed at head granularity:
//! - Each ciphertext holds d_head values for one attention head
//! - Q, K, V for all heads are stored as arrays of ciphertexts
//! - This aligns naturally with the head-parallel structure of multi-head attention

use crate::activations::PolyApprox;
use crate::params::{ChimeraParams, ModelDims};

/// Configuration for a single attention layer under FHE.
#[derive(Clone, Debug)]
pub struct AttentionConfig {
    /// Model dimensions.
    pub dims: ModelDims,
    /// CHIMERA parameters.
    pub params: ChimeraParams,
    /// Polynomial approximation for softmax.
    pub softmax_approx: SoftmaxStrategy,
    /// Whether to use causal masking.
    pub causal: bool,
}

/// Strategy for evaluating softmax under FHE.
#[derive(Clone, Debug)]
pub enum SoftmaxStrategy {
    /// Polynomial softmax: (1+x+x²/2)² / Σ(1+x+x²/2)².
    /// Degree 4, multiplicative depth 3.
    PolynomialDeg4,
    /// Linear attention: replace softmax with identity (no normalisation).
    /// Depth 0, but may degrade model quality.
    Linear,
    /// ReLU-squared attention: replace exp(x) with max(0, x)².
    /// Degree 2, depth 1.
    ReluSquared,
    /// Custom polynomial approximation.
    Custom(PolyApprox),
}

impl SoftmaxStrategy {
    /// Returns the multiplicative depth of this softmax strategy.
    pub fn depth(&self) -> usize {
        match self {
            SoftmaxStrategy::PolynomialDeg4 => 3,
            SoftmaxStrategy::Linear => 0,
            SoftmaxStrategy::ReluSquared => 1,
            SoftmaxStrategy::Custom(p) => p.degree(),
        }
    }

    /// Returns a human-readable name.
    pub fn name(&self) -> &str {
        match self {
            SoftmaxStrategy::PolynomialDeg4 => "poly_softmax_deg4",
            SoftmaxStrategy::Linear => "linear_attention",
            SoftmaxStrategy::ReluSquared => "relu_squared",
            SoftmaxStrategy::Custom(_) => "custom",
        }
    }
}

/// Weight matrices for a single attention layer (in plaintext).
///
/// Each matrix is stored as a flat vector of i64 values (quantised INT8
/// weights scaled to torus representation).
#[derive(Clone, Debug)]
pub struct AttentionWeights {
    /// Query projection: [d_model × d_model]
    pub w_q: Vec<Vec<i64>>,
    /// Key projection: [d_model × d_model]
    pub w_k: Vec<Vec<i64>>,
    /// Value projection: [d_model × d_model]
    pub w_v: Vec<Vec<i64>>,
    /// Output projection: [d_model × d_model]
    pub w_o: Vec<Vec<i64>>,
}

impl AttentionWeights {
    /// Creates zero-initialised weights for the given dimensions.
    pub fn zeros(dims: &ModelDims) -> Self {
        let d = dims.d_model;
        AttentionWeights {
            w_q: vec![vec![0i64; d]; d],
            w_k: vec![vec![0i64; d]; d],
            w_v: vec![vec![0i64; d]; d],
            w_o: vec![vec![0i64; d]; d],
        }
    }
}

/// Plan for attention computation, documenting the cost.
#[derive(Clone, Debug)]
pub struct AttentionPlan {
    /// Number of ciphertext-plaintext multiplications for QKV projection.
    pub qkv_muls: usize,
    /// Number of ciphertext-ciphertext multiplications for attention scores.
    pub score_muls: usize,
    /// Multiplicative depth of softmax.
    pub softmax_depth: usize,
    /// Number of ciphertext-ciphertext multiplications for context computation.
    pub context_muls: usize,
    /// Number of ciphertext-plaintext multiplications for output projection.
    pub output_muls: usize,
    /// Total multiplicative depth of the attention layer.
    pub total_depth: usize,
    /// Number of rotation operations needed.
    pub num_rotations: usize,
    /// Number of ciphertexts in flight (memory estimate).
    pub ciphertexts_in_flight: usize,
}

/// Plans an attention computation, returning cost estimates.
pub fn plan_attention(config: &AttentionConfig) -> AttentionPlan {
    let d = config.dims.d_model;
    let h = config.dims.n_heads;
    let dk = config.dims.d_head;

    // QKV projections: 3 matrix-vector products, each d_model multiplications
    let qkv_muls = 3 * d;

    // Attention scores: h heads, each requires dk multiplications + log₂(dk) rotations
    let log_dk = (dk as f64).log2().ceil() as usize;
    let score_muls = h * dk;

    let softmax_depth = config.softmax_approx.depth();

    // Context = attn · V: h heads, dk multiplications each
    let context_muls = h * dk;

    // Output projection: d_model multiplications
    let output_muls = d;

    // Total depth: QKV(1) + scores(1) + softmax(var) + context(1) + output(1)
    let total_depth = 1 + 1 + softmax_depth + 1 + 1;

    // Rotations: QKV reduction + score reduction + context reduction + output reduction
    let num_rotations = 3 * log_dk + h * log_dk + h * log_dk + log_dk;

    // Ciphertexts in flight: Q + K + V + scores + attn + context + output
    let ciphertexts_in_flight = 3 * h + h + h + h + 1;

    AttentionPlan {
        qkv_muls,
        score_muls,
        softmax_depth,
        context_muls,
        output_muls,
        total_depth,
        num_rotations,
        ciphertexts_in_flight,
    }
}

/// Evaluates polynomial softmax on plaintext scores (for verification).
///
/// poly_softmax(x_i) = p(x_i) / Σ_j p(x_j)
/// where p(x) = (1 + x + x²/2)² (degree-4 approximation of exp)
pub fn poly_softmax_plaintext(scores: &[f64]) -> Vec<f64> {
    // p(x) = (1 + x + x²/2)²
    let p = |x: f64| -> f64 {
        let inner = 1.0 + x + 0.5 * x * x;
        inner * inner
    };

    let exps: Vec<f64> = scores.iter().map(|&s| p(s)).collect();
    let sum: f64 = exps.iter().sum();

    if sum.abs() < 1e-10 {
        // Uniform distribution as fallback
        vec![1.0 / scores.len() as f64; scores.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// Evaluates ReLU-squared softmax on plaintext scores (for verification).
///
/// relu2_softmax(x_i) = max(0, x_i)² / Σ_j max(0, x_j)²
pub fn relu_squared_softmax_plaintext(scores: &[f64]) -> Vec<f64> {
    let sq_relu: Vec<f64> = scores
        .iter()
        .map(|&s| {
            let r = s.max(0.0);
            r * r
        })
        .collect();
    let sum: f64 = sq_relu.iter().sum();

    if sum.abs() < 1e-10 {
        vec![1.0 / scores.len() as f64; scores.len()]
    } else {
        sq_relu.iter().map(|v| v / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Precision, SecurityLevel};

    #[test]
    fn test_poly_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let result = poly_softmax_plaintext(&scores);

        // Should sum to ~1
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {sum}");

        // Should be monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_relu_squared_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let result = relu_squared_softmax_plaintext(&scores);

        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_softmax_with_negatives() {
        let scores = vec![-1.0, 0.0, 1.0];
        let result = poly_softmax_plaintext(&scores);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_attention_plan() {
        let dims = ModelDims::dense_7b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = AttentionConfig {
            dims: dims.clone(),
            params,
            softmax_approx: SoftmaxStrategy::PolynomialDeg4,
            causal: true,
        };

        let plan = plan_attention(&config);
        assert_eq!(plan.qkv_muls, 3 * dims.d_model);
        assert_eq!(plan.softmax_depth, 3);
        assert!(plan.total_depth > 0);
        assert!(plan.num_rotations > 0);
        assert!(plan.ciphertexts_in_flight > 0);
    }

    #[test]
    fn test_softmax_strategy_depth() {
        assert_eq!(SoftmaxStrategy::PolynomialDeg4.depth(), 3);
        assert_eq!(SoftmaxStrategy::Linear.depth(), 0);
        assert_eq!(SoftmaxStrategy::ReluSquared.depth(), 1);
    }
}
