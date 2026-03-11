//! Realistic f32/f64 plaintext inference pipeline.
//!
//! This module provides a gold-standard reference implementation of
//! transformer inference using standard floating-point arithmetic with
//! **exact** nonlinearities (real exp, real 1/sqrt, real sigmoid, etc.).
//!
//! The purpose is to enable quantitative comparison between FHE inference
//! outputs and "what the model would actually produce" in cleartext,
//! decomposing the error into:
//!
//! - **(a) Polynomial approximation error**: The gap between exact softmax /
//!   SiLU / GELU and their low-degree polynomial approximations used under FHE.
//! - **(b) INT8 quantisation error**: The loss from using quantised i64 weights
//!   instead of full-precision floats.
//! - **(c) FHE noise**: Additional error introduced by the RLWE encryption
//!   and homomorphic operations.
//!
//! ## Design Philosophy
//!
//! The encrypted pipeline should mirror a realistic plaintext pipeline, not
//! the other way around. This module answers the question: "does the
//! encrypted inference give me the same answer as running the model
//! normally?"
//!
//! ## Weight Representation
//!
//! All weights are `Vec<Vec<i64>>` (INT8 quantised, stored as i64). This
//! module converts them to f64 for computation: `w_f64 = w_i64 as f64`.
//! The weights are already in INT8 scale, so no additional scaling is needed
//! for the plaintext reference — the comparison isolates FHE noise and
//! polynomial approximation error while keeping quantisation identical.
//!
//! ## Module Structure
//!
//! The module mirrors the FHE pipeline structure:
//!
//! | Plaintext function | FHE counterpart |
//! |--------------------|-----------------|
//! | `dot_product` | `chimera_dot_product` |
//! | `matvec` | `chimera_dot_product` per row |
//! | `rms_norm` | `chimera_rms_norm_vec` |
//! | `exact_softmax` | `chimera_apply_softmax` |
//! | `exact_silu` | `apply_poly_activation` with SiLU poly |
//! | `exact_gelu` | `apply_poly_activation` with GELU poly |
//! | `ffn_swiglu` | `chimera_ffn_swiglu_vec` |
//! | `ffn_standard` | `chimera_ffn_standard_vec` |
//! | `single_head_attention` | `chimera_single_head_attention` |
//! | `multi_head_attention` | `chimera_multi_head_attention_vec` |
//! | `transformer_block` | `chimera_transformer_block_vec` |
//! | `forward_pass` | `chimera_forward_pass_vec` |

use crate::attention::{AttentionConfig, AttentionWeights, RoPEConfig, SoftmaxStrategy};
use crate::transformer::{ActivationChoice, FFNConfig, FFNWeights, TransformerBlockConfig, TransformerBlockWeights};

// ---------------------------------------------------------------------------
// Basic linear algebra
// ---------------------------------------------------------------------------

/// Computes the dot product of two f64 vectors.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot_product: length mismatch");
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Computes a matrix-vector product: `y[i] = Σ_j W[i][j] * x[j]`.
///
/// Weight matrix `w` is `[n_out × n_in]` stored as `Vec<Vec<i64>>`.
/// Input `x` is `[n_in]` as `&[f64]`. Output is `[n_out]` as `Vec<f64>`.
pub fn matvec(w: &[Vec<i64>], x: &[f64]) -> Vec<f64> {
    w.iter()
        .map(|row| {
            let d_in = x.len().min(row.len());
            row[..d_in]
                .iter()
                .zip(x[..d_in].iter())
                .map(|(&wi, &xi)| wi as f64 * xi)
                .sum()
        })
        .collect()
}

/// Element-wise addition of two vectors.
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vec_add: length mismatch");
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect()
}

/// Element-wise multiplication of two vectors.
pub fn vec_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vec_mul: length mismatch");
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).collect()
}

// ---------------------------------------------------------------------------
// Exact nonlinearities
// ---------------------------------------------------------------------------

/// Exact softmax: `softmax(x_i) = exp(x_i) / Σ_j exp(x_j)`.
///
/// Uses the standard numerically-stable implementation with max subtraction.
pub fn exact_softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum.abs() < 1e-30 {
        vec![1.0 / scores.len() as f64; scores.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// Exact SiLU (Swish): `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
pub fn exact_silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Exact sigmoid: `σ(x) = 1 / (1 + exp(-x))`.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Exact GELU: `gelu(x) = x * Φ(x)` where `Φ` is the standard normal CDF.
///
/// Uses the tanh approximation:
/// `gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`.
pub fn exact_gelu(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

/// Exact squared ReLU: `squared_relu(x) = max(0, x)²`.
pub fn exact_squared_relu(x: f64) -> f64 {
    let r = x.max(0.0);
    r * r
}

/// Exact RMS normalisation.
///
/// `rms_norm(x)_i = x_i / sqrt(mean(x²) + ε)`
///
/// Optionally applies learned gamma scaling: `out_i = rms_norm(x)_i * γ_i`.
/// Gamma values are INT8-scaled integers (divided by 256 to convert to f64).
pub fn rms_norm(x: &[f64], gamma: Option<&[i64]>, epsilon: f64) -> Vec<f64> {
    let n = x.len();
    assert!(n > 0, "rms_norm: input must not be empty");

    let sum_sq: f64 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f64 + epsilon).sqrt();
    let inv_rms = 1.0 / rms;

    let mut result: Vec<f64> = x.iter().map(|&v| v * inv_rms).collect();

    if let Some(gamma) = gamma {
        for (i, r) in result.iter_mut().enumerate() {
            if i < gamma.len() {
                // Gamma is stored as a scaled i64; convert to f64.
                // The FHE layernorm uses gamma[i] / 256.0 (COEFF_SCALE_BITS = 8).
                let g = gamma[i] as f64 / 256.0;
                *r *= g;
            }
        }
    }

    result
}

/// Applies RoPE (Rotary Position Embedding) to a vector of f64 values.
///
/// Mirrors `chimera_apply_rope_vec` but in f64 domain.
///
/// For each pair (2i, 2i+1):
/// - `out[2i]   = q[2i] * cos[i] - q[2i+1] * sin[i]`
/// - `out[2i+1] = q[2i] * sin[i] + q[2i+1] * cos[i]`
///
/// The cos/sin values in `RoPEConfig` are scaled integers; we convert them
/// to f64 by dividing by the scale factor used during precomputation.
pub fn apply_rope(q_or_k: &[f64], rope: &RoPEConfig, rope_scale_bits: u32) -> Vec<f64> {
    let d = q_or_k.len();
    assert!(d % 2 == 0, "apply_rope: dimension must be even");
    assert_eq!(d, rope.cos_table.len() * 2, "apply_rope: dimension mismatch with RoPE table");

    let scale = (1i64 << rope_scale_bits) as f64;
    let n_pairs = d / 2;
    let mut out = Vec::with_capacity(d);

    for i in 0..n_pairs {
        let cos_i = rope.cos_table[i] as f64 / scale;
        let sin_i = rope.sin_table[i] as f64 / scale;

        let even = q_or_k[2 * i];
        let odd = q_or_k[2 * i + 1];

        out.push(even * cos_i - odd * sin_i);
        out.push(even * sin_i + odd * cos_i);
    }

    out
}

/// Precomputes RoPE rotation factors in f64 domain (exact, not quantised).
///
/// Returns (cos_table, sin_table) as `Vec<f64>`, each of length `d_head / 2`.
pub fn precompute_rope_f64(position: usize, d_head: usize, base: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(d_head % 2 == 0, "precompute_rope_f64: d_head must be even");
    let n_pairs = d_head / 2;
    let mut cos_table = Vec::with_capacity(n_pairs);
    let mut sin_table = Vec::with_capacity(n_pairs);

    for i in 0..n_pairs {
        let theta = (position as f64) / base.powf(2.0 * i as f64 / d_head as f64);
        cos_table.push(theta.cos());
        sin_table.push(theta.sin());
    }

    (cos_table, sin_table)
}

/// Applies exact RoPE rotation using f64 cos/sin tables (no quantisation).
pub fn apply_rope_exact(q_or_k: &[f64], cos_table: &[f64], sin_table: &[f64]) -> Vec<f64> {
    let d = q_or_k.len();
    assert!(d % 2 == 0, "apply_rope_exact: dimension must be even");
    assert_eq!(d, cos_table.len() * 2);
    assert_eq!(cos_table.len(), sin_table.len());

    let n_pairs = d / 2;
    let mut out = Vec::with_capacity(d);

    for i in 0..n_pairs {
        let cos_i = cos_table[i];
        let sin_i = sin_table[i];
        let even = q_or_k[2 * i];
        let odd = q_or_k[2 * i + 1];

        out.push(even * cos_i - odd * sin_i);
        out.push(even * sin_i + odd * cos_i);
    }

    out
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Computes Q, K, V projections in f64 domain.
///
/// Returns (Q, K, V) where Q has `n_q_rows` elements and K/V have `n_kv_rows`.
pub fn qkv_project(x: &[f64], weights: &AttentionWeights, n_q_rows: usize, n_kv_rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let project = |w: &[Vec<i64>], n_rows: usize| -> Vec<f64> {
        w.iter()
            .take(n_rows)
            .map(|row| {
                let d_in = x.len().min(row.len());
                row[..d_in]
                    .iter()
                    .zip(x[..d_in].iter())
                    .map(|(&wi, &xi)| wi as f64 * xi)
                    .sum()
            })
            .collect()
    };

    let q = project(&weights.w_q, n_q_rows);
    let k = project(&weights.w_k, n_kv_rows);
    let v = project(&weights.w_v, n_kv_rows);
    (q, k, v)
}

/// Computes single-head attention with exact softmax.
///
/// Given Q, K, V slices for one head (each of length d_head):
/// 1. score = Q · K (dot product)
/// 2. attn = softmax([score]) = 1.0 (for single-position attention)
/// 3. context[j] = attn * V[j]
///
/// For single-position (seq_len=1), softmax of a single score is always 1.0,
/// so context = V. This matches the FHE behaviour.
pub fn single_head_attention(q_h: &[f64], k_h: &[f64], v_h: &[f64]) -> Vec<f64> {
    let d_head = q_h.len();
    assert_eq!(d_head, k_h.len());
    assert_eq!(d_head, v_h.len());

    // Score: dot product of Q and K
    let score = dot_product(q_h, k_h);

    // Softmax over a single score is always 1.0
    let attn = exact_softmax(&[score]);
    let attn_weight = attn[0];

    // Context: attn_weight * V[j] for each dimension
    v_h.iter().map(|&vj| attn_weight * vj).collect()
}

/// Computes multi-head attention in f64 domain with exact nonlinearities.
///
/// Mirrors `chimera_multi_head_attention_vec`:
/// 1. QKV projection via dot products (with GQA support)
/// 2. Optional RoPE rotation
/// 3. Per-head: score = Q·K, softmax, context = attn*V
/// 4. Output projection
///
/// Uses exact softmax instead of polynomial approximation.
pub fn multi_head_attention(x: &[f64], weights: &AttentionWeights, config: &AttentionConfig) -> Vec<f64> {
    let d_model = config.dims.d_model;
    let n_heads = config.dims.n_heads;
    let d_head = config.dims.d_head;
    let d_kv = config.dims.d_kv();
    let gqa_group = config.dims.gqa_group_size();

    assert_eq!(d_model, n_heads * d_head);
    assert_eq!(x.len(), d_model);

    // Step 1: QKV projections
    let (q_all, k_all, v_all) = qkv_project(x, weights, d_model, d_kv);

    // Step 1b: Apply RoPE if configured
    // For the plaintext reference, we use the same quantised RoPE tables
    // to keep the comparison fair (isolate only FHE noise, not RoPE
    // quantisation differences).
    let (q_all, k_all) = if let Some(rope) = &config.rope {
        let mut q_rope = Vec::with_capacity(q_all.len());
        let mut k_rope = Vec::with_capacity(k_all.len());

        // Default scale_bits = 7 (matching precompute_rope convention)
        let rope_scale_bits = 7u32;

        for h in 0..n_heads {
            let q_start = h * d_head;
            let q_end = q_start + d_head;
            let q_h_rotated = apply_rope(&q_all[q_start..q_end], rope, rope_scale_bits);
            q_rope.extend(q_h_rotated);
        }

        let n_kv_heads = d_kv / d_head;
        for kv_h in 0..n_kv_heads {
            let kv_start = kv_h * d_head;
            let kv_end = kv_start + d_head;
            let k_h_rotated = apply_rope(&k_all[kv_start..kv_end], rope, rope_scale_bits);
            k_rope.extend(k_h_rotated);
        }

        (q_rope, k_rope)
    } else {
        (q_all, k_all)
    };

    // Step 2: Per-head attention with GQA mapping
    let mut all_head_contexts: Vec<f64> = Vec::with_capacity(d_model);

    for h in 0..n_heads {
        let q_start = h * d_head;
        let q_end = q_start + d_head;
        let q_h = &q_all[q_start..q_end];

        let kv_h = h / gqa_group;
        let kv_start = kv_h * d_head;
        let kv_end = kv_start + d_head;
        let k_h = &k_all[kv_start..kv_end];
        let v_h = &v_all[kv_start..kv_end];

        let head_ctx = single_head_attention(q_h, k_h, v_h);
        all_head_contexts.extend(head_ctx);
    }

    // Step 3: Output projection
    matvec(&weights.w_o, &all_head_contexts)
}

// ---------------------------------------------------------------------------
// FFN
// ---------------------------------------------------------------------------

/// Computes a standard FFN: `y = W2 · activation(W1 · x)`.
///
/// Uses exact activation functions (not polynomial approximations).
pub fn ffn_standard(x: &[f64], weights: &FFNWeights, activation: &ActivationChoice) -> Vec<f64> {
    // Phase 1: Up-projection
    let h = matvec(&weights.w1, x);

    // Phase 2: Activation
    let act_fn = match activation {
        ActivationChoice::PolyGELU | ActivationChoice::LutGELU => exact_gelu,
        ActivationChoice::SquaredReLU => exact_squared_relu,
        ActivationChoice::PolySiLU => exact_silu,
    };
    let h_act: Vec<f64> = h.iter().map(|&v| act_fn(v)).collect();

    // Phase 3: Down-projection
    matvec(&weights.w2, &h_act)
}

/// Computes a SwiGLU FFN: `y = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))`.
///
/// Uses exact SiLU (not polynomial approximation).
pub fn ffn_swiglu(x: &[f64], weights: &FFNWeights) -> Vec<f64> {
    let w_up = weights.w3.as_ref().expect("ffn_swiglu: SwiGLU requires w3 (W_up)");

    // Gate projection + SiLU
    let gate = matvec(&weights.w1, x);
    let gate_activated: Vec<f64> = gate.iter().map(|&v| exact_silu(v)).collect();

    // Up projection
    let up = matvec(w_up, x);

    // Element-wise product
    let h = vec_mul(&gate_activated, &up);

    // Down projection
    matvec(&weights.w2, &h)
}

/// Dispatches to the appropriate FFN variant.
pub fn ffn(x: &[f64], weights: &FFNWeights, config: &FFNConfig) -> Vec<f64> {
    match config {
        FFNConfig::Standard { activation } => ffn_standard(x, weights, activation),
        FFNConfig::SwiGLU => ffn_swiglu(x, weights),
    }
}

// ---------------------------------------------------------------------------
// Transformer block
// ---------------------------------------------------------------------------

/// Evaluates a complete transformer block in f64 domain.
///
/// Implements the pre-norm architecture with exact nonlinearities:
///
/// ```text
/// x = x + Attention(RMSNorm(x))
/// x = x + FFN(RMSNorm(x))
/// ```
///
/// # Arguments
///
/// * `x` - Input vector of length d_model.
/// * `config` - Transformer block configuration.
/// * `weights` - Weight matrices for this block.
///
/// # Returns
///
/// Output vector of length d_model.
pub fn transformer_block(x: &[f64], config: &TransformerBlockConfig, weights: &TransformerBlockWeights) -> Vec<f64> {
    let epsilon = config.pre_attn_norm.epsilon;

    // Step 1: Pre-attention RMSNorm
    let normed_pre_attn = rms_norm(x, weights.pre_attn_norm_gamma.as_deref(), epsilon);

    // Step 2: Multi-head attention
    let attn_out = multi_head_attention(&normed_pre_attn, &weights.attention, &config.attention);

    // Step 3: Residual connection (attention)
    let residual_1 = if config.residual { vec_add(x, &attn_out) } else { attn_out };

    // Step 4: Pre-FFN RMSNorm
    let normed_pre_ffn = rms_norm(
        &residual_1,
        weights.pre_ffn_norm_gamma.as_deref(),
        config.pre_ffn_norm.epsilon,
    );

    // Step 5: FFN
    let ffn_out = ffn(&normed_pre_ffn, &weights.ffn, &config.ffn);

    // Step 6: Residual connection (FFN)
    if config.residual {
        vec_add(&residual_1, &ffn_out)
    } else {
        ffn_out
    }
}

/// Evaluates a sequence of transformer blocks in f64 domain.
///
/// # Arguments
///
/// * `x` - Initial input vector (d_model).
/// * `config` - Block configuration (shared across layers).
/// * `layer_weights` - Per-layer weights.
///
/// # Returns
///
/// Final hidden state vector (d_model).
pub fn forward_pass(x: &[f64], config: &TransformerBlockConfig, layer_weights: &[TransformerBlockWeights]) -> Vec<f64> {
    let mut current = x.to_vec();
    for weights in layer_weights {
        current = transformer_block(&current, config, weights);
    }
    current
}

/// Evaluates a full forward pass with optional final RMSNorm.
///
/// This mirrors the FHE `fhe_forward` method in `inference.rs`.
///
/// # Arguments
///
/// * `x` - Input vector (d_model).
/// * `config` - Block configuration.
/// * `layer_weights` - Per-layer weights.
/// * `final_norm_gamma` - Optional final RMSNorm gamma.
/// * `epsilon` - Epsilon for final norm (typically 1e-5).
///
/// # Returns
///
/// Final hidden state after all layers and optional final norm.
pub fn forward_pass_with_final_norm(
    x: &[f64],
    config: &TransformerBlockConfig,
    layer_weights: &[TransformerBlockWeights],
    final_norm_gamma: Option<&[i64]>,
    epsilon: f64,
) -> Vec<f64> {
    let mut current = forward_pass(x, config, layer_weights);

    if let Some(gamma) = final_norm_gamma {
        current = rms_norm(&current, Some(gamma), epsilon);
    }

    current
}

// ---------------------------------------------------------------------------
// LM head (cleartext matmul for logits)
// ---------------------------------------------------------------------------

/// Applies the LM head to a hidden state to produce logits.
///
/// The hidden state is in f64 domain; the LM head weights are i64.
/// Each logit is: `logit[i] = Σ_j W_lm[i][j] * hidden[j]`.
pub fn lm_head_forward(hidden: &[f64], lm_head_weights: &[Vec<i64>]) -> Vec<f64> {
    lm_head_weights
        .iter()
        .map(|row| {
            let d = hidden.len().min(row.len());
            row[..d].iter().zip(hidden[..d].iter()).map(|(&w, &h)| w as f64 * h).sum()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

/// Computes error metrics between two f64 vectors.
///
/// Returns `(l_inf, l2, mean_abs_error)`.
///
/// - **L-inf**: Maximum absolute difference across all dimensions.
/// - **L2**: Root mean squared error.
/// - **Mean absolute error**: Average absolute difference.
pub fn error_metrics(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len(), "error_metrics: length mismatch");
    if a.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut l_inf: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let diff = (ai - bi).abs();
        l_inf = l_inf.max(diff);
        sum_sq += diff * diff;
        sum_abs += diff;
    }

    let n = a.len() as f64;
    let l2 = (sum_sq / n).sqrt();
    let mae = sum_abs / n;

    (l_inf, l2, mae)
}

/// Computes error metrics between an i8 FHE result and an f64 plaintext reference.
///
/// Converts the i8 vector to f64 before computing metrics.
pub fn error_metrics_i8_vs_f64(fhe_result: &[i8], reference: &[f64]) -> (f64, f64, f64) {
    let fhe_f64: Vec<f64> = fhe_result.iter().map(|&v| v as f64).collect();
    error_metrics(&fhe_f64, reference)
}

/// Formats error metrics as a human-readable string.
pub fn format_error_metrics(l_inf: f64, l2: f64, mae: f64) -> String {
    format!("L-inf={:.4}, L2={:.4}, MAE={:.4}", l_inf, l2, mae)
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Compares two sets of logits and returns whether they agree on top-1 prediction.
pub fn top1_agrees(logits_a: &[f64], logits_b: &[f64]) -> bool {
    if logits_a.is_empty() || logits_b.is_empty() {
        return false;
    }
    let argmax_a = logits_a
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    let argmax_b = logits_b
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    argmax_a == argmax_b
}

/// Returns the top-k token indices and logit values, sorted by logit descending.
pub fn top_k(logits: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

// ---------------------------------------------------------------------------
// End-to-end plaintext inference step
// ---------------------------------------------------------------------------

/// Result of a single plaintext inference step.
#[derive(Clone, Debug)]
pub struct PlaintextStepResult {
    /// Hidden state after transformer forward pass (f64 domain).
    pub hidden_state: Vec<f64>,
    /// Logits from LM head (f64 domain).
    pub logits: Vec<f64>,
    /// Predicted next token ID (argmax of logits).
    pub token_id: usize,
    /// Top-5 logits for diagnostics.
    pub top_logits: Vec<(usize, f64)>,
}

/// Runs a single plaintext inference step.
///
/// Takes an embedding (as i8 values), runs the transformer forward pass
/// with exact nonlinearities, applies the LM head, and returns the result.
///
/// # Arguments
///
/// * `embedding` - Token embedding as i8 values (d_model dimensions).
/// * `block_config` - Transformer block configuration.
/// * `layer_weights` - Per-layer weights.
/// * `final_norm_gamma` - Optional final RMSNorm gamma weights.
/// * `lm_head_weights` - LM head weight matrix `[vocab_size × d_model]`.
/// * `effective_d_model` - Effective model dimension (for truncated inference).
///
/// # Returns
///
/// A [`PlaintextStepResult`] with the hidden state, logits, and predicted token.
pub fn plaintext_step(
    embedding: &[i8],
    block_config: &TransformerBlockConfig,
    layer_weights: &[TransformerBlockWeights],
    final_norm_gamma: Option<&[i64]>,
    lm_head_weights: &[Vec<i64>],
    effective_d_model: usize,
) -> PlaintextStepResult {
    // Convert i8 embedding to f64
    let x: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();
    assert_eq!(
        x.len(),
        effective_d_model,
        "plaintext_step: embedding length ({}) != effective_d_model ({effective_d_model})",
        x.len()
    );

    // Transformer forward pass with exact nonlinearities
    let hidden = forward_pass_with_final_norm(
        &x,
        block_config,
        layer_weights,
        final_norm_gamma,
        block_config.pre_attn_norm.epsilon,
    );

    // LM head
    let logits = lm_head_forward(&hidden, lm_head_weights);
    let top = top_k(&logits, 5);
    let token_id = if logits.is_empty() { 0 } else { top[0].0 };

    PlaintextStepResult {
        hidden_state: hidden,
        logits,
        token_id,
        top_logits: top,
    }
}

// ---------------------------------------------------------------------------
// Polynomial-approximation reference (for decomposing error sources)
// ---------------------------------------------------------------------------

/// SiLU using the same polynomial approximation as the FHE pipeline.
///
/// This allows isolating FHE noise from polynomial approximation error by
/// comparing: FHE output vs poly-reference vs exact-reference.
pub fn poly_silu(x: f64) -> f64 {
    // Coefficients from activations.rs: [0.0, 0.5, 0.1196, 0.0018]
    // p(x) = 0.0 + 0.5*x + 0.1196*x^2 + 0.0018*x^3
    0.0 + 0.5 * x + 0.1196 * x * x + 0.0018 * x * x * x
}

/// GELU using the same polynomial approximation as the FHE pipeline.
pub fn poly_gelu(x: f64) -> f64 {
    // Coefficients from activations.rs: [0.0, 0.5, 0.247, 0.0]
    // p(x) = 0.0 + 0.5*x + 0.247*x^2
    0.0 + 0.5 * x + 0.247 * x * x
}

/// Squared ReLU using the same polynomial as the FHE pipeline.
pub fn poly_squared_relu(x: f64) -> f64 {
    // [0.0, 0.0, 1.0] → x^2 (no clamping at 0!)
    x * x
}

/// SwiGLU FFN using polynomial SiLU (for error decomposition).
///
/// This allows measuring: exact_silu_error = ffn_swiglu(exact) - ffn_swiglu(poly).
pub fn ffn_swiglu_poly(x: &[f64], weights: &FFNWeights) -> Vec<f64> {
    let w_up = weights.w3.as_ref().expect("ffn_swiglu_poly: SwiGLU requires w3 (W_up)");

    let gate = matvec(&weights.w1, x);
    let gate_activated: Vec<f64> = gate.iter().map(|&v| poly_silu(v)).collect();
    let up = matvec(w_up, x);
    let h = vec_mul(&gate_activated, &up);
    matvec(&weights.w2, &h)
}

/// Transformer block using polynomial approximations (mirrors FHE behaviour exactly).
///
/// Uses polynomial SiLU, polynomial inv_sqrt, etc. — everything the FHE
/// pipeline uses, but without the encryption noise. This isolates the
/// noise contribution from the approximation contribution.
pub fn transformer_block_poly_approx(x: &[f64], config: &TransformerBlockConfig, weights: &TransformerBlockWeights) -> Vec<f64> {
    let epsilon = config.pre_attn_norm.epsilon;

    // Step 1: Pre-attention RMSNorm (exact — same as FHE uses exact 1/rms via polynomial)
    let normed_pre_attn = rms_norm(x, weights.pre_attn_norm_gamma.as_deref(), epsilon);

    // Step 2: Multi-head attention (with polynomial softmax)
    let attn_out = multi_head_attention_poly_softmax(&normed_pre_attn, &weights.attention, &config.attention);

    // Step 3: Residual
    let residual_1 = if config.residual { vec_add(x, &attn_out) } else { attn_out };

    // Step 4: Pre-FFN RMSNorm
    let normed_pre_ffn = rms_norm(
        &residual_1,
        weights.pre_ffn_norm_gamma.as_deref(),
        config.pre_ffn_norm.epsilon,
    );

    // Step 5: FFN with polynomial activations
    let ffn_out = match &config.ffn {
        FFNConfig::SwiGLU => ffn_swiglu_poly(&normed_pre_ffn, &weights.ffn),
        FFNConfig::Standard { activation } => {
            let act_fn: fn(f64) -> f64 = match activation {
                ActivationChoice::PolyGELU | ActivationChoice::LutGELU => poly_gelu,
                ActivationChoice::SquaredReLU => poly_squared_relu,
                ActivationChoice::PolySiLU => poly_silu,
            };
            let h = matvec(&weights.ffn.w1, &normed_pre_ffn);
            let h_act: Vec<f64> = h.iter().map(|&v| act_fn(v)).collect();
            matvec(&weights.ffn.w2, &h_act)
        }
    };

    // Step 6: Residual
    if config.residual {
        vec_add(&residual_1, &ffn_out)
    } else {
        ffn_out
    }
}

/// Multi-head attention with polynomial softmax (mirrors FHE behaviour).
fn multi_head_attention_poly_softmax(x: &[f64], weights: &AttentionWeights, config: &AttentionConfig) -> Vec<f64> {
    let d_model = config.dims.d_model;
    let n_heads = config.dims.n_heads;
    let d_head = config.dims.d_head;
    let d_kv = config.dims.d_kv();
    let gqa_group = config.dims.gqa_group_size();

    let (q_all, k_all, v_all) = qkv_project(x, weights, d_model, d_kv);

    // Apply RoPE if configured
    let (q_all, k_all) = if let Some(rope) = &config.rope {
        let rope_scale_bits = 7u32;
        let mut q_rope = Vec::with_capacity(q_all.len());
        let mut k_rope = Vec::with_capacity(k_all.len());

        for h in 0..n_heads {
            let start = h * d_head;
            let end = start + d_head;
            q_rope.extend(apply_rope(&q_all[start..end], rope, rope_scale_bits));
        }
        let n_kv_heads = d_kv / d_head;
        for kv_h in 0..n_kv_heads {
            let start = kv_h * d_head;
            let end = start + d_head;
            k_rope.extend(apply_rope(&k_all[start..end], rope, rope_scale_bits));
        }
        (q_rope, k_rope)
    } else {
        (q_all, k_all)
    };

    let mut all_head_contexts: Vec<f64> = Vec::with_capacity(d_model);

    for h in 0..n_heads {
        let q_start = h * d_head;
        let q_h = &q_all[q_start..q_start + d_head];

        let kv_h = h / gqa_group;
        let kv_start = kv_h * d_head;
        let k_h = &k_all[kv_start..kv_start + d_head];
        let v_h = &v_all[kv_start..kv_start + d_head];

        let score = dot_product(q_h, k_h);

        // Use the polynomial softmax strategy matching FHE
        let attn_weight = match &config.softmax_approx {
            SoftmaxStrategy::PolynomialDeg4 => {
                let vals = crate::attention::poly_softmax_plaintext(&[score]);
                vals[0]
            }
            SoftmaxStrategy::ReluSquared => {
                let vals = crate::attention::relu_squared_softmax_plaintext(&[score]);
                vals[0]
            }
            SoftmaxStrategy::Linear => score,
            SoftmaxStrategy::Custom(poly) => {
                let p = poly.eval(score);
                p // single score: normalisation = p/p = 1, but keep p for consistency
            }
        };

        let ctx: Vec<f64> = v_h.iter().map(|&vj| attn_weight * vj).collect();
        all_head_contexts.extend(ctx);
    }

    matvec(&weights.w_o, &all_head_contexts)
}

/// Full forward pass using polynomial approximations (mirrors FHE exactly).
pub fn forward_pass_poly_approx(
    x: &[f64],
    config: &TransformerBlockConfig,
    layer_weights: &[TransformerBlockWeights],
    final_norm_gamma: Option<&[i64]>,
    epsilon: f64,
) -> Vec<f64> {
    let mut current = x.to_vec();
    for weights in layer_weights {
        current = transformer_block_poly_approx(&current, config, weights);
    }
    if let Some(gamma) = final_norm_gamma {
        current = rms_norm(&current, Some(gamma), epsilon);
    }
    current
}

// ---------------------------------------------------------------------------
// Three-way comparison
// ---------------------------------------------------------------------------

/// Result of a three-way comparison between FHE, polynomial-reference, and exact-reference.
#[derive(Clone, Debug)]
pub struct ThreeWayComparison {
    /// Error between FHE output and exact plaintext reference.
    /// This is the total error (noise + approximation + quantisation).
    pub fhe_vs_exact: (f64, f64, f64),
    /// Error between polynomial-reference and exact plaintext reference.
    /// This isolates polynomial approximation error.
    pub poly_vs_exact: (f64, f64, f64),
    /// Error between FHE output and polynomial-reference.
    /// This isolates FHE noise (since both use the same polynomials).
    pub fhe_vs_poly: (f64, f64, f64),
}

impl std::fmt::Display for ThreeWayComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Three-way error decomposition:")?;
        writeln!(
            f,
            "  FHE vs exact (total error):      {}",
            format_error_metrics(self.fhe_vs_exact.0, self.fhe_vs_exact.1, self.fhe_vs_exact.2)
        )?;
        writeln!(
            f,
            "  Poly vs exact (approx error):    {}",
            format_error_metrics(self.poly_vs_exact.0, self.poly_vs_exact.1, self.poly_vs_exact.2)
        )?;
        write!(
            f,
            "  FHE vs poly (FHE noise):         {}",
            format_error_metrics(self.fhe_vs_poly.0, self.fhe_vs_poly.1, self.fhe_vs_poly.2)
        )
    }
}

/// Performs a three-way comparison to decompose error sources.
///
/// Given an FHE hidden state (i8), computes:
/// 1. Exact plaintext forward pass (with exact nonlinearities)
/// 2. Polynomial plaintext forward pass (with FHE's polynomial approximations)
/// 3. Error metrics between all three
///
/// # Arguments
///
/// * `fhe_hidden` - Decrypted FHE hidden state (i8 values).
/// * `embedding` - Original token embedding (i8 values).
/// * `block_config` - Transformer block configuration.
/// * `layer_weights` - Per-layer weights.
/// * `final_norm_gamma` - Optional final RMSNorm gamma.
/// * `epsilon` - Epsilon for norm layers.
pub fn three_way_comparison(
    fhe_hidden: &[i8],
    embedding: &[i8],
    block_config: &TransformerBlockConfig,
    layer_weights: &[TransformerBlockWeights],
    final_norm_gamma: Option<&[i64]>,
    epsilon: f64,
) -> ThreeWayComparison {
    let x: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();

    // Exact reference (real softmax, real SiLU, real 1/sqrt)
    let exact = forward_pass_with_final_norm(&x, block_config, layer_weights, final_norm_gamma, epsilon);

    // Polynomial reference (FHE's polynomial approximations, no noise)
    let poly = forward_pass_poly_approx(&x, block_config, layer_weights, final_norm_gamma, epsilon);

    // FHE result (convert i8 to f64)
    let fhe: Vec<f64> = fhe_hidden.iter().map(|&v| v as f64).collect();

    ThreeWayComparison {
        fhe_vs_exact: error_metrics(&fhe, &exact),
        poly_vs_exact: error_metrics(&poly, &exact),
        fhe_vs_poly: error_metrics(&fhe, &poly),
    }
}
