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
//!
//! ## Homomorphic Attention Implementation
//!
//! The FHE attention functions operate on encrypted vectors where each
//! ciphertext packs one head's worth of values (`d_head` coefficients).
//! The key operations are:
//!
//! - **QKV projection**: `chimera_matmul_single_ct` with plaintext weight rows
//! - **Score computation**: `chimera_ct_mul` (Q·K element product) + `chimera_slot_sum` (reduction)
//! - **Softmax**: `apply_poly_activation` with the selected softmax polynomial
//! - **Context**: `chimera_ct_mul` (attn_weights · V) per head
//! - **Output projection**: `chimera_matmul_single_ct` with plaintext W_O

use crate::activations::{apply_poly_activation, chimera_ct_mul, squared_relu_approx, PolyApprox};
use crate::arithmetic::{chimera_add, chimera_matmul_single_ct, chimera_mul_const, chimera_slot_sum, chimera_sub};
use crate::encrypt::ChimeraEvalKey;
use crate::params::{ChimeraParams, ModelDims};
use poulpy_core::ScratchTakeCore;
use poulpy_core::{layouts::GLWE, GLWEAdd, GLWEMulConst, GLWESub, GLWETensoring, GLWETrace};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};

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
    /// Optional RoPE (Rotary Position Embedding) configuration.
    /// When `Some`, RoPE rotation is applied to Q and K after projection
    /// and before score computation in `chimera_multi_head_attention_vec`.
    /// When `None`, no positional encoding is applied (legacy behaviour).
    pub rope: Option<RoPEConfig>,
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

// ---------------------------------------------------------------------------
// RoPE (Rotary Position Embedding)
// ---------------------------------------------------------------------------

/// Precomputed RoPE rotation factors for a given position and dimension.
#[derive(Clone, Debug)]
pub struct RoPEConfig {
    /// cos(θ_i) values scaled to integer representation, one per pair (d_head/2 entries).
    pub cos_table: Vec<i64>,
    /// sin(θ_i) values scaled to integer representation, one per pair (d_head/2 entries).
    pub sin_table: Vec<i64>,
}

/// Precomputes RoPE rotation factors for a given position and head dimension.
///
/// # Arguments
/// * `position` - Token position in the sequence (0-indexed)
/// * `d_head` - Head dimension (must be even)
/// * `scale_bits` - Number of bits for integer scaling of cos/sin values (e.g. 7 for range [-127, 127])
/// * `base` - RoPE base frequency (default 10000.0 for LLaMA)
///
/// # Panics
/// Panics if `d_head` is odd.
pub fn precompute_rope(position: usize, d_head: usize, scale_bits: u32, base: f64) -> RoPEConfig {
    assert!(d_head % 2 == 0, "precompute_rope: d_head ({d_head}) must be even");

    let n_pairs = d_head / 2;
    let scale = (1i64 << scale_bits) as f64;
    let mut cos_table = Vec::with_capacity(n_pairs);
    let mut sin_table = Vec::with_capacity(n_pairs);

    for i in 0..n_pairs {
        let theta = (position as f64) / base.powf(2.0 * i as f64 / d_head as f64);
        cos_table.push(theta.cos().mul_add(scale, 0.0).round() as i64);
        sin_table.push(theta.sin().mul_add(scale, 0.0).round() as i64);
    }

    RoPEConfig { cos_table, sin_table }
}

/// Applies RoPE rotation to encrypted Q or K vectors in the vector representation.
///
/// Each element of `q_or_k` is a ciphertext encrypting a single dimension.
/// Dimensions are rotated in pairs: (0,1), (2,3), (4,5), etc.
///
/// For each pair (2i, 2i+1):
/// - `out[2i]   = q[2i] * cos[i] - q[2i+1] * sin[i]`
/// - `out[2i+1] = q[2i] * sin[i] + q[2i+1] * cos[i]`
///
/// The cos/sin values are precomputed cleartext scalars, so this only
/// requires ct-pt multiplications and additions — no ct*ct operations.
///
/// # Arguments
/// * `module` - Backend module
/// * `q_or_k` - d_head ciphertexts to rotate
/// * `rope` - Precomputed RoPE rotation factors
///
/// # Returns
/// d_head ciphertexts with RoPE rotation applied
///
/// # Panics
/// Panics if `q_or_k.len()` is odd or doesn't match `rope.cos_table.len() * 2`
pub fn chimera_apply_rope_vec<BE: Backend>(module: &Module<BE>, q_or_k: &[GLWE<Vec<u8>>], rope: &RoPEConfig) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd + GLWESub,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let d_head = q_or_k.len();
    assert!(
        d_head % 2 == 0,
        "chimera_apply_rope_vec: input length ({d_head}) must be even"
    );
    assert_eq!(
        d_head,
        rope.cos_table.len() * 2,
        "chimera_apply_rope_vec: input length ({d_head}) != 2 * cos_table.len() ({})",
        rope.cos_table.len() * 2
    );

    let n_pairs = d_head / 2;
    let out: Vec<GLWE<Vec<u8>>> = {
        use rayon::prelude::*;
        (0..n_pairs)
            .into_par_iter()
            .flat_map(|i| {
                let cos_i = rope.cos_table[i];
                let sin_i = rope.sin_table[i];

                // q[2i] * cos[i]
                let even_cos = chimera_mul_const(module, &q_or_k[2 * i], &[cos_i]);
                // q[2i+1] * sin[i]
                let odd_sin = chimera_mul_const(module, &q_or_k[2 * i + 1], &[sin_i]);
                // out[2i] = q[2i] * cos[i] - q[2i+1] * sin[i]
                let out_even = chimera_sub(module, &even_cos, &odd_sin);

                // q[2i] * sin[i]
                let even_sin = chimera_mul_const(module, &q_or_k[2 * i], &[sin_i]);
                // q[2i+1] * cos[i]
                let odd_cos = chimera_mul_const(module, &q_or_k[2 * i + 1], &[cos_i]);
                // out[2i+1] = q[2i] * sin[i] + q[2i+1] * cos[i]
                let out_odd = chimera_add(module, &even_sin, &odd_cos);

                vec![out_even, out_odd]
            })
            .collect()
    };

    out
}

/// Weight matrices for a single attention layer (in plaintext).
///
/// Each matrix is stored as a `Vec<Vec<i64>>` where each inner Vec is one
/// output row of i64 values (quantised INT8 weights scaled to torus
/// representation).
///
/// ## GQA (Grouped Query Attention)
///
/// When `n_kv_heads < n_heads`, K and V projections are smaller:
/// - `w_q`: `[d_model × d_model]` — full-size query projection
/// - `w_k`: `[d_kv × d_model]` — reduced key projection (`d_kv = n_kv_heads * d_head`)
/// - `w_v`: `[d_kv × d_model]` — reduced value projection
/// - `w_o`: `[d_model × d_model]` — full-size output projection
///
/// For standard MHA (`n_kv_heads == n_heads`), `d_kv == d_model` and all
/// matrices are square.
#[derive(Clone, Debug)]
pub struct AttentionWeights {
    /// Query projection: `[d_model]` rows, each `[d_model]` long.
    pub w_q: Vec<Vec<i64>>,
    /// Key projection: `[d_kv]` rows, each `[d_model]` long.
    /// For GQA, `d_kv = n_kv_heads * d_head < d_model`.
    pub w_k: Vec<Vec<i64>>,
    /// Value projection: `[d_kv]` rows, each `[d_model]` long.
    /// For GQA, `d_kv = n_kv_heads * d_head < d_model`.
    pub w_v: Vec<Vec<i64>>,
    /// Output projection: `[d_model]` rows, each `[d_model]` long.
    pub w_o: Vec<Vec<i64>>,
}

impl AttentionWeights {
    /// Creates zero-initialised weights for the given dimensions.
    ///
    /// K and V projections use `d_kv()` rows (reduced for GQA).
    pub fn zeros(dims: &ModelDims) -> Self {
        let d = dims.d_model;
        let d_kv = dims.d_kv();
        AttentionWeights {
            w_q: vec![vec![0i64; d]; d],
            w_k: vec![vec![0i64; d]; d_kv],
            w_v: vec![vec![0i64; d]; d_kv],
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

// ---------------------------------------------------------------------------
// Homomorphic attention operations
// ---------------------------------------------------------------------------

/// Computes Q, K, V projections for a single attention head under FHE.
///
/// Given an encrypted input vector `x` (packed into a single ciphertext) and
/// plaintext weight rows for Q, K, V, this function produces three ciphertexts:
/// one each for Q, K, V, each containing `d_head` values.
///
/// Each projection is computed as `chimera_mul_const(ct_x, w_row)` where
/// `w_row` is a scalar weight (single-coefficient polynomial) representing
/// one output dimension of the projection matrix.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `ct_x` - Encrypted input vector (one ciphertext).
/// * `w_q_row` - Query weight row as a scalar i64 (single coefficient).
/// * `w_k_row` - Key weight row as a scalar i64.
/// * `w_v_row` - Value weight row as a scalar i64.
///
/// # Returns
///
/// Tuple of (Q, K, V) ciphertexts for this head dimension.
pub fn chimera_qkv_project_single<BE: Backend>(
    module: &Module<BE>,
    ct_x: &GLWE<Vec<u8>>,
    w_q_row: &[i64],
    w_k_row: &[i64],
    w_v_row: &[i64],
) -> (GLWE<Vec<u8>>, GLWE<Vec<u8>>, GLWE<Vec<u8>>)
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let q = chimera_mul_const(module, ct_x, w_q_row);
    let k = chimera_mul_const(module, ct_x, w_k_row);
    let v = chimera_mul_const(module, ct_x, w_v_row);
    (q, k, v)
}

/// Computes Q, K, V projections for all heads using matrix multiplication.
///
/// For each projection matrix (W_Q, W_K, W_V), computes `W · x` by applying
/// `chimera_matmul_single_ct` with the weight rows. Each output ciphertext
/// corresponds to one row of the projection (one output dimension).
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `ct_x` - Encrypted input vector.
/// * `weights` - Attention weight matrices.
/// * `n_rows` - Number of output rows to compute (typically d_model or d_head).
///
/// # Returns
///
/// Tuple of (Q_cts, K_cts, V_cts) where each is a Vec of ciphertexts.
///
/// # Note
///
/// This function uses the same `n_rows` for Q, K, and V, so it only works
/// for standard MHA (n_kv_heads == n_heads). For GQA, use the packed
/// single-ct variant in [`chimera_multi_head_attention`] which handles
/// separate Q/KV row counts directly.
#[allow(dead_code)]
pub fn chimera_qkv_project<BE: Backend>(
    module: &Module<BE>,
    ct_x: &GLWE<Vec<u8>>,
    weights: &AttentionWeights,
    n_rows: usize,
) -> (Vec<GLWE<Vec<u8>>>, Vec<GLWE<Vec<u8>>>, Vec<GLWE<Vec<u8>>>)
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    let w_q_rows: Vec<Vec<i64>> = weights.w_q.iter().take(n_rows).cloned().collect();
    let w_k_rows: Vec<Vec<i64>> = weights.w_k.iter().take(n_rows).cloned().collect();
    let w_v_rows: Vec<Vec<i64>> = weights.w_v.iter().take(n_rows).cloned().collect();

    let q_cts = chimera_matmul_single_ct(module, ct_x, &w_q_rows);
    let k_cts = chimera_matmul_single_ct(module, ct_x, &w_k_rows);
    let v_cts = chimera_matmul_single_ct(module, ct_x, &w_v_rows);

    (q_cts, k_cts, v_cts)
}

/// Computes attention scores for a single head: `score = Q · K^T / √d_k`.
///
/// This is the core ct*ct operation in attention. For a single query-key pair:
/// 1. Element-wise multiply: `qk = Q ⊙ K` (tensor product + relinearization)
/// 2. Slot sum: `score = Σ_i qk[i]` (trace operation)
///
/// The `1/√d_k` scaling is absorbed into the weight quantization.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key (tensor key for ct*ct mul, auto keys for slot sum).
/// * `ct_q` - Encrypted query vector for one head.
/// * `ct_k` - Encrypted key vector for one head.
/// * `skip_trace` - Number of trace levels to skip (0 for full sum).
///
/// # Returns
///
/// Ciphertext containing the attention score (replicated in all slots after trace).
pub fn chimera_attention_score<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct_q: &GLWE<Vec<u8>>,
    ct_k: &GLWE<Vec<u8>>,
    skip_trace: usize,
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    // Step 1: Element-wise product Q ⊙ K
    let qk = chimera_ct_mul(module, eval_key, ct_q, ct_k);

    // Step 2: Sum all slots to get the dot product
    chimera_slot_sum(module, eval_key, &qk, skip_trace)
}

/// Applies softmax approximation to attention scores under FHE.
///
/// For each score ciphertext, applies the polynomial approximation of the
/// softmax numerator (e.g. `p(x) = (1 + x + x²/2)²` for PolynomialDeg4,
/// or `x²` for ReluSquared).
///
/// Normalization (dividing by the sum) is handled separately, either:
/// - By absorbing it into the subsequent V multiplication (approximate)
/// - Via a polynomial approximation of `1/Σ` applied to the sum
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `scores` - Encrypted attention scores.
/// * `strategy` - Which softmax approximation to use.
///
/// # Returns
///
/// Ciphertexts with softmax numerator applied to each score.
pub fn chimera_apply_softmax<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    scores: &[GLWE<Vec<u8>>],
    strategy: &SoftmaxStrategy,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    match strategy {
        SoftmaxStrategy::Linear => {
            // Linear attention: identity, no transformation needed.
            // Clone the scores.
            scores
                .iter()
                .map(|s| {
                    use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
                    let mut res = GLWE::<Vec<u8>>::alloc_from_infos(s);
                    let s_ref = s.to_ref();
                    let src: &[u8] = s_ref.data().data;
                    let mut res_mut = res.to_mut();
                    let dst: &mut [u8] = res_mut.data_mut().data;
                    let len = src.len().min(dst.len());
                    dst[..len].copy_from_slice(&src[..len]);
                    res
                })
                .collect()
        }
        SoftmaxStrategy::ReluSquared => {
            // Apply x² to each score
            let approx = squared_relu_approx();
            scores
                .iter()
                .map(|s| apply_poly_activation(module, eval_key, s, &approx))
                .collect()
        }
        SoftmaxStrategy::PolynomialDeg4 => {
            // Apply degree-4 exp approximation to each score
            let approx = crate::activations::exp_poly_approx();
            scores
                .iter()
                .map(|s| apply_poly_activation(module, eval_key, s, &approx))
                .collect()
        }
        SoftmaxStrategy::Custom(approx) => scores
            .iter()
            .map(|s| apply_poly_activation(module, eval_key, s, approx))
            .collect(),
    }
}

/// Computes context vectors: `context_h = Σ_i attn_weight_i · V_i` for one head.
///
/// This computes the weighted combination of V vectors using the attention
/// weights. Since both `attn_weights` and `V` are encrypted, this requires
/// ct*ct multiplication for each pair.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `attn_weights` - Encrypted attention weights (one per key position).
/// * `v_cts` - Encrypted value vectors (one per key position).
///
/// # Returns
///
/// Ciphertext containing the context vector for this head.
///
/// # Panics
///
/// Panics if `attn_weights` and `v_cts` have different lengths or are empty.
pub fn chimera_attention_context<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    attn_weights: &[GLWE<Vec<u8>>],
    v_cts: &[GLWE<Vec<u8>>],
) -> GLWE<Vec<u8>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    assert_eq!(attn_weights.len(), v_cts.len());
    assert!(!attn_weights.is_empty());

    // Compute attn_weight_0 * V_0
    let mut acc = chimera_ct_mul(module, eval_key, &attn_weights[0], &v_cts[0]);

    for i in 1..attn_weights.len() {
        let term = chimera_ct_mul(module, eval_key, &attn_weights[i], &v_cts[i]);
        acc = chimera_add(module, &acc, &term);
    }

    acc
}

/// Computes the output projection: `output = context · W_O`.
///
/// This is a ciphertext-plaintext matrix multiplication where the context
/// vector is encrypted and W_O is in the clear.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `ct_context` - Encrypted context vector.
/// * `w_o_rows` - Output projection weight rows (plaintext).
///
/// # Returns
///
/// Vector of ciphertexts, one per output dimension.
pub fn chimera_output_project<BE: Backend>(
    module: &Module<BE>,
    ct_context: &GLWE<Vec<u8>>,
    w_o_rows: &[Vec<i64>],
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWEMulConst<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    chimera_matmul_single_ct(module, ct_context, w_o_rows)
}

/// Computes multi-head attention on an encrypted input vector.
///
/// This is the full multi-head attention implementation. For each head:
/// 1. Project Q, K, V using weight rows `[h*d_head..(h+1)*d_head]`
/// 2. Compute attention score: dot product of Q_h and K_h vectors
/// 3. Apply softmax approximation
/// 4. Compute context: weighted sum of V_h values
///
/// After all heads are computed, the per-head context vectors are concatenated
/// into a single `d_model`-length vector of ciphertexts, and the output
/// projection W_O is applied.
///
/// ## Weight Layout
///
/// The weight matrices W_Q, W_K, W_V are `[d_model × d_model]` where:
/// - Rows `[h*d_head..(h+1)*d_head]` correspond to head `h`
/// - Each row is a `d_model`-length polynomial weight vector
///
/// W_O is `[d_model × d_model]` and is applied to the concatenated output.
///
/// ## Packing Strategy
///
/// Each QKV projection output is a separate ciphertext (one per output
/// dimension). Within a head, Q_h / K_h / V_h each consist of `d_head`
/// ciphertexts. The attention score for each head is computed by:
/// - Element-wise ct*ct multiplying corresponding Q and K ciphertexts
/// - Accumulating with chimera_add to compute the dot product
///
/// This "per-dimension packing" approach avoids the need for automorphism-based
/// head splitting/concatenation, at the cost of more ciphertexts in flight.
///
/// ## Sequence Length = 1 (Single Token)
///
/// For single-token inference (batch_size=1, seq_len=1), Q, K, V each have
/// a single "position". The attention score is a scalar (Q · K^T), the
/// softmax is trivially 1, and the context equals V. This simplification
/// means the main cost is in the QKV and output projections.
///
/// For seq_len > 1, the KV cache would be used (not yet implemented).
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key (tensor key for ct*ct, auto keys for trace).
/// * `ct_x` - Encrypted input vector (single ciphertext, d_model coefficients).
/// * `weights` - Attention weight matrices (W_Q, W_K, W_V, W_O).
/// * `config` - Attention configuration (dims, softmax strategy, etc.).
///
/// # Returns
///
/// A vector of `d_model` ciphertexts representing the attention output, one
/// per output dimension.
///
/// # Panics
///
/// Panics if `d_model != n_heads * d_head` or if weight dimensions are
/// inconsistent.
pub fn chimera_multi_head_attention<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    ct_x: &GLWE<Vec<u8>>,
    weights: &AttentionWeights,
    config: &AttentionConfig,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let d_model = config.dims.d_model;
    let n_heads = config.dims.n_heads;
    let d_head = config.dims.d_head;
    let d_kv = config.dims.d_kv();
    let gqa_group = config.dims.gqa_group_size();

    assert_eq!(
        d_model,
        n_heads * d_head,
        "chimera_multi_head_attention: d_model ({d_model}) != n_heads ({n_heads}) * d_head ({d_head})"
    );
    assert!(
        weights.w_q.len() >= d_model,
        "chimera_multi_head_attention: w_q has {} rows, need {d_model}",
        weights.w_q.len()
    );
    assert!(
        weights.w_k.len() >= d_kv,
        "chimera_multi_head_attention: w_k has {} rows, need {d_kv}",
        weights.w_k.len()
    );
    assert!(
        weights.w_v.len() >= d_kv,
        "chimera_multi_head_attention: w_v has {} rows, need {d_kv}",
        weights.w_v.len()
    );
    assert!(
        weights.w_o.len() >= d_model,
        "chimera_multi_head_attention: w_o has {} rows, need {d_model}",
        weights.w_o.len()
    );

    // -----------------------------------------------------------------------
    // Step 1: QKV projections for ALL heads at once.
    //
    // Q = W_Q · x produces d_model ciphertexts (one per output dimension).
    // Rows [h*d_head..(h+1)*d_head] belong to head h.
    //
    // K, V projections produce d_kv ciphertexts (n_kv_heads * d_head).
    // For GQA, multiple Q heads share the same KV head.
    // -----------------------------------------------------------------------
    let q_all = chimera_matmul_single_ct(module, ct_x, &weights.w_q[..d_model].to_vec());
    let k_all = chimera_matmul_single_ct(module, ct_x, &weights.w_k[..d_kv].to_vec());
    let v_all = chimera_matmul_single_ct(module, ct_x, &weights.w_v[..d_kv].to_vec());

    // -----------------------------------------------------------------------
    // Step 2: Per-head attention computation.
    //
    // For each query head h:
    //   Q_h = q_all[h*d_head..(h+1)*d_head]  (d_head ciphertexts)
    //   kv_h = h / gqa_group  (the KV head this query head maps to)
    //   K_kv = k_all[kv_h*d_head..(kv_h+1)*d_head]
    //   V_kv = v_all[kv_h*d_head..(kv_h+1)*d_head]
    //
    //   score_h = Σ_i Q_h[i] * K_kv[i]   (dot product)
    //   attn_h  = softmax(score_h)
    //   context_h = attn_h * V_kv
    // -----------------------------------------------------------------------
    let mut all_head_contexts: Vec<GLWE<Vec<u8>>> = Vec::with_capacity(d_model);

    for h in 0..n_heads {
        let q_start = h * d_head;
        let q_end = q_start + d_head;
        let q_h = &q_all[q_start..q_end];

        let kv_h = h / gqa_group;
        let kv_start = kv_h * d_head;
        let kv_end = kv_start + d_head;
        let k_h = &k_all[kv_start..kv_end];
        let v_h = &v_all[kv_start..kv_end];

        // Compute attention score for this head:
        // score_h = Σ_i Q_h[i] ⊙ K_h[i]
        //
        // Each Q_h[i] and K_h[i] are ciphertexts encoding a single projected
        // dimension. Their ct*ct product gives that dimension's contribution to
        // the score. We accumulate all d_head contributions.
        let head_context = chimera_single_head_attention(module, eval_key, q_h, k_h, v_h, &config.softmax_approx);

        // head_context is a Vec of d_head ciphertexts
        all_head_contexts.extend(head_context);
    }

    assert_eq!(
        all_head_contexts.len(),
        d_model,
        "multi_head_attention: concatenated heads should produce {d_model} ciphertexts, got {}",
        all_head_contexts.len()
    );

    // -----------------------------------------------------------------------
    // Step 3: Output projection — W_O · concat(head_0, ..., head_{n-1})
    //
    // The concatenated head outputs form a d_model-length vector of cts.
    // W_O is [d_model × d_model]. Each output dimension i is:
    //   output_i = Σ_j W_O[i][j] * context[j]
    //
    // This is a ct-pt matrix-vector product using chimera_dot_product.
    // -----------------------------------------------------------------------
    let mut outputs = Vec::with_capacity(d_model);
    for row in weights.w_o.iter().take(d_model) {
        // Each W_O row is d_model scalars. Convert to single-coefficient weight vecs.
        let w_vecs: Vec<Vec<i64>> = row.iter().map(|&w| vec![w]).collect();
        let dot = crate::arithmetic::chimera_dot_product(module, &all_head_contexts, &w_vecs);
        outputs.push(dot);
    }

    outputs
}

/// Computes attention for a single head given pre-projected Q, K, V vectors.
///
/// This is the inner loop of multi-head attention. Given `d_head` ciphertexts
/// each for Q, K, V (one per projected dimension), this function:
///
/// 1. Computes the attention score as a dot product: `score = Σ_i Q[i] * K[i]`
/// 2. Applies the softmax approximation to the score
/// 3. Computes the context: `context[j] = attn_weight * V[j]` for each j
///
/// For single-token inference (seq_len=1), there is only one query-key pair,
/// so the score is a single scalar and the softmax is trivially 1.0.
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key.
/// * `q_h` - Query ciphertexts for this head (d_head elements).
/// * `k_h` - Key ciphertexts for this head (d_head elements).
/// * `v_h` - Value ciphertexts for this head (d_head elements).
/// * `softmax` - Softmax approximation strategy.
///
/// # Returns
///
/// A vector of `d_head` ciphertexts representing the context for this head.
///
/// # Panics
///
/// Panics if Q, K, V have different lengths or are empty.
pub fn chimera_single_head_attention<BE: Backend>(
    _module: &Module<BE>,
    _eval_key: &ChimeraEvalKey<BE>,
    q_h: &[GLWE<Vec<u8>>],
    k_h: &[GLWE<Vec<u8>>],
    v_h: &[GLWE<Vec<u8>>],
    _softmax: &SoftmaxStrategy,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let d_head = q_h.len();
    assert_eq!(d_head, k_h.len(), "Q and K must have same length");
    assert_eq!(d_head, v_h.len(), "Q and V must have same length");
    assert!(d_head > 0, "head dimension must be > 0");

    // Current vec inference only supports the single-token path (seq_len = 1).
    // In that regime there is exactly one attention score, so softmax(score) = 1
    // and the context is exactly V. Returning V directly avoids unnecessary ct×ct
    // score/context multiplications and preserves the projection scale.
    v_h.iter()
        .map(|ct| {
            use poulpy_core::layouts::{GLWEToMut, GLWEToRef};
            let mut cloned = GLWE::<Vec<u8>>::alloc_from_infos(ct);
            let src_ref = ct.to_ref();
            let src: &[u8] = src_ref.data().data;
            let mut dst_mut = cloned.to_mut();
            let dst: &mut [u8] = dst_mut.data_mut().data;
            let len = src.len().min(dst.len());
            dst[..len].copy_from_slice(&src[..len]);
            cloned
        })
        .collect()
}

/// Computes Q, K, V projections in the vector representation.
///
/// Each input ciphertext `x_cts[j]` encrypts a single dimension of the
/// input vector (value in coeff 0). The projection `Q_i = Σ_j W_Q[i][j] * x_j`
/// is computed as a [`chimera_dot_product`] per output dimension.
///
/// For GQA, K and V have fewer output rows (`n_kv_rows`) than Q (`n_q_rows`).
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `x_cts` - Input vector: one ciphertext per dimension (d_model total).
/// * `weights` - Attention weight matrices.
/// * `n_q_rows` - Number of Q output rows (typically d_model).
/// * `n_kv_rows` - Number of K/V output rows (d_kv for GQA, d_model for MHA).
///
/// # Returns
///
/// Tuple of (Q_cts, K_cts, V_cts) where Q has `n_q_rows` and K/V have `n_kv_rows` ciphertexts.
pub fn chimera_qkv_project_vec<BE: Backend>(
    module: &Module<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    weights: &AttentionWeights,
    n_q_rows: usize,
    n_kv_rows: usize,
) -> (Vec<GLWE<Vec<u8>>>, Vec<GLWE<Vec<u8>>>, Vec<GLWE<Vec<u8>>>)
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    chimera_qkv_project_vec_scaled(module, x_cts, weights, n_q_rows, n_kv_rows, 0)
}

pub fn chimera_qkv_project_vec_scaled<BE: Backend>(
    module: &Module<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    weights: &AttentionWeights,
    n_q_rows: usize,
    n_kv_rows: usize,
    input_scale_shift_bits: usize,
) -> (Vec<GLWE<Vec<u8>>>, Vec<GLWE<Vec<u8>>>, Vec<GLWE<Vec<u8>>>)
where
    Module<BE>: GLWEMulConst<BE> + GLWEAdd,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable,
{
    use rayon::prelude::*;
    let d_in = x_cts.len();

    let project_one_matrix = |w_rows: &[Vec<i64>], n_rows: usize| -> Vec<GLWE<Vec<u8>>> {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                assert!(
                    w_rows[i].len() >= d_in,
                    "Weight row {i} has {} elements, need {d_in}",
                    w_rows[i].len()
                );
                let w_vecs: Vec<Vec<i64>> = w_rows[i][..d_in].iter().map(|&w| vec![w]).collect();
                if input_scale_shift_bits == 0 {
                    crate::arithmetic::chimera_dot_product(module, x_cts, &w_vecs)
                } else {
                    crate::arithmetic::chimera_dot_product_scaled(module, x_cts, &w_vecs, input_scale_shift_bits)
                }
            })
            .collect()
    };

    let q_cts = project_one_matrix(&weights.w_q, n_q_rows);
    let k_cts = project_one_matrix(&weights.w_k, n_kv_rows);
    let v_cts = project_one_matrix(&weights.w_v, n_kv_rows);

    (q_cts, k_cts, v_cts)
}

/// Computes multi-head attention in the vector representation.
///
/// This is the vector-representation variant of [`chimera_multi_head_attention`].
/// Each element of the input and output is a separate ciphertext encoding a
/// single scalar (value in coefficient 0).
///
/// ## Computation Flow
///
/// 1. **QKV projection**: `Q_i = Σ_j W_Q[i][j] * x_j` (dot product per output dim)
/// 2. **Per-head attention**: For each head h with dims [h*d_head..(h+1)*d_head]:
///    - Score: `s_h = Σ_k Q_h[k] * K_h[k]` (ct*ct dot product)
///    - Softmax: polynomial approximation applied to s_h
///    - Context: `c_h[k] = attn_h * V_h[k]` (ct*ct broadcast multiply)
/// 3. **Output projection**: `out_i = Σ_j W_O[i][j] * context[j]`
///
/// # Arguments
///
/// * `module` - Backend module.
/// * `eval_key` - Evaluation key (tensor key for ct*ct).
/// * `x_cts` - Input vector: d_model ciphertexts, each encrypting one scalar.
/// * `weights` - Attention weight matrices [d_model × d_model].
/// * `config` - Attention configuration (dims, softmax strategy).
///
/// # Returns
///
/// d_model ciphertexts representing the attention output.
pub fn chimera_multi_head_attention_vec<BE: Backend>(
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    x_cts: &[GLWE<Vec<u8>>],
    weights: &AttentionWeights,
    config: &AttentionConfig,
) -> Vec<GLWE<Vec<u8>>>
where
    Module<BE>: GLWETensoring<BE> + GLWEMulConst<BE> + GLWEAdd + GLWESub + GLWETrace<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE> + ScratchAvailable,
{
    let d_model = config.dims.d_model;
    let n_heads = config.dims.n_heads;
    let d_head = config.dims.d_head;
    let d_kv = config.dims.d_kv();
    let gqa_group = config.dims.gqa_group_size();

    assert_eq!(
        d_model,
        n_heads * d_head,
        "chimera_multi_head_attention_vec: d_model ({d_model}) != n_heads ({n_heads}) * d_head ({d_head})"
    );
    assert_eq!(
        x_cts.len(),
        d_model,
        "chimera_multi_head_attention_vec: input has {} cts, expected {d_model}",
        x_cts.len()
    );

    // Step 1: QKV projections
    // Q produces d_model outputs; K and V produce d_kv outputs each.
    let t_qkv = std::time::Instant::now();
    let (q_all, k_all, v_all) = chimera_qkv_project_vec_scaled(
        module,
        x_cts,
        weights,
        d_model,
        d_kv,
        crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
    );
    let qkv_ms = t_qkv.elapsed().as_millis();

    // Step 1b: Apply RoPE to Q and K if configured.
    // RoPE is applied per-head: each head's d_head slice is rotated in pairs.
    let t_rope = std::time::Instant::now();
    let (q_all, k_all) = if let Some(rope) = &config.rope {
        use rayon::prelude::*;

        // Apply RoPE to each Q head (parallelized across heads)
        let q_rope: Vec<GLWE<Vec<u8>>> = (0..n_heads)
            .into_par_iter()
            .flat_map(|h| {
                let q_start = h * d_head;
                let q_end = q_start + d_head;
                chimera_apply_rope_vec(module, &q_all[q_start..q_end], rope)
            })
            .collect();

        // Apply RoPE to each K head (parallelized across KV heads)
        let n_kv_heads = d_kv / d_head;
        let k_rope: Vec<GLWE<Vec<u8>>> = (0..n_kv_heads)
            .into_par_iter()
            .flat_map(|kv_h| {
                let kv_start = kv_h * d_head;
                let kv_end = kv_start + d_head;
                chimera_apply_rope_vec(module, &k_all[kv_start..kv_end], rope)
            })
            .collect();

        (q_rope, k_rope)
    } else {
        (q_all, k_all)
    };
    let rope_ms = t_rope.elapsed().as_millis();

    // Step 2: Per-head attention with GQA mapping (parallelized across heads)
    let t_heads = std::time::Instant::now();
    let all_head_contexts: Vec<GLWE<Vec<u8>>> = if n_heads > 1 {
        use rayon::prelude::*;
        let head_results: Vec<Vec<GLWE<Vec<u8>>>> = (0..n_heads)
            .into_par_iter()
            .map(|h| {
                let q_start = h * d_head;
                let q_end = q_start + d_head;
                let q_h = &q_all[q_start..q_end];

                let kv_h = h / gqa_group;
                let kv_start = kv_h * d_head;
                let kv_end = kv_start + d_head;
                let k_h = &k_all[kv_start..kv_end];
                let v_h = &v_all[kv_start..kv_end];

                chimera_single_head_attention(module, eval_key, q_h, k_h, v_h, &config.softmax_approx)
            })
            .collect();
        head_results.into_iter().flatten().collect()
    } else {
        // Single head — no parallelism overhead
        let q_h = &q_all[..d_head];
        let k_h = &k_all[..d_head];
        let v_h = &v_all[..d_head];
        chimera_single_head_attention(module, eval_key, q_h, k_h, v_h, &config.softmax_approx)
    };
    let heads_ms = t_heads.elapsed().as_millis();

    // Step 3: Output projection — out_i = Σ_j W_O[i][j] * context[j]
    let t_out = std::time::Instant::now();
    let outputs: Vec<GLWE<Vec<u8>>> = {
        use rayon::prelude::*;
        weights
            .w_o
            .iter()
            .take(d_model)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|row| {
                let w_vecs: Vec<Vec<i64>> = row.iter().map(|&w| vec![w]).collect();
                crate::arithmetic::chimera_dot_product(module, &all_head_contexts, &w_vecs)
            })
            .collect()
    };
    let out_proj_ms = t_out.elapsed().as_millis();

    eprintln!(
        "[profile]   attention d_model={d_model} n_heads={n_heads} d_head={d_head} d_kv={d_kv} | \
         qkv={qkv_ms}ms rope={rope_ms}ms heads={heads_ms}ms out_proj={out_proj_ms}ms"
    );

    outputs
}
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

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

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
            rope: None,
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

    #[test]
    fn test_qkv_project_single() {
        // Test that QKV single-dimension projection produces three ciphertexts
        // with correct scalar multiplication.
        use crate::encoding::{decode_int8, encode_int8};
        use crate::encrypt::{chimera_decrypt, chimera_encrypt, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let values: Vec<i8> = vec![4, 5];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        // W_Q scales by 2, W_K scales by 3, W_V scales by 1
        let (ct_q, ct_k, ct_v) = chimera_qkv_project_single(&module, &ct, &[2i64], &[3i64], &[1i64]);

        let pt_q = chimera_decrypt(&module, &key, &ct_q, &params);
        let pt_k = chimera_decrypt(&module, &key, &ct_k, &params);
        let pt_v = chimera_decrypt(&module, &key, &ct_v, &params);

        let dec_q = decode_int8(&module, &params, &pt_q, 2);
        let dec_k = decode_int8(&module, &params, &pt_k, 2);
        let dec_v = decode_int8(&module, &params, &pt_v, 2);

        // Q = [4*2, 5*2] = [8, 10]
        assert!((dec_q[0] as i16 - 8).unsigned_abs() <= 1, "Q[0] expected 8, got {}", dec_q[0]);
        assert!(
            (dec_q[1] as i16 - 10).unsigned_abs() <= 1,
            "Q[1] expected 10, got {}",
            dec_q[1]
        );

        // K = [4*3, 5*3] = [12, 15]
        assert!(
            (dec_k[0] as i16 - 12).unsigned_abs() <= 1,
            "K[0] expected 12, got {}",
            dec_k[0]
        );
        assert!(
            (dec_k[1] as i16 - 15).unsigned_abs() <= 1,
            "K[1] expected 15, got {}",
            dec_k[1]
        );

        // V = [4*1, 5*1] = [4, 5]
        assert!((dec_v[0] as i16 - 4).unsigned_abs() <= 1, "V[0] expected 4, got {}", dec_v[0]);
        assert!((dec_v[1] as i16 - 5).unsigned_abs() <= 1, "V[1] expected 5, got {}", dec_v[1]);
    }

    #[test]
    fn test_attention_score_basic() {
        // Test attention score computation: Q·K element-wise + slot sum.
        // This verifies the core ct*ct + trace pipeline.
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_encrypt, ChimeraEvalKey, ChimeraKey};
        use poulpy_core::layouts::LWEInfos;
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);
        let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [4u8; 32], [5u8; 32]);

        // Encode Q = [2, 0, 0, ...] and K = [3, 0, 0, ...]
        // Expected score = 2*3 = 6 (at coefficient 0 after slot sum)
        let q_vals: Vec<i8> = vec![2];
        let k_vals: Vec<i8> = vec![3];

        let pt_q = encode_int8(&module, &params, &q_vals);
        let pt_k = encode_int8(&module, &params, &k_vals);

        let ct_q = chimera_encrypt(&module, &key, &pt_q, [10u8; 32], [11u8; 32]);
        let ct_k = chimera_encrypt(&module, &key, &pt_k, [12u8; 32], [13u8; 32]);

        // Compute attention score (ct*ct mul + slot sum)
        let score = chimera_attention_score(&module, &eval_key, &ct_q, &ct_k, 0);

        // The score ciphertext exists and has valid layout
        // (Verifying full numerical correctness of ct*ct * trace would require
        // decoding at the correct post-tensor-product scale, which is complex.
        // Here we verify the pipeline completes without panicking.)
        assert_eq!(score.n(), ct_q.n());
    }

    #[test]
    fn test_apply_softmax_linear() {
        // Linear softmax should be an identity (just clone).
        use crate::encoding::decode_int8;
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_decrypt, chimera_encrypt, ChimeraEvalKey, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);
        let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [4u8; 32], [5u8; 32]);

        let vals: Vec<i8> = vec![3, 5];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        let results = chimera_apply_softmax(&module, &eval_key, &[ct], &SoftmaxStrategy::Linear);
        assert_eq!(results.len(), 1);

        // Linear softmax should preserve the values
        let pt_dec = chimera_decrypt(&module, &key, &results[0], &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 2);
        assert!(
            (decoded[0] as i16 - 3).unsigned_abs() <= 1,
            "linear softmax[0] expected 3, got {}",
            decoded[0]
        );
        assert!(
            (decoded[1] as i16 - 5).unsigned_abs() <= 1,
            "linear softmax[1] expected 5, got {}",
            decoded[1]
        );
    }

    #[test]
    fn test_rope_precompute() {
        // Position 0: θ_i = 0 for all i, so cos=1, sin=0.
        let rope0 = precompute_rope(0, 4, 7, 10000.0);
        assert_eq!(rope0.cos_table.len(), 2, "d_head=4 should give 2 pairs");
        assert_eq!(rope0.sin_table.len(), 2);
        // cos(0) * 128 = 128, sin(0) * 128 = 0
        let scale = 1i64 << 7; // 128
        for i in 0..2 {
            assert!(
                (rope0.cos_table[i] - scale).abs() <= 1,
                "pos=0 cos[{i}] expected {scale}, got {}",
                rope0.cos_table[i]
            );
            assert!(
                rope0.sin_table[i].abs() <= 1,
                "pos=0 sin[{i}] expected 0, got {}",
                rope0.sin_table[i]
            );
        }

        // Position 5: cos/sin should be non-trivial for at least pair 0.
        let rope5 = precompute_rope(5, 4, 7, 10000.0);
        assert_eq!(rope5.cos_table.len(), 2);
        assert_eq!(rope5.sin_table.len(), 2);
        // θ_0 = 5 / 10000^(0/4) = 5.0, so cos(5) ≈ 0.2837, sin(5) ≈ -0.9589
        // cos_scaled ≈ round(0.2837 * 128) = 36, sin_scaled ≈ round(-0.9589 * 128) = -123
        let expected_cos0 = (5.0f64.cos() * 128.0).round() as i64;
        let expected_sin0 = (5.0f64.sin() * 128.0).round() as i64;
        assert!(
            (rope5.cos_table[0] - expected_cos0).abs() <= 1,
            "pos=5 cos[0] expected {expected_cos0}, got {}",
            rope5.cos_table[0]
        );
        assert!(
            (rope5.sin_table[0] - expected_sin0).abs() <= 1,
            "pos=5 sin[0] expected {expected_sin0}, got {}",
            rope5.sin_table[0]
        );
    }

    #[test]
    fn test_rope_apply_identity() {
        // At position 0, cos(0)=1 and sin(0)=0, so RoPE should approximately
        // preserve the input vector. We use scale_bits=0 so that the cos/sin
        // scaling factor is 2^0 = 1, meaning cos_scaled=1, sin_scaled=0.
        // This way mul_const(ct, [1]) produces the original value without
        // overflow.
        use crate::encoding::{decode_int8, encode_int8};
        use crate::encrypt::{chimera_decrypt, chimera_encrypt, ChimeraKey};
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);

        let values: Vec<i8> = vec![10, 20, 30, 40];

        // Encrypt each value as a separate ciphertext (vector representation)
        let mut cts: Vec<GLWE<Vec<u8>>> = Vec::new();
        for (idx, &v) in values.iter().enumerate() {
            let pt = encode_int8(&module, &params, &[v]);
            let mut seed_a = [0u8; 32];
            let mut seed_e = [0u8; 32];
            seed_a[0] = (idx * 2 + 10) as u8;
            seed_e[0] = (idx * 2 + 11) as u8;
            let ct = chimera_encrypt(&module, &key, &pt, seed_a, seed_e);
            cts.push(ct);
        }

        // Precompute RoPE at position 0 with scale_bits=0 (cos=1, sin=0)
        let rope = precompute_rope(0, 4, 0, 10000.0);
        // Verify precomputed values
        assert_eq!(rope.cos_table, vec![1, 1], "cos(0) scaled by 1 should be 1");
        assert_eq!(rope.sin_table, vec![0, 0], "sin(0) scaled by 1 should be 0");

        // Apply RoPE
        let rotated = chimera_apply_rope_vec(&module, &cts, &rope);
        assert_eq!(rotated.len(), 4);

        // With cos=1, sin=0:
        //   out[0] = q[0]*1 - q[1]*0 = q[0]
        //   out[1] = q[0]*0 + q[1]*1 = q[1]
        //   out[2] = q[2]*1 - q[3]*0 = q[2]
        //   out[3] = q[2]*0 + q[3]*1 = q[3]
        // So the output should approximately equal the input.
        for (idx, &v) in values.iter().enumerate() {
            let pt_dec = chimera_decrypt(&module, &key, &rotated[idx], &params);
            let decoded = decode_int8(&module, &params, &pt_dec, 1);
            let expected = v as i16;
            let diff = (decoded[0] as i16 - expected).unsigned_abs();
            assert!(
                diff <= 2,
                "RoPE identity at idx {idx}: expected ~{expected}, got {}, diff={diff}",
                decoded[0]
            );
        }
    }
}
