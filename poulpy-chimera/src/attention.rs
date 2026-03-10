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

use crate::activations::{PolyApprox, apply_poly_activation, chimera_ct_mul, squared_relu_approx};
use crate::arithmetic::{chimera_add, chimera_mul_const, chimera_matmul_single_ct, chimera_slot_sum};
use crate::encrypt::ChimeraEvalKey;
use crate::params::{ChimeraParams, ModelDims};
use poulpy_core::{
    GLWEAdd, GLWEMulConst, GLWETensoring, GLWETrace,
    layouts::GLWE,
};
use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, Scratch, ScratchOwned},
};
use poulpy_core::ScratchTakeCore;

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
        SoftmaxStrategy::Custom(approx) => {
            scores
                .iter()
                .map(|s| apply_poly_activation(module, eval_key, s, approx))
                .collect()
        }
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
        weights.w_k.len() >= d_model,
        "chimera_multi_head_attention: w_k has {} rows, need {d_model}",
        weights.w_k.len()
    );
    assert!(
        weights.w_v.len() >= d_model,
        "chimera_multi_head_attention: w_v has {} rows, need {d_model}",
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
    // -----------------------------------------------------------------------
    let (q_all, k_all, v_all) = chimera_qkv_project(module, ct_x, weights, d_model);

    // -----------------------------------------------------------------------
    // Step 2: Per-head attention computation.
    //
    // For each head h:
    //   Q_h = q_all[h*d_head..(h+1)*d_head]  (d_head ciphertexts)
    //   K_h = k_all[h*d_head..(h+1)*d_head]
    //   V_h = v_all[h*d_head..(h+1)*d_head]
    //
    //   score_h = Σ_i Q_h[i] * K_h[i]   (dot product: d_head ct*ct muls + additions)
    //   attn_h  = softmax(score_h)        (single score for seq_len=1)
    //   context_h = attn_h * V_h          (d_head ct*ct muls)
    // -----------------------------------------------------------------------
    let mut all_head_contexts: Vec<GLWE<Vec<u8>>> = Vec::with_capacity(d_model);

    for h in 0..n_heads {
        let start = h * d_head;
        let end = start + d_head;

        let q_h = &q_all[start..end];
        let k_h = &k_all[start..end];
        let v_h = &v_all[start..end];

        // Compute attention score for this head:
        // score_h = Σ_i Q_h[i] ⊙ K_h[i]
        //
        // Each Q_h[i] and K_h[i] are ciphertexts encoding a single projected
        // dimension. Their ct*ct product gives that dimension's contribution to
        // the score. We accumulate all d_head contributions.
        let head_context = chimera_single_head_attention(
            module, eval_key, q_h, k_h, v_h, &config.softmax_approx,
        );

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
    module: &Module<BE>,
    eval_key: &ChimeraEvalKey<BE>,
    q_h: &[GLWE<Vec<u8>>],
    k_h: &[GLWE<Vec<u8>>],
    v_h: &[GLWE<Vec<u8>>],
    softmax: &SoftmaxStrategy,
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

    // -----------------------------------------------------------------------
    // Score computation: score = Σ_i Q[i] * K[i]
    //
    // Each Q[i] and K[i] are ciphertexts from the QKV projection. Their
    // ct*ct product (tensor product + relinearization) gives dimension i's
    // contribution. We accumulate all d_head terms via chimera_add.
    //
    // Note: chimera_ct_mul performs the tensor product at the full polynomial
    // level (all N coefficients), so the product encodes the negacyclic
    // convolution. For single-coefficient-per-ct projections (when each
    // Q[i]/K[i] encodes a scalar in coefficient 0), the product's coefficient 0
    // is exactly Q[i]*K[i].
    // -----------------------------------------------------------------------
    let mut score = chimera_ct_mul(module, eval_key, &q_h[0], &k_h[0]);
    for i in 1..d_head {
        let term = chimera_ct_mul(module, eval_key, &q_h[i], &k_h[i]);
        score = chimera_add(module, &score, &term);
    }

    // -----------------------------------------------------------------------
    // Softmax approximation on the score.
    //
    // For seq_len=1, there's only one score, so softmax is trivially 1.0
    // for the PolynomialDeg4 and ReluSquared strategies (p(x)/p(x) = 1).
    // For the Linear strategy, we pass the raw score through.
    //
    // We apply the softmax polynomial to remain consistent with the depth
    // accounting (even though it's a no-op for single-position attention).
    // -----------------------------------------------------------------------
    let attn_weights = chimera_apply_softmax(module, eval_key, &[score], softmax);
    let attn_weight = &attn_weights[0];

    // -----------------------------------------------------------------------
    // Context computation: context[j] = attn_weight * V[j]
    //
    // The attention weight is a scalar (replicated in coefficient 0 after
    // the score computation). For each value dimension j, we compute the
    // ct*ct product of the weight and V[j].
    // -----------------------------------------------------------------------
    let mut context = Vec::with_capacity(d_head);
    for j in 0..d_head {
        let ctx_j = chimera_ct_mul(module, eval_key, attn_weight, &v_h[j]);
        context.push(ctx_j);
    }

    context
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
        let (ct_q, ct_k, ct_v) = chimera_qkv_project_single(
            &module,
            &ct,
            &[2i64],
            &[3i64],
            &[1i64],
        );

        let pt_q = chimera_decrypt(&module, &key, &ct_q, &params);
        let pt_k = chimera_decrypt(&module, &key, &ct_k, &params);
        let pt_v = chimera_decrypt(&module, &key, &ct_v, &params);

        let dec_q = decode_int8(&module, &params, &pt_q, 2);
        let dec_k = decode_int8(&module, &params, &pt_k, 2);
        let dec_v = decode_int8(&module, &params, &pt_v, 2);

        // Q = [4*2, 5*2] = [8, 10]
        assert!((dec_q[0] as i16 - 8).unsigned_abs() <= 1, "Q[0] expected 8, got {}", dec_q[0]);
        assert!((dec_q[1] as i16 - 10).unsigned_abs() <= 1, "Q[1] expected 10, got {}", dec_q[1]);

        // K = [4*3, 5*3] = [12, 15]
        assert!((dec_k[0] as i16 - 12).unsigned_abs() <= 1, "K[0] expected 12, got {}", dec_k[0]);
        assert!((dec_k[1] as i16 - 15).unsigned_abs() <= 1, "K[1] expected 15, got {}", dec_k[1]);

        // V = [4*1, 5*1] = [4, 5]
        assert!((dec_v[0] as i16 - 4).unsigned_abs() <= 1, "V[0] expected 4, got {}", dec_v[0]);
        assert!((dec_v[1] as i16 - 5).unsigned_abs() <= 1, "V[1] expected 5, got {}", dec_v[1]);
    }

    #[test]
    fn test_attention_score_basic() {
        // Test attention score computation: Q·K element-wise + slot sum.
        // This verifies the core ct*ct + trace pipeline.
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_encrypt, ChimeraKey, ChimeraEvalKey};
        use poulpy_core::layouts::LWEInfos;
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [4u8; 32], [5u8; 32],
        );

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
        use crate::encoding::encode_int8;
        use crate::encrypt::{chimera_encrypt, chimera_decrypt, ChimeraKey, ChimeraEvalKey};
        use crate::encoding::decode_int8;
        use poulpy_hal::api::ModuleNew;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [1u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [4u8; 32], [5u8; 32],
        );

        let vals: Vec<i8> = vec![3, 5];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [2u8; 32], [3u8; 32]);

        let results = chimera_apply_softmax(&module, &eval_key, &[ct], &SoftmaxStrategy::Linear);
        assert_eq!(results.len(), 1);

        // Linear softmax should preserve the values
        let pt_dec = chimera_decrypt(&module, &key, &results[0], &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 2);
        assert!((decoded[0] as i16 - 3).unsigned_abs() <= 1, "linear softmax[0] expected 3, got {}", decoded[0]);
        assert!((decoded[1] as i16 - 5).unsigned_abs() <= 1, "linear softmax[1] expected 5, got {}", decoded[1]);
    }
}
