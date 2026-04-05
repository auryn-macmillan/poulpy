//! End-to-end encrypted inference pipeline.
//!
//! This module ties together all FHE_LLM components into a complete
//! inference pipeline that takes a user's text prompt and produces
//! a next-token prediction (or multi-token generation) under FHE.
//!
//! ## Architecture
//!
//! The pipeline is split across two parties:
//!
//! **User side (cleartext):**
//! 1. Tokenize input text → token IDs
//! 2. Look up token embedding (cleartext) → `Vec<i64>`
//! 3. Encrypt each embedding dimension → `Vec<GLWE>` (one ct per dim)
//! 4. Send encrypted input to provider
//!
//! **Provider side (encrypted):**
//! 5. Run N transformer layers homomorphically
//! 6. Return encrypted hidden state to user
//!
//! **User side (cleartext):**
//! 7. Decrypt hidden state → `Vec<i8>`
//! 8. Apply final RMSNorm locally when configured
//! 9. Apply LM head (cleartext matmul) → logits
//! 10. Argmax → next token ID
//! 11. Decode token ID → text
//!
//! ## Usage
//!
//! ```ignore
//! use poulpy_FHE_LLM::inference::{InferencePipeline, InferenceConfig};
//!
//! let pipeline = InferencePipeline::load(
//!     "/path/to/model.safetensors",
//!     "/path/to/tokenizer.json",
//!     InferenceConfig::default(),
//! )?;
//!
//! let output = pipeline.generate("Hello, world!", 1)?;
//! println!("Next token: {}", output.text);
//! ```

use std::path::Path;
use std::time::Instant;

use poulpy_core::layouts::{GLWEInfos, LWEInfos, GLWE};
use poulpy_hal::api::ModuleNew;
use poulpy_hal::layouts::Module;
use tokenizers::Tokenizer;

use crate::arithmetic::{_FHE_LLM_add, _FHE_LLM_align_layout};
use crate::attention::{_FHE_LLM_multi_head_attention_vec, AttentionConfig, AttentionWeights, SoftmaxStrategy};
use crate::bootstrapping::{BootstrappingConfig, FHE_LLMBootstrapKey, FHE_LLMBootstrapKeyPrepared};
use crate::encoding::encode_int8;
use crate::encrypt::{_FHE_LLM_decrypt, _FHE_LLM_encrypt, FHE_LLMEvalKey, FHE_LLMKey};
use crate::layernorm::{_FHE_LLM_rms_norm_vec, _FHE_LLM_rms_norm_vec_debug_stages, LayerNormConfig};
use crate::model_loader::{
    load_embedding_from_file, load_final_norm, load_layer_from_file, load_lm_head_from_file, EmbeddingTable, LMHead,
    LoaderConfig, ModelLoadError,
};
use crate::noise::NoiseTracker;
use crate::params::{FHE_LLMParams, ModelDims, Precision, SecurityLevel};
use crate::transformer::{
    _FHE_LLM_ffn_vec, _FHE_LLM_ffn_vec_scaled, _FHE_LLM_transformer_block_vec, FFNConfig, TransformerBlockConfig,
    TransformerBlockWeights,
};

// ---------------------------------------------------------------------------
// Backend selection (mirrors tests.rs)
// ---------------------------------------------------------------------------

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
type BE = poulpy_cpu_ref::FFT64Ref;
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
type BE = poulpy_cpu_avx::FFT64Avx;

const VEC_EFFECTIVE_DECODE_SCALE: u32 = 16;
const VEC_EFFECTIVE_QUANT_SCALE: f64 = 1.0;
const REFRESHED_EFFECTIVE_DECODE_SCALE: u32 = 18;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the inference pipeline.
#[derive(Clone, Debug)]
pub struct InferenceConfig {
    /// Security level for FHE parameters.
    pub security: SecurityLevel,
    /// Precision for FHE encoding.
    pub precision: Precision,
    /// Number of transformer layers to evaluate.
    /// If `None`, uses all layers from the model.
    /// For practical FHE inference, this is typically truncated
    /// to a small number (1-4) to stay within noise budget.
    pub num_layers: Option<usize>,
    /// Truncated model dimension.
    /// If `Some(d)`, truncates all weight matrices to the first `d`
    /// dimensions of the embedding space. This dramatically reduces
    /// FHE cost at the expense of model quality.
    /// If `None`, uses the full model dimension.
    pub trunc_d_model: Option<usize>,
    /// Truncated FFN dimension.
    /// If `Some(d)`, truncates FFN weight matrices to `d` hidden dims.
    /// If `None`, uses the full FFN dimension.
    pub trunc_d_ffn: Option<usize>,
    /// Number of attention heads to use.
    /// If `None`, defaults to 1 (for truncated models) or full heads.
    pub num_heads: Option<usize>,
    /// Number of KV heads (for GQA).
    /// If `None`, matches num_heads.
    pub num_kv_heads: Option<usize>,
    /// Softmax approximation strategy.
    pub softmax_strategy: SoftmaxStrategy,
    /// Whether to apply the final RMSNorm before LM head.
    pub apply_final_norm: bool,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Seed bytes for key generation.
    pub key_seed: [u8; 32],
    /// Seed bytes for eval key generation (tensor key).
    pub eval_seed_a: [u8; 32],
    /// Seed bytes for eval key generation (noise).
    pub eval_seed_b: [u8; 32],
    /// Bootstrap precision for SiLU LUT (log2 of modulus).
    pub fhe_silu_log_msg_mod: Option<u8>,
    /// Bootstrap precision for identity refresh LUT (log2 of modulus).
    pub fhe_identity_log_msg_mod: Option<u8>,
    /// Enable frequent bootstraps after each operation.
    pub fhe_frequent_bootstrap: bool,
    /// Enable extra refreshes after sub-operations.
    pub fhe_extra_refresh: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            security: SecurityLevel::Bits80,
            precision: Precision::Int8,
            num_layers: Some(1),
            trunc_d_model: Some(64),
            trunc_d_ffn: Some(128),
            num_heads: Some(1),
            num_kv_heads: Some(1),
            softmax_strategy: SoftmaxStrategy::ReluSquared,
            apply_final_norm: true,
            max_new_tokens: 1,
            key_seed: [42u8; 32],
            eval_seed_a: [43u8; 32],
            eval_seed_b: [44u8; 32],
            fhe_silu_log_msg_mod: None,
            fhe_identity_log_msg_mod: None,
            fhe_frequent_bootstrap: false,
            fhe_extra_refresh: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Model specification
// ---------------------------------------------------------------------------

/// Full model dimensions parsed from config.json or known model configs.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    /// Full model dimensions.
    pub dims: ModelDims,
    /// Name of the embedding tensor in safetensors.
    pub embed_name: String,
    /// Name of the LM head tensor in safetensors.
    pub lm_head_name: String,
    /// Name of the final norm tensor in safetensors.
    pub final_norm_name: String,
    /// RoPE theta parameter.
    pub rope_theta: f64,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// BOS token ID.
    pub bos_token_id: u32,
    /// EOS token ID.
    pub eos_token_id: u32,
}

impl ModelSpec {
    /// TinyLlama 1.1B Chat v1.0 specification.
    pub fn tinyllama_1_1b() -> Self {
        ModelSpec {
            dims: ModelDims {
                d_model: 2048,
                d_head: 64,
                n_heads: 32,
                n_kv_heads: 4,
                d_ffn: 5632,
                n_layers: 22,
                n_experts: 1,
                n_active_experts: 1,
            },
            embed_name: "model.embed_tokens.weight".to_string(),
            lm_head_name: "lm_head.weight".to_string(),
            final_norm_name: "model.norm.weight".to_string(),
            rope_theta: 10000.0,
            max_seq_len: 2048,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// LLaMA 7B specification.
    pub fn llama_7b() -> Self {
        ModelSpec {
            dims: ModelDims::dense_7b(),
            embed_name: "model.embed_tokens.weight".to_string(),
            lm_head_name: "lm_head.weight".to_string(),
            final_norm_name: "model.norm.weight".to_string(),
            rope_theta: 10000.0,
            max_seq_len: 2048,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Inference result
// ---------------------------------------------------------------------------

/// Result of a single inference step (one new token).
#[derive(Clone, Debug)]
pub struct InferenceStepResult {
    /// Predicted next token ID.
    pub token_id: u32,
    /// Decoded text for this token.
    pub token_text: String,
    /// Raw decrypted hidden state (d_model or trunc_d_model values).
    pub hidden_state: Vec<i8>,
    /// Top-5 logits (token_id, logit_value) for diagnostics.
    pub top_logits: Vec<(u32, i64)>,
    /// Wall-clock time for the FHE forward pass.
    pub fhe_time: std::time::Duration,
    /// Wall-clock time for the entire step (including cleartext ops).
    pub total_time: std::time::Duration,
}

/// Result of a multi-token generation.
#[derive(Clone, Debug)]
pub struct GenerationResult {
    /// Input prompt text.
    pub prompt: String,
    /// Input token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Generated token IDs.
    pub generated_tokens: Vec<u32>,
    /// Generated text (decoded from generated_tokens).
    pub generated_text: String,
    /// Full text (prompt + generated).
    pub full_text: String,
    /// Per-step results.
    pub steps: Vec<InferenceStepResult>,
    /// Total wall-clock time.
    pub total_time: std::time::Duration,
}

#[derive(Clone, Debug)]
pub struct HiddenRangeStats {
    pub stage: String,
    pub min: i64,
    pub max: i64,
    pub overflow_dims: usize,
    pub total_dims: usize,
}

#[derive(Clone, Debug)]
pub struct PlaintextRmsStats {
    pub stage: String,
    pub mean_sq: f64,
}

#[derive(Clone, Debug)]
pub struct ScalarStageValue {
    pub stage: String,
    pub value: i64,
}

#[derive(Clone, Debug)]
pub struct StageErrorStats {
    pub stage: String,
    pub linf: f64,
    pub l2: f64,
    pub mae: f64,
}

#[derive(Clone, Debug)]
pub struct RefreshedDecodeEval {
    pub precision: u32,
    pub range: HiddenRangeStats,
    pub linf: f64,
    pub l2: f64,
    pub mae: f64,
}

#[derive(Clone, Debug)]
pub struct VariantRangeStats {
    pub variant: String,
    pub decode_precision: u32,
    pub min: i64,
    pub max: i64,
    pub overflow_dims: usize,
    pub total_dims: usize,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during inference.
#[derive(Debug)]
pub enum InferenceError {
    /// Model loading failed.
    ModelLoad(ModelLoadError),
    /// Tokenizer loading or encoding/decoding failed.
    Tokenizer(String),
    /// Configuration error.
    Config(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::ModelLoad(e) => write!(f, "model load error: {e}"),
            InferenceError::Tokenizer(e) => write!(f, "tokenizer error: {e}"),
            InferenceError::Config(e) => write!(f, "config error: {e}"),
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<ModelLoadError> for InferenceError {
    fn from(e: ModelLoadError) -> Self {
        InferenceError::ModelLoad(e)
    }
}

impl From<std::io::Error> for InferenceError {
    fn from(e: std::io::Error) -> Self {
        InferenceError::ModelLoad(ModelLoadError::Io(e))
    }
}

// ---------------------------------------------------------------------------
// Weight truncation helpers
// ---------------------------------------------------------------------------

/// Truncates a 2D weight matrix to the first `rows` rows and `cols` columns.
fn truncate_weights(w: &[Vec<i64>], rows: usize, cols: usize) -> Vec<Vec<i64>> {
    w.iter().take(rows).map(|row| row[..cols].to_vec()).collect()
}

/// Truncates a 1D vector to the first `len` elements.
fn truncate_1d(v: &[i64], len: usize) -> Vec<i64> {
    v[..len].to_vec()
}

// ---------------------------------------------------------------------------
// Inference pipeline
// ---------------------------------------------------------------------------

/// Complete inference pipeline for encrypted LLM inference.
///
/// Holds the tokenizer, model weights, FHE keys, and configuration
/// needed to run prompt-to-text inference under FHE.
pub struct InferencePipeline {
    /// HuggingFace tokenizer.
    tokenizer: Tokenizer,
    /// Token embedding table (cleartext, user-side).
    embedding: EmbeddingTable,
    /// LM head (cleartext, user-side).
    lm_head: LMHead,
    /// Per-layer transformer weights (truncated if applicable).
    layer_weights: Vec<TransformerBlockWeights>,
    /// Final RMSNorm gamma (truncated if applicable).
    final_norm_gamma: Option<Vec<i64>>,
    /// Model specification.
    model_spec: ModelSpec,
    /// Inference configuration.
    config: InferenceConfig,
    /// Effective dimensions (after truncation).
    effective_dims: ModelDims,
    /// FHE parameters.
    params: FHE_LLMParams,
    /// Backend module.
    module: Module<BE>,
    /// FHE secret key (user-side).
    key: FHE_LLMKey<BE>,
    /// FHE evaluation key (sent to provider).
    eval_key: FHE_LLMEvalKey<BE>,
    /// Prepared bootstrap key for LUT/refresh experiments.
    bsk_prepared: Option<FHE_LLMBootstrapKeyPrepared<BE>>,
}

impl InferencePipeline {
    /// Loads a complete inference pipeline from model files.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the safetensors model file.
    /// * `tokenizer_path` - Path to the tokenizer.json file.
    /// * `model_spec` - Model specification (dimensions, tensor names).
    /// * `config` - Inference configuration.
    ///
    /// # Returns
    ///
    /// A ready-to-use inference pipeline.
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        model_path: P,
        tokenizer_path: Q,
        model_spec: ModelSpec,
        config: InferenceConfig,
    ) -> Result<Self, InferenceError> {
        let model_path = model_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();

        eprintln!("[inference] Loading tokenizer from {:?}...", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| InferenceError::Tokenizer(format!("failed to load tokenizer: {e}")))?;

        // Determine effective dimensions (after truncation)
        let full_dims = &model_spec.dims;
        let d_model = config.trunc_d_model.unwrap_or(full_dims.d_model);
        let d_ffn = config.trunc_d_ffn.unwrap_or(full_dims.d_ffn);
        let n_heads = config.num_heads.unwrap_or(1);
        let n_kv_heads = config.num_kv_heads.unwrap_or(n_heads);
        let d_head = d_model / n_heads;
        let num_layers = config.num_layers.unwrap_or(full_dims.n_layers);

        if d_model > full_dims.d_model {
            return Err(InferenceError::Config(format!(
                "trunc_d_model ({d_model}) exceeds full d_model ({})",
                full_dims.d_model
            )));
        }

        let effective_dims = ModelDims {
            d_model,
            d_head,
            n_heads,
            n_kv_heads,
            d_ffn,
            n_layers: num_layers,
            n_experts: 1,
            n_active_experts: 1,
        };

        eprintln!(
            "[inference] Effective dims: d_model={}, d_ffn={}, n_heads={}, n_kv_heads={}, d_head={}, layers={}",
            d_model, d_ffn, n_heads, n_kv_heads, d_head, num_layers
        );

        // Load embedding table (full vocab, full d_model — we truncate at lookup time)
        eprintln!("[inference] Loading embedding table...");
        let embedding = load_embedding_from_file(model_path, &model_spec.embed_name, full_dims.d_model)?;
        eprintln!(
            "[inference] Embedding loaded: vocab_size={}, d_model={}",
            embedding.vocab_size, embedding.d_model
        );

        // Load LM head (full vocab × full d_model — we truncate at forward time)
        eprintln!("[inference] Loading LM head...");
        let lm_head = load_lm_head_from_file(model_path, &model_spec.lm_head_name, full_dims.d_model)?;
        eprintln!(
            "[inference] LM head loaded: vocab_size={}, d_model={}",
            lm_head.vocab_size, lm_head.d_model
        );

        // Load final norm (if configured)
        let final_norm_gamma = if config.apply_final_norm {
            eprintln!("[inference] Loading final norm...");
            let file = std::fs::File::open(model_path)?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let tensors = safetensors::SafeTensors::deserialize(&mmap).map_err(|e| ModelLoadError::SafeTensors(e.to_string()))?;
            let (gamma, _qi) = load_final_norm(&tensors, &model_spec.final_norm_name, full_dims.d_model)?;
            let truncated = if d_model < full_dims.d_model {
                truncate_1d(&gamma, d_model)
            } else {
                gamma
            };
            eprintln!("[inference] Final norm loaded ({} dims)", truncated.len());
            Some(truncated)
        } else {
            None
        };

        // Load transformer layers
        eprintln!("[inference] Loading {} transformer layer(s)...", num_layers);
        let loader_config = LoaderConfig::llama(full_dims.clone());
        let mut layer_weights = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            eprintln!("[inference]   Loading layer {layer_idx}...");
            let layer_result = load_layer_from_file(model_path, layer_idx, &loader_config)?;
            let full_w = layer_result.weights;

            // Truncate weights to effective dimensions
            let trunc_attn = AttentionWeights {
                w_q: truncate_weights(&full_w.attention.w_q, d_model, d_model),
                w_k: truncate_weights(&full_w.attention.w_k, d_model, d_model),
                w_v: truncate_weights(&full_w.attention.w_v, d_model, d_model),
                w_o: truncate_weights(&full_w.attention.w_o, d_model, d_model),
            };

            let trunc_ffn_loader = crate::transformer::FFNWeights {
                w1: truncate_weights(&full_w.ffn.w1, d_model, d_ffn),
                w2: truncate_weights(&full_w.ffn.w2, d_ffn, d_model),
                w3: full_w.ffn.w3.as_ref().map(|w3| truncate_weights(w3, d_model, d_ffn)),
            };

            let trunc_attn_gamma = full_w.pre_attn_norm_gamma.as_ref().map(|g| truncate_1d(g, d_model));
            let trunc_ffn_gamma = full_w.pre_ffn_norm_gamma.as_ref().map(|g| truncate_1d(g, d_model));

            let weights = TransformerBlockWeights {
                attention: trunc_attn,
                ffn: trunc_ffn_loader.into_vec_pipeline_weights(),
                pre_attn_norm_gamma: trunc_attn_gamma,
                pre_ffn_norm_gamma: trunc_ffn_gamma,
            };

            layer_weights.push(weights);
        }

        eprintln!("[inference] All layers loaded and truncated");

        // Generate FHE keys
        let params = FHE_LLMParams::new(config.security, config.precision);
        eprintln!(
            "[inference] Generating FHE keys ({:?} security, N={})...",
            config.security,
            params.n()
        );
        let module: Module<BE> = Module::new(params.n());
        let key = FHE_LLMKey::generate(&module, &params, config.key_seed);
        let eval_key = FHE_LLMEvalKey::generate(&module, &key, &params, config.eval_seed_a, config.eval_seed_b);
        eprintln!("[inference] FHE keys generated");

        let bootstrap_config = BootstrappingConfig::with_functional_bootstrap(6.0, 1);
        let bsk_prepared = if bootstrap_config.enabled {
            eprintln!("[inference] Generating bootstrap key...");
            let bsk = FHE_LLMBootstrapKey::generate(
                &module,
                &params,
                &key.secret,
                &key.prepared,
                [201u8; 32],
                [202u8; 32],
                [203u8; 32],
            );
            let prepared = FHE_LLMBootstrapKeyPrepared::prepare(&module, &bsk);
            eprintln!("[inference] Bootstrap key prepared");
            Some(prepared)
        } else {
            None
        };

        Ok(InferencePipeline {
            tokenizer,
            embedding,
            lm_head,
            layer_weights,
            final_norm_gamma,
            model_spec,
            config,
            effective_dims,
            params,
            module,
            key,
            eval_key,
            bsk_prepared,
        })
    }

    /// Tokenizes a text prompt into token IDs.
    ///
    /// Prepends the BOS token if the tokenizer doesn't do so automatically.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| InferenceError::Tokenizer(format!("encode failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decodes a sequence of token IDs back to text.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, InferenceError> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|e| InferenceError::Tokenizer(format!("decode failed: {e}")))
    }

    /// Decodes a single token ID to text.
    pub fn decode_token(&self, token_id: u32) -> Result<String, InferenceError> {
        self.decode(&[token_id])
    }

    /// Looks up the embedding for a token and returns it truncated to
    /// the effective d_model, clamped to i8 range.
    fn embed_token(&self, token_id: u32) -> Vec<i8> {
        let full_emb = self.embedding.lookup(token_id as usize);
        let d = self.effective_dims.d_model;
        full_emb[..d].iter().map(|&v| v.clamp(-127, 127) as i8).collect()
    }

    /// Encrypts a vector of i8 values (one per model dimension) into
    /// per-dimension ciphertexts.
    fn encrypt_embedding(&self, values: &[i8]) -> Vec<GLWE<Vec<u8>>> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let pt = encode_int8(&self.module, &self.params, &[v]);
                let mut seed_a = [0u8; 32];
                let mut seed_e = [0u8; 32];
                // Use dimension index as entropy source for deterministic seeds
                seed_a[0] = (i as u8).wrapping_add(100);
                seed_a[1] = ((i >> 8) as u8).wrapping_add(50);
                seed_e[0] = (i as u8).wrapping_add(200);
                seed_e[1] = ((i >> 8) as u8).wrapping_add(150);
                _FHE_LLM_encrypt(&self.module, &self.key, &pt, seed_a, seed_e)
            })
            .collect()
    }

    fn decrypt_hidden_state_at_precision(&self, cts: &[GLWE<Vec<u8>>], precision: u32) -> Vec<i8> {
        cts.iter()
            .map(|ct| {
                let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
                let mut decoded = vec![0i64; self.module.n() as usize];
                pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
                decoded[0].clamp(-128, 127) as i8
            })
            .collect()
    }

    /// Decrypts a vector of per-dimension ciphertexts to raw i64 values at the
    /// INT8 encoding scale, without truncating to i8.
    pub fn decrypt_hidden_state_raw(&self, cts: &[GLWE<Vec<u8>>]) -> Vec<i64> {
        cts.iter()
            .map(|ct| {
                let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
                let mut decoded = vec![0i64; self.module.n() as usize];
                pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(VEC_EFFECTIVE_DECODE_SCALE));
                decoded[0]
            })
            .collect()
    }

    /// Runs the legacy encrypted forward pass.
    ///
    /// This keeps final RMSNorm on the provider side when configured. The
    /// refreshed production path instead leaves final RMSNorm to the client
    /// after decryption.
    fn fhe_forward(&self, encrypted_input: &[GLWE<Vec<u8>>]) -> Vec<GLWE<Vec<u8>>> {
        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None, // RoPE requires position-dependent precomputation per token
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        // Run transformer layers sequentially
        let mut current = encrypted_input.to_vec();
        for (layer_idx, weights) in self.layer_weights.iter().enumerate() {
            eprintln!(
                "[inference]   FHE layer {}/{} (d_model={})...",
                layer_idx + 1,
                self.layer_weights.len(),
                d
            );
            let layer_start = Instant::now();
            current = _FHE_LLM_transformer_block_vec(&self.module, &self.eval_key, &current, &block_config, weights);
            eprintln!("[inference]   Layer {} done in {:.2?}", layer_idx, layer_start.elapsed());
        }

        // Apply final RMSNorm if configured.
        if let Some(ref gamma) = self.final_norm_gamma {
            eprintln!("[inference]   Applying final RMSNorm...");
            let norm_cfg = LayerNormConfig::rms_norm(d).with_gamma(gamma.clone());
            current = crate::layernorm::_FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &current, &norm_cfg);
        }

        current
    }

    /// Production refreshed forward pass.
    ///
    /// This inserts decrypt/re-encrypt style refresh boundaries and uses a
    /// LUT-based FFN gate activation. Final RMSNorm is intentionally left for
    /// the client after decryption.
    fn fhe_forward_refreshed(&self, encrypted_input: &[GLWE<Vec<u8>>]) -> Vec<GLWE<Vec<u8>>> {
        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm_midrange(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let mut current = encrypted_input.to_vec();
        for weights in self.layer_weights.iter() {
            let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
                Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
                None => block_config.pre_attn_norm.clone(),
            };
            let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &current, &pre_attn_norm_cfg);
            let attn_out = _FHE_LLM_multi_head_attention_vec(
                &self.module,
                &self.eval_key,
                &normed_pre_attn,
                &weights.attention,
                &block_config.attention,
            );
            let residual_1 = if self.config.fhe_extra_refresh {
                let log_msg_mod = self.config.fhe_identity_log_msg_mod.unwrap_or(7) as u32;
                let attn_out_refreshed = self.refresh_vec_at_precision(&attn_out, log_msg_mod);
                self.project_and_add_diag(&current, &attn_out_refreshed)
            } else {
                self.project_and_add_diag(&current, &attn_out)
            };
            let residual_1_refreshed = self.refresh_vec_at_effective_scale(&residual_1);

            let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
                Some(gamma) => block_config.pre_ffn_norm.clone().with_gamma(gamma.clone()),
                None => block_config.pre_ffn_norm.clone(),
            };
            let normed_pre_ffn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &residual_1_refreshed, &pre_ffn_norm_cfg);

            let d_in = normed_pre_ffn.len();
            let d_ffn = weights.ffn.w1.len();
            let w_up = weights.ffn.w3.as_ref().expect("SwiGLU requires w3");

            let mut gate_fhe: Vec<GLWE<Vec<u8>>> = (0..d_ffn)
                .map(|j| {
                    let wg_vecs: Vec<Vec<i64>> = weights.ffn.w1[j][..d_in].iter().map(|&w| vec![w]).collect();
                    crate::arithmetic::_FHE_LLM_dot_product_scaled(
                        &self.module,
                        &normed_pre_ffn,
                        &wg_vecs,
                        crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
                    )
                })
                .collect();
            if self.config.fhe_extra_refresh {
                let log_msg_mod = self.config.fhe_identity_log_msg_mod.unwrap_or(7) as u32;
                gate_fhe = self.refresh_vec_at_precision(&gate_fhe, log_msg_mod);
            }
            let gate_act_fhe = self.apply_silu_lut_vec(&gate_fhe);

            let up_fhe: Vec<GLWE<Vec<u8>>> = (0..d_ffn)
                .map(|j| {
                    let wu_vecs: Vec<Vec<i64>> = w_up[j][..d_in].iter().map(|&w| vec![w]).collect();
                    crate::arithmetic::_FHE_LLM_dot_product_scaled(
                        &self.module,
                        &normed_pre_ffn,
                        &wu_vecs,
                        crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
                    )
                })
                .collect();

            // Refresh up projection before elementwise multiplication
            let up_fhe_refreshed = if self.config.fhe_extra_refresh {
                let log_msg_mod = self.config.fhe_identity_log_msg_mod.unwrap_or(7) as u32;
                self.refresh_vec_at_precision(&up_fhe, log_msg_mod)
            } else {
                up_fhe.clone()
            };

            let hidden_fhe: Vec<GLWE<Vec<u8>>> = gate_act_fhe
                .iter()
                .zip(up_fhe_refreshed.iter())
                .map(|(gate, up)| {
                    let gate_layout = poulpy_core::layouts::GLWELayout {
                        n: gate.n(),
                        base2k: gate.base2k(),
                        k: gate.k(),
                        rank: gate.rank(),
                    };
                    let up_proj = if up.base2k() == gate.base2k() {
                        up.clone()
                    } else {
                        crate::arithmetic::_FHE_LLM_align_layout(&self.module, up, &gate_layout)
                    };
                    crate::activations::_FHE_LLM_ct_mul(&self.module, &self.eval_key, gate, &up_proj)
                })
                .collect();
            let hidden_fhe_refreshed = self.refresh_vec_at_effective_scale(&hidden_fhe);

            // compute down projection (FFN down)
            let down_fhe: Vec<GLWE<Vec<u8>>> = weights
                .ffn
                .w2
                .iter()
                .map(|row| {
                    let w2_vecs: Vec<Vec<i64>> = row.iter().map(|&w| vec![w]).collect();
                    crate::arithmetic::_FHE_LLM_dot_product(&self.module, &hidden_fhe_refreshed, &w2_vecs)
                })
                .collect();

            // refresh down projection if extra refresh enabled
            let down_fhe_refreshed = if self.config.fhe_extra_refresh {
                let log_msg_mod = self.config.fhe_identity_log_msg_mod.unwrap_or(7) as u32;
                self.refresh_vec_at_precision(&down_fhe, log_msg_mod)
            } else {
                down_fhe.clone()
            };

            // final residual addition after refreshed down projection
            current = self.project_and_add_diag(&residual_1_refreshed, &down_fhe_refreshed);

            // final refresh after the last residual addition
            if self.config.fhe_extra_refresh {
                let log_msg_mod = self.config.fhe_identity_log_msg_mod.unwrap_or(7) as u32;
                current = self.refresh_vec_at_precision(&current, log_msg_mod);
            }
        }

        current
    }

    /// Runs the plaintext forward pass using exact f64 nonlinearities.
    ///
    /// This is the gold-standard reference for comparing FHE outputs.
    /// It uses the same INT8 quantised weights but operates entirely in
    /// f64 domain with exact softmax, SiLU, and 1/sqrt.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Token embedding as i8 values (d_model dimensions).
    ///
    /// # Returns
    ///
    /// Hidden state in f64 domain (d_model dimensions).
    pub fn plaintext_forward(&self, embedding: &[i8]) -> Vec<f64> {
        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let x: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();

        crate::plaintext_forward::forward_pass_with_final_norm(
            &x,
            &block_config,
            &self.layer_weights,
            self.final_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
        )
    }

    /// Runs the polynomial-approximation plaintext forward pass.
    ///
    /// Uses the same polynomial approximations as the FHE pipeline but
    /// without encryption noise. This isolates the FHE noise contribution
    /// when compared against the FHE output.
    pub fn plaintext_forward_poly(&self, embedding: &[i8]) -> Vec<f64> {
        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let x: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();

        crate::plaintext_forward::forward_pass_poly_approx(
            &x,
            &block_config,
            &self.layer_weights,
            self.final_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
        )
    }

    /// Runs a plaintext inference step and returns the result.
    ///
    /// This is the plaintext equivalent of [`step`], returning both the
    /// hidden state (in f64) and LM head logits for comparison with FHE.
    pub fn plaintext_step(&self, token_id: u32) -> crate::plaintext_forward::PlaintextStepResult {
        let embedding = self.embed_token(token_id);
        let d = self.effective_dims.d_model;

        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        // Prepare LM head weights (truncated if needed)
        let lm_weights: Vec<Vec<i64>> = if d < self.lm_head.d_model {
            self.lm_head.weights.iter().map(|row| row[..d].to_vec()).collect()
        } else {
            self.lm_head.weights.clone()
        };

        crate::plaintext_forward::plaintext_step(
            &embedding,
            &block_config,
            &self.layer_weights,
            self.final_norm_gamma.as_deref(),
            &lm_weights,
            d,
        )
    }

    /// Performs a three-way error comparison after an FHE inference step.
    ///
    /// Given the FHE-decrypted hidden state and the original token ID,
    /// computes error metrics that decompose the total error into:
    /// - Polynomial approximation error
    /// - FHE noise
    ///
    /// # Returns
    ///
    /// A [`ThreeWayComparison`] with L-inf, L2, and MAE for each pair.
    pub fn compare_fhe_vs_plaintext(&self, fhe_hidden: &[i8], token_id: u32) -> crate::plaintext_forward::ThreeWayComparison {
        if self.final_norm_gamma.is_some() {
            let (linf, l2, mae) = self.compare_fhe_vs_plaintext_refreshed(fhe_hidden, token_id);
            return crate::plaintext_forward::ThreeWayComparison {
                fhe_vs_exact: (linf, l2, mae),
                poly_vs_exact: (0.0, 0.0, 0.0),
                fhe_vs_poly: (linf, l2, mae),
            };
        }

        let embedding = self.embed_token(token_id);
        let d = self.effective_dims.d_model;

        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        crate::plaintext_forward::three_way_comparison_quantized(
            fhe_hidden,
            &embedding,
            &block_config,
            &self.layer_weights,
            self.final_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
            VEC_EFFECTIVE_QUANT_SCALE,
        )
    }

    pub fn compare_fhe_vs_plaintext_refreshed(&self, fhe_hidden: &[i8], token_id: u32) -> (f64, f64, f64) {
        let target = self.refreshed_plain_target(token_id);
        let fhe: Vec<f64> = fhe_hidden.iter().map(|&v| v as f64).collect();
        crate::plaintext_forward::error_metrics(&fhe, &target)
    }

    pub fn refreshed_plain_target(&self, token_id: u32) -> Vec<f64> {
        let output_scale_divisor = 256.0;
        let d = self.effective_dims.d_model;
        let embedding = self.embed_token(token_id);
        let x: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();

        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm_midrange(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let w_up = weights.ffn.w3.as_ref().expect("SwiGLU requires w3");

        let pre_attn =
            crate::plaintext_forward::rms_norm(&x, weights.pre_attn_norm_gamma.as_deref(), block_config.pre_attn_norm.epsilon);
        let attn = crate::plaintext_forward::multi_head_attention(&pre_attn, &weights.attention, &block_config.attention);
        let residual_1: Vec<f64> = x.iter().zip(attn.iter()).map(|(a, b)| a + b).collect();
        let residual_1_refreshed: Vec<f64> = residual_1.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();

        let pre_ffn = crate::plaintext_forward::rms_norm(
            &residual_1_refreshed,
            weights.pre_ffn_norm_gamma.as_deref(),
            block_config.pre_ffn_norm.epsilon,
        );
        let gate = crate::plaintext_forward::matvec(&weights.ffn.w1, &pre_ffn);
        let gate_activated: Vec<f64> = gate.iter().map(|&v| crate::plaintext_forward::exact_silu(v)).collect();
        let up = crate::plaintext_forward::matvec(w_up, &pre_ffn);
        let hidden: Vec<f64> = gate_activated.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
        let hidden_refreshed: Vec<f64> = hidden.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        let down = crate::plaintext_forward::matvec(&weights.ffn.w2, &hidden_refreshed);
        let block_out: Vec<f64> = residual_1_refreshed.iter().zip(down.iter()).map(|(a, b)| a + b).collect();
        let block_out_refreshed: Vec<f64> = block_out.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        let final_plain: Vec<f64> = if let Some(gamma) = self.final_norm_gamma.as_deref() {
            crate::plaintext_forward::rms_norm(&block_out_refreshed, Some(gamma), 1e-5)
        } else {
            block_out_refreshed
        };

        final_plain
            .iter()
            .map(|&v| {
                let divisor = if self.final_norm_gamma.is_some() {
                    1.0
                } else {
                    output_scale_divisor
                };
                (v / divisor).round().clamp(-128.0, 127.0)
            })
            .collect()
    }

    fn decode_refreshed_hidden_from_output(&self, encrypted_output: &[GLWE<Vec<u8>>], precision: u32) -> Vec<i8> {
        if self.final_norm_gamma.is_some() {
            let decoded = self.decode_ct_vec_at_precision(encrypted_output, precision);
            self.apply_cleartext_final_norm_quantized(&decoded)
        } else {
            self.decrypt_hidden_state_at_precision(encrypted_output, precision)
        }
    }

    pub fn step_refreshed(&self, token_id: u32) -> Result<InferenceStepResult, InferenceError> {
        let step_start = Instant::now();

        let embedding = self.embed_token(token_id);
        let encrypted = self.encrypt_embedding(&embedding);

        let fhe_start = Instant::now();
        let encrypted_output = self.fhe_forward_refreshed(&encrypted);
        let fhe_time = fhe_start.elapsed();

        let hidden_state = self.decode_refreshed_hidden_from_output(&encrypted_output, REFRESHED_EFFECTIVE_DECODE_SCALE);
        let logits = self.lm_head_forward(&hidden_state);

        let mut indexed_logits: Vec<(u32, i64)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        indexed_logits.sort_by_key(|&(_, l)| std::cmp::Reverse(l));
        let top_logits = indexed_logits.into_iter().take(5).collect::<Vec<_>>();

        let predicted_token = top_logits[0].0;
        let token_text = self
            .tokenizer
            .decode(&[predicted_token], true)
            .unwrap_or_else(|_| format!("<tok:{}>", predicted_token));

        Ok(InferenceStepResult {
            token_id: predicted_token,
            token_text,
            hidden_state,
            top_logits,
            fhe_time,
            total_time: step_start.elapsed(),
        })
    }

    pub fn refreshed_hidden_range_at_precision(&self, token_id: u32, precision: u32) -> HiddenRangeStats {
        let embedding = self.embed_token(token_id);
        let encrypted = self.encrypt_embedding(&embedding);
        let encrypted_output = self.fhe_forward_refreshed(&encrypted);
        let raw_i64: Vec<i64> = self
            .decode_refreshed_hidden_from_output(&encrypted_output, precision)
            .iter()
            .map(|&v| v as i64)
            .collect();
        HiddenRangeStats {
            stage: format!("refreshed_final@{precision}"),
            min: raw_i64.iter().copied().min().unwrap_or(0),
            max: raw_i64.iter().copied().max().unwrap_or(0),
            overflow_dims: raw_i64.iter().filter(|&&v| !(-128..=127).contains(&v)).count(),
            total_dims: raw_i64.len(),
        }
    }

    pub fn refreshed_hidden_at_precision(&self, token_id: u32, precision: u32) -> Vec<i8> {
        let embedding = self.embed_token(token_id);
        let encrypted = self.encrypt_embedding(&embedding);
        let encrypted_output = self.fhe_forward_refreshed(&encrypted);
        self.decode_refreshed_hidden_from_output(&encrypted_output, precision)
    }

    pub fn refreshed_decode_sweep(&self, token_id: u32, precisions: &[u32]) -> Vec<RefreshedDecodeEval> {
        let embedding = self.embed_token(token_id);
        let encrypted = self.encrypt_embedding(&embedding);
        let encrypted_output = self.fhe_forward_refreshed(&encrypted);
        let target = self.refreshed_plain_target(token_id);

        precisions
            .iter()
            .copied()
            .map(|precision| {
                let hidden = self.decode_refreshed_hidden_from_output(&encrypted_output, precision);
                let fhe: Vec<f64> = hidden.iter().map(|&v| v as f64).collect();
                let (linf, l2, mae) = crate::plaintext_forward::error_metrics(&fhe, &target);
                let raw_i64: Vec<i64> = fhe.iter().map(|&v| v as i64).collect();
                let range = HiddenRangeStats {
                    stage: format!("refreshed_final@{precision}"),
                    min: raw_i64.iter().copied().min().unwrap_or(0),
                    max: raw_i64.iter().copied().max().unwrap_or(0),
                    overflow_dims: raw_i64.iter().filter(|&&v| !(-128..=127).contains(&v)).count(),
                    total_dims: raw_i64.len(),
                };
                RefreshedDecodeEval {
                    precision,
                    range,
                    linf,
                    l2,
                    mae,
                }
            })
            .collect()
    }

    /// Runs the FHE forward path for one token and returns the decrypted hidden
    /// state as raw i64 values before truncation to i8.
    pub fn raw_hidden_state_for_token(&self, token_id: u32) -> Vec<i64> {
        let embedding = self.embed_token(token_id);
        let encrypted = self.encrypt_embedding(&embedding);
        let encrypted_output = self.fhe_forward(&encrypted);
        self.decrypt_hidden_state_raw(&encrypted_output)
    }

    pub fn diagnose_first_block_ranges_for_token(&self, token_id: u32) -> Vec<HiddenRangeStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_first_block_ranges_for_token: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));
        let mut stats = vec![self.hidden_range_stats("input", &encrypted)];

        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        stats.push(self.hidden_range_stats("pre_attn_norm", &normed_pre_attn));

        let attn_out = _FHE_LLM_multi_head_attention_vec(
            &self.module,
            &self.eval_key,
            &normed_pre_attn,
            &weights.attention,
            &block_config.attention,
        );
        stats.push(self.hidden_range_stats("attn_out", &attn_out));

        let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);
        stats.push(self.hidden_range_stats("residual_1", &residual_1));

        let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
            Some(gamma) => LayerNormConfig::rms_norm_midrange(d).with_gamma(gamma.clone()),
            None => LayerNormConfig::rms_norm_midrange(d),
        };
        let residual_1_for_norm: Vec<_> = residual_1
            .iter()
            .map(|ct| crate::arithmetic::_FHE_LLM_mul_const_scaled(&self.module, ct, &[1], 1))
            .collect();
        let normed_pre_ffn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &residual_1_for_norm, &pre_ffn_norm_cfg);
        stats.push(self.hidden_range_stats("pre_ffn_norm", &normed_pre_ffn));

        let ffn_out = _FHE_LLM_ffn_vec(&self.module, &self.eval_key, &normed_pre_ffn, &weights.ffn, &block_config.ffn);
        stats.push(self.hidden_range_stats("ffn_out", &ffn_out));

        let block_out = self.project_and_add_diag(&residual_1, &ffn_out);
        stats.push(self.hidden_range_stats("block_out", &block_out));

        stats
    }

    pub fn diagnose_first_block_stage_for_token_at_precision(
        &self,
        token_id: u32,
        stage: &str,
        precision: u32,
    ) -> HiddenRangeStats {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_first_block_stage_for_token_at_precision: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));

        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        if stage == "pre_attn_norm" {
            return self.hidden_range_stats_at_precision(stage, &normed_pre_attn, precision);
        }

        let attn_out = _FHE_LLM_multi_head_attention_vec(
            &self.module,
            &self.eval_key,
            &normed_pre_attn,
            &weights.attention,
            &block_config.attention,
        );
        if stage == "attn_out" {
            return self.hidden_range_stats_at_precision(stage, &attn_out, precision);
        }

        let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);
        if stage == "residual_1" {
            return self.hidden_range_stats_at_precision(stage, &residual_1, precision);
        }

        let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
            Some(gamma) => LayerNormConfig::rms_norm_midrange(d).with_gamma(gamma.clone()),
            None => LayerNormConfig::rms_norm_midrange(d),
        };
        let normed_pre_ffn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &residual_1, &pre_ffn_norm_cfg);
        if stage == "pre_ffn_norm" {
            return self.hidden_range_stats_at_precision(stage, &normed_pre_ffn, precision);
        }

        let ffn_out = _FHE_LLM_ffn_vec_scaled(
            &self.module,
            &self.eval_key,
            &normed_pre_ffn,
            &weights.ffn,
            &block_config.ffn,
            crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
        );
        if stage == "ffn_out" {
            return self.hidden_range_stats_at_precision(stage, &ffn_out, precision);
        }

        let block_out = self.project_and_add_diag(&residual_1, &ffn_out);
        if stage == "block_out" {
            return self.hidden_range_stats_at_precision(stage, &block_out, precision);
        }

        panic!("unknown stage: {stage}");
    }

    pub fn diagnose_pre_attn_norm_range_for_token_at_precision(&self, token_id: u32, precision: u32) -> HiddenRangeStats {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_pre_attn_norm_range_for_token_at_precision: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        self.hidden_range_stats_at_precision("pre_attn_norm", &normed_pre_attn, precision)
    }

    pub fn diagnose_pre_attn_rms_internals_for_token(&self, token_id: u32, precision: u32) -> Vec<ScalarStageValue> {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_pre_attn_rms_internals_for_token: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };

        let debug = _FHE_LLM_rms_norm_vec_debug_stages(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        let decode_one = |ct: &GLWE<Vec<u8>>| -> i64 {
            let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
            let mut decoded = vec![0i64; self.module.n() as usize];
            pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
            decoded[0]
        };

        vec![
            ScalarStageValue {
                stage: format!("sum_sq@{precision}"),
                value: decode_one(&debug.sum_sq),
            },
            ScalarStageValue {
                stage: format!("mean_sq@{precision}"),
                value: decode_one(&debug.mean_sq),
            },
            ScalarStageValue {
                stage: format!("mean_sq_for_poly@{precision}"),
                value: decode_one(&debug.mean_sq_for_poly),
            },
            ScalarStageValue {
                stage: format!("inv_rms_pre@{precision}"),
                value: decode_one(&debug.inv_rms_pre),
            },
            ScalarStageValue {
                stage: format!("inv_rms@{precision}"),
                value: decode_one(&debug.inv_rms),
            },
        ]
    }

    pub fn diagnose_pre_ffn_rms_internals_for_token(&self, token_id: u32, precision: u32) -> Vec<ScalarStageValue> {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_pre_ffn_rms_internals_for_token: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        let attn_out = _FHE_LLM_multi_head_attention_vec(
            &self.module,
            &self.eval_key,
            &normed_pre_attn,
            &weights.attention,
            &block_config.attention,
        );
        let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);

        let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
            Some(gamma) => LayerNormConfig::rms_norm_midrange(d).with_gamma(gamma.clone()),
            None => LayerNormConfig::rms_norm_midrange(d),
        };

        let debug = _FHE_LLM_rms_norm_vec_debug_stages(&self.module, &self.eval_key, &residual_1, &pre_ffn_norm_cfg);
        let decode_one = |ct: &GLWE<Vec<u8>>| -> i64 {
            let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
            let mut decoded = vec![0i64; self.module.n() as usize];
            pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
            decoded[0]
        };

        vec![
            ScalarStageValue {
                stage: format!("pre_ffn_sum_sq@{precision}"),
                value: decode_one(&debug.sum_sq),
            },
            ScalarStageValue {
                stage: format!("pre_ffn_mean_sq@{precision}"),
                value: decode_one(&debug.mean_sq),
            },
            ScalarStageValue {
                stage: format!("pre_ffn_mean_sq_for_poly@{precision}"),
                value: decode_one(&debug.mean_sq_for_poly),
            },
            ScalarStageValue {
                stage: format!("pre_ffn_inv_rms_pre@{precision}"),
                value: decode_one(&debug.inv_rms_pre),
            },
            ScalarStageValue {
                stage: format!("pre_ffn_inv_rms@{precision}"),
                value: decode_one(&debug.inv_rms),
            },
        ]
    }

    pub fn compare_first_block_stages_quantized(&self, token_id: u32) -> Vec<StageErrorStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "compare_first_block_stages_quantized: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let embedding_i8 = self.embed_token(token_id);
        let embedding_f64: Vec<f64> = embedding_i8.iter().map(|&v| v as f64).collect();
        let weights = &self.layer_weights[0];

        let pre_attn_plain = crate::plaintext_forward::rms_norm(
            &embedding_f64,
            weights.pre_attn_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
        );
        let attn_plain =
            crate::plaintext_forward::multi_head_attention(&pre_attn_plain, &weights.attention, &block_config.attention);
        let residual_1_plain: Vec<f64> = embedding_f64.iter().zip(attn_plain.iter()).map(|(a, b)| a + b).collect();
        let residual_1_plain_refreshed: Vec<f64> = residual_1_plain.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        let pre_ffn_plain = crate::plaintext_forward::rms_norm(
            &residual_1_plain_refreshed,
            weights.pre_ffn_norm_gamma.as_deref(),
            block_config.pre_ffn_norm.epsilon,
        );
        let ffn_plain = crate::plaintext_forward::ffn(&pre_ffn_plain, &weights.ffn, &block_config.ffn);
        let block_out_plain: Vec<f64> = residual_1_plain_refreshed
            .iter()
            .zip(ffn_plain.iter())
            .map(|(a, b)| a + b)
            .collect();

        let quantize = |vals: &[f64]| -> Vec<f64> { vals.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect() };

        let stage_names = [
            "pre_attn_norm",
            "attn_out",
            "residual_1",
            "pre_ffn_norm",
            "ffn_out",
            "block_out",
        ];
        let stage_plain = [
            quantize(&pre_attn_plain),
            quantize(&attn_plain),
            quantize(&residual_1_plain_refreshed),
            quantize(&pre_ffn_plain),
            quantize(&ffn_plain),
            quantize(&block_out_plain),
        ];

        stage_names
            .iter()
            .zip(stage_plain.iter())
            .map(|(name, plain)| {
                let fhe = self.diagnose_first_block_stage_for_token_at_precision(token_id, name, VEC_EFFECTIVE_DECODE_SCALE);
                let fhe_vals = self.decode_first_block_stage_for_token(token_id, name, VEC_EFFECTIVE_DECODE_SCALE);
                let (linf, l2, mae) = crate::plaintext_forward::error_metrics(&fhe_vals, plain);
                let _ = fhe;
                StageErrorStats {
                    stage: (*name).to_string(),
                    linf,
                    l2,
                    mae,
                }
            })
            .collect()
    }

    pub fn compare_first_block_stages_quantized_with_residual_refresh(&self, token_id: u32) -> Vec<StageErrorStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "compare_first_block_stages_quantized_with_residual_refresh: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let embedding_i8 = self.embed_token(token_id);
        let embedding_f64: Vec<f64> = embedding_i8.iter().map(|&v| v as f64).collect();
        let weights = &self.layer_weights[0];

        let pre_attn_plain = crate::plaintext_forward::rms_norm(
            &embedding_f64,
            weights.pre_attn_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
        );
        let attn_plain =
            crate::plaintext_forward::multi_head_attention(&pre_attn_plain, &weights.attention, &block_config.attention);
        let residual_1_plain: Vec<f64> = embedding_f64.iter().zip(attn_plain.iter()).map(|(a, b)| a + b).collect();
        let residual_1_plain_refreshed: Vec<f64> = residual_1_plain.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        let pre_ffn_plain = crate::plaintext_forward::rms_norm(
            &residual_1_plain_refreshed,
            weights.pre_ffn_norm_gamma.as_deref(),
            block_config.pre_ffn_norm.epsilon,
        );
        let ffn_plain = crate::plaintext_forward::ffn(&pre_ffn_plain, &weights.ffn, &block_config.ffn);
        let block_out_plain: Vec<f64> = residual_1_plain.iter().zip(ffn_plain.iter()).map(|(a, b)| a + b).collect();

        let quantize = |vals: &[f64]| -> Vec<f64> { vals.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect() };

        let encrypted = self.encrypt_embedding(&embedding_i8);
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        let attn_out = _FHE_LLM_multi_head_attention_vec(
            &self.module,
            &self.eval_key,
            &normed_pre_attn,
            &weights.attention,
            &block_config.attention,
        );
        let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);
        let residual_1_refreshed = self.refresh_vec_at_effective_scale(&residual_1);
        let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
            Some(gamma) => LayerNormConfig::rms_norm_midrange(d).with_gamma(gamma.clone()),
            None => LayerNormConfig::rms_norm_midrange(d),
        };
        let normed_pre_ffn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &residual_1_refreshed, &pre_ffn_norm_cfg);
        let ffn_out = _FHE_LLM_ffn_vec_scaled(
            &self.module,
            &self.eval_key,
            &normed_pre_ffn,
            &weights.ffn,
            &block_config.ffn,
            crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
        );
        let block_out = self.project_and_add_diag(&residual_1_refreshed, &ffn_out);

        let fhe_stage_vals = [
            self.decode_ct_vec_at_effective_scale(&normed_pre_attn),
            self.decode_ct_vec_at_effective_scale(&attn_out),
            self.decode_ct_vec_at_effective_scale(&residual_1_refreshed),
            self.decode_ct_vec_at_effective_scale(&normed_pre_ffn),
            self.decode_ct_vec_at_effective_scale(&ffn_out),
            self.decode_ct_vec_at_effective_scale(&block_out),
        ];

        let stage_names = [
            "pre_attn_norm",
            "attn_out",
            "residual_1_refreshed",
            "pre_ffn_norm",
            "ffn_out",
            "block_out",
        ];
        let stage_plain = [
            quantize(&pre_attn_plain),
            quantize(&attn_plain),
            quantize(&residual_1_plain),
            quantize(&pre_ffn_plain),
            quantize(&ffn_plain),
            quantize(&block_out_plain),
        ];

        stage_names
            .iter()
            .zip(fhe_stage_vals.iter())
            .zip(stage_plain.iter())
            .map(|((name, fhe), plain)| {
                let (linf, l2, mae) = crate::plaintext_forward::error_metrics(fhe, plain);
                StageErrorStats {
                    stage: (*name).to_string(),
                    linf,
                    l2,
                    mae,
                }
            })
            .collect()
    }

    fn decode_ct_vec_at_effective_scale(&self, cts: &[GLWE<Vec<u8>>]) -> Vec<f64> {
        self.decode_ct_vec_at_precision(cts, VEC_EFFECTIVE_DECODE_SCALE)
    }

    fn decode_ct_vec_at_precision(&self, cts: &[GLWE<Vec<u8>>], precision: u32) -> Vec<f64> {
        cts.iter()
            .map(|ct| {
                let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
                let mut decoded = vec![0i64; self.module.n() as usize];
                pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
                decoded[0] as f64
            })
            .collect()
    }

    fn apply_silu_lut_vec(&self, cts: &[GLWE<Vec<u8>>]) -> Vec<GLWE<Vec<u8>>> {
        let bsk = self
            .bsk_prepared
            .as_ref()
            .expect("apply_silu_lut_vec: bootstrap key not available");
        let bp = &bsk.bootstrap_params;
        let lut_entries = crate::lut::NonlinearLUT::silu_message_lut(bp.log_message_modulus);
        cts.iter()
            .map(|ct| {
                let mut tracker = NoiseTracker::fresh();
                crate::bootstrapping::_FHE_LLM_bootstrap_with_lut(&self.module, ct, &mut tracker, bsk, &lut_entries)
            })
            .collect()
    }

    pub fn compare_ffn_substages_with_residual_refresh(&self, token_id: u32) -> Vec<StageErrorStats> {
        self.compare_ffn_substages_with_residual_refresh_for_shift(token_id, crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS)
    }

    pub fn compare_ffn_substages_with_residual_refresh_for_shift(
        &self,
        token_id: u32,
        input_scale_shift_bits: usize,
    ) -> Vec<StageErrorStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "compare_ffn_substages_with_residual_refresh: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let embedding_i8 = self.embed_token(token_id);
        let embedding_f64: Vec<f64> = embedding_i8.iter().map(|&v| v as f64).collect();
        let weights = &self.layer_weights[0];
        let w_up = weights.ffn.w3.as_ref().expect("SwiGLU requires w3");

        let pre_attn_plain = crate::plaintext_forward::rms_norm(
            &embedding_f64,
            weights.pre_attn_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
        );
        let attn_plain =
            crate::plaintext_forward::multi_head_attention(&pre_attn_plain, &weights.attention, &block_config.attention);
        let residual_1_plain: Vec<f64> = embedding_f64.iter().zip(attn_plain.iter()).map(|(a, b)| a + b).collect();
        let pre_ffn_plain = crate::plaintext_forward::rms_norm(
            &residual_1_plain,
            weights.pre_ffn_norm_gamma.as_deref(),
            block_config.pre_ffn_norm.epsilon,
        );
        let gate_plain = crate::plaintext_forward::matvec(&weights.ffn.w1, &pre_ffn_plain);
        let up_plain = crate::plaintext_forward::matvec(w_up, &pre_ffn_plain);
        let gate_act_plain: Vec<f64> = gate_plain.iter().map(|&v| crate::plaintext_forward::poly_silu(v)).collect();
        let hidden_plain: Vec<f64> = gate_act_plain.iter().zip(up_plain.iter()).map(|(a, b)| a * b).collect();
        let down_plain = crate::plaintext_forward::matvec(&weights.ffn.w2, &hidden_plain);

        let quantize = |vals: &[f64]| -> Vec<f64> { vals.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect() };

        let encrypted = self.encrypt_embedding(&embedding_i8);
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        let attn_out = _FHE_LLM_multi_head_attention_vec(
            &self.module,
            &self.eval_key,
            &normed_pre_attn,
            &weights.attention,
            &block_config.attention,
        );
        let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);
        let residual_1_refreshed = self.refresh_vec_at_effective_scale(&residual_1);
        let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
            Some(gamma) => LayerNormConfig::rms_norm_midrange(d).with_gamma(gamma.clone()),
            None => LayerNormConfig::rms_norm_midrange(d),
        };
        let normed_pre_ffn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &residual_1_refreshed, &pre_ffn_norm_cfg);

        let d_in = normed_pre_ffn.len();
        let d_ffn = weights.ffn.w1.len();
        let silu_approx = crate::activations::silu_poly_approx_narrow();

        let gate_fhe: Vec<GLWE<Vec<u8>>> = (0..d_ffn)
            .map(|j| {
                let wg_vecs: Vec<Vec<i64>> = weights.ffn.w1[j][..d_in].iter().map(|&w| vec![w]).collect();
                if input_scale_shift_bits == 0 {
                    crate::arithmetic::_FHE_LLM_dot_product(&self.module, &normed_pre_ffn, &wg_vecs)
                } else {
                    crate::arithmetic::_FHE_LLM_dot_product_scaled(&self.module, &normed_pre_ffn, &wg_vecs, input_scale_shift_bits)
                }
            })
            .collect();

        let up_fhe: Vec<GLWE<Vec<u8>>> = (0..d_ffn)
            .map(|j| {
                let wu_vecs: Vec<Vec<i64>> = w_up[j][..d_in].iter().map(|&w| vec![w]).collect();
                if input_scale_shift_bits == 0 {
                    crate::arithmetic::_FHE_LLM_dot_product(&self.module, &normed_pre_ffn, &wu_vecs)
                } else {
                    crate::arithmetic::_FHE_LLM_dot_product_scaled(&self.module, &normed_pre_ffn, &wu_vecs, input_scale_shift_bits)
                }
            })
            .collect();

        let gate_act_fhe: Vec<GLWE<Vec<u8>>> = gate_fhe
            .iter()
            .map(|ct| crate::activations::apply_poly_activation(&self.module, &self.eval_key, ct, &silu_approx))
            .collect();

        let hidden_fhe: Vec<GLWE<Vec<u8>>> = gate_act_fhe
            .iter()
            .zip(up_fhe.iter())
            .map(|(gate, up)| {
                let gate_layout = poulpy_core::layouts::GLWELayout {
                    n: gate.n(),
                    base2k: gate.base2k(),
                    k: gate.k(),
                    rank: gate.rank(),
                };
                let up_proj = if up.base2k() == gate.base2k() {
                    up.clone()
                } else {
                    crate::arithmetic::_FHE_LLM_align_layout(&self.module, up, &gate_layout)
                };
                if input_scale_shift_bits == 0 {
                    crate::activations::_FHE_LLM_ct_mul(&self.module, &self.eval_key, gate, &up_proj)
                } else {
                    crate::activations::_FHE_LLM_ct_mul_with_res_offset(
                        &self.module,
                        &self.eval_key,
                        gate,
                        &up_proj,
                        self.eval_key.res_offset.saturating_sub(input_scale_shift_bits),
                    )
                }
            })
            .collect();

        let down_fhe: Vec<GLWE<Vec<u8>>> = weights
            .ffn
            .w2
            .iter()
            .map(|row| {
                let w2_vecs: Vec<Vec<i64>> = row.iter().map(|&w| vec![w]).collect();
                crate::arithmetic::_FHE_LLM_dot_product_scaled(
                    &self.module,
                    &hidden_fhe,
                    &w2_vecs,
                    crate::activations::COEFF_SCALE_BITS,
                )
            })
            .collect();

        let fhe_stage_vals = [
            self.decode_ct_vec_at_effective_scale(&normed_pre_ffn),
            self.decode_ct_vec_at_effective_scale(&gate_fhe),
            self.decode_ct_vec_at_effective_scale(&gate_act_fhe),
            self.decode_ct_vec_at_effective_scale(&up_fhe),
            self.decode_ct_vec_at_effective_scale(&hidden_fhe),
            self.decode_ct_vec_at_effective_scale(&down_fhe),
        ];

        let stage_names = ["pre_ffn_norm", "gate", "gate_activated", "up", "hidden", "down"];
        let stage_plain = [
            quantize(&pre_ffn_plain),
            quantize(&gate_plain),
            quantize(&gate_act_plain),
            quantize(&up_plain),
            quantize(&hidden_plain),
            quantize(&down_plain),
        ];

        stage_names
            .iter()
            .zip(fhe_stage_vals.iter())
            .zip(stage_plain.iter())
            .map(|((name, fhe), plain)| {
                let (linf, l2, mae) = crate::plaintext_forward::error_metrics(fhe, plain);
                StageErrorStats {
                    stage: (*name).to_string(),
                    linf,
                    l2,
                    mae,
                }
            })
            .collect()
    }

    pub fn compare_ffn_substages_with_residual_refresh_lut_gate(&self, token_id: u32) -> Vec<StageErrorStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "compare_ffn_substages_with_residual_refresh_lut_gate: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let embedding_i8 = self.embed_token(token_id);
        let embedding_f64: Vec<f64> = embedding_i8.iter().map(|&v| v as f64).collect();
        let weights = &self.layer_weights[0];
        let w_up = weights.ffn.w3.as_ref().expect("SwiGLU requires w3");

        let pre_attn_plain = crate::plaintext_forward::rms_norm(
            &embedding_f64,
            weights.pre_attn_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
        );
        let attn_plain =
            crate::plaintext_forward::multi_head_attention(&pre_attn_plain, &weights.attention, &block_config.attention);
        let residual_1_plain: Vec<f64> = embedding_f64.iter().zip(attn_plain.iter()).map(|(a, b)| a + b).collect();
        let residual_1_plain_refreshed: Vec<f64> = residual_1_plain.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        let pre_ffn_plain = crate::plaintext_forward::rms_norm(
            &residual_1_plain_refreshed,
            weights.pre_ffn_norm_gamma.as_deref(),
            block_config.pre_ffn_norm.epsilon,
        );
        let gate_plain = crate::plaintext_forward::matvec(&weights.ffn.w1, &pre_ffn_plain);
        let up_plain = crate::plaintext_forward::matvec(w_up, &pre_ffn_plain);
        let gate_act_plain: Vec<f64> = gate_plain.iter().map(|&v| crate::plaintext_forward::exact_silu(v)).collect();
        let hidden_plain: Vec<f64> = gate_act_plain.iter().zip(up_plain.iter()).map(|(a, b)| a * b).collect();
        let hidden_plain_refreshed: Vec<f64> = hidden_plain.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        let down_plain = crate::plaintext_forward::matvec(&weights.ffn.w2, &hidden_plain_refreshed);

        let quantize = |vals: &[f64]| -> Vec<f64> { vals.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect() };

        let encrypted = self.encrypt_embedding(&embedding_i8);
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        let attn_out = _FHE_LLM_multi_head_attention_vec(
            &self.module,
            &self.eval_key,
            &normed_pre_attn,
            &weights.attention,
            &block_config.attention,
        );
        let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);
        let residual_1_refreshed = self.refresh_vec_at_effective_scale(&residual_1);
        let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
            Some(gamma) => LayerNormConfig::rms_norm_midrange(d).with_gamma(gamma.clone()),
            None => LayerNormConfig::rms_norm_midrange(d),
        };
        let normed_pre_ffn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &residual_1_refreshed, &pre_ffn_norm_cfg);

        let d_in = normed_pre_ffn.len();
        let d_ffn = weights.ffn.w1.len();
        let gate_fhe: Vec<GLWE<Vec<u8>>> = (0..d_ffn)
            .map(|j| {
                let wg_vecs: Vec<Vec<i64>> = weights.ffn.w1[j][..d_in].iter().map(|&w| vec![w]).collect();
                crate::arithmetic::_FHE_LLM_dot_product_scaled(
                    &self.module,
                    &normed_pre_ffn,
                    &wg_vecs,
                    crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
                )
            })
            .collect();
        let gate_act_fhe = self.apply_silu_lut_vec(&gate_fhe);

        let up_fhe: Vec<GLWE<Vec<u8>>> = (0..d_ffn)
            .map(|j| {
                let wu_vecs: Vec<Vec<i64>> = w_up[j][..d_in].iter().map(|&w| vec![w]).collect();
                crate::arithmetic::_FHE_LLM_dot_product_scaled(
                    &self.module,
                    &normed_pre_ffn,
                    &wu_vecs,
                    crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
                )
            })
            .collect();

        let hidden_fhe: Vec<GLWE<Vec<u8>>> = gate_act_fhe
            .iter()
            .zip(up_fhe.iter())
            .map(|(gate, up)| {
                let gate_layout = poulpy_core::layouts::GLWELayout {
                    n: gate.n(),
                    base2k: gate.base2k(),
                    k: gate.k(),
                    rank: gate.rank(),
                };
                let up_proj = if up.base2k() == gate.base2k() {
                    up.clone()
                } else {
                    crate::arithmetic::_FHE_LLM_align_layout(&self.module, up, &gate_layout)
                };
                crate::activations::_FHE_LLM_ct_mul(&self.module, &self.eval_key, gate, &up_proj)
            })
            .collect();

        let hidden_fhe_refreshed = self.refresh_vec_at_effective_scale(&hidden_fhe);

        let down_fhe: Vec<GLWE<Vec<u8>>> = weights
            .ffn
            .w2
            .iter()
            .map(|row| {
                let w2_vecs: Vec<Vec<i64>> = row.iter().map(|&w| vec![w]).collect();
                crate::arithmetic::_FHE_LLM_dot_product(&self.module, &hidden_fhe_refreshed, &w2_vecs)
            })
            .collect();

        let fhe_stage_vals = [
            self.decode_ct_vec_at_effective_scale(&normed_pre_ffn),
            self.decode_ct_vec_at_effective_scale(&gate_fhe),
            self.decode_ct_vec_at_effective_scale(&gate_act_fhe),
            self.decode_ct_vec_at_effective_scale(&up_fhe),
            self.decode_ct_vec_at_effective_scale(&hidden_fhe_refreshed),
            self.decode_ct_vec_at_effective_scale(&down_fhe),
        ];

        let stage_names = ["pre_ffn_norm", "gate", "gate_activated_lut", "up", "hidden", "down"];
        let stage_plain = [
            quantize(&pre_ffn_plain),
            quantize(&gate_plain),
            quantize(&gate_act_plain),
            quantize(&up_plain),
            quantize(&hidden_plain_refreshed),
            quantize(&down_plain),
        ];

        stage_names
            .iter()
            .zip(fhe_stage_vals.iter())
            .zip(stage_plain.iter())
            .map(|((name, fhe), plain)| {
                let (linf, l2, mae) = crate::plaintext_forward::error_metrics(fhe, plain);
                StageErrorStats {
                    stage: (*name).to_string(),
                    linf,
                    l2,
                    mae,
                }
            })
            .collect()
    }

    pub fn diagnose_pre_ffn_rms_internals_with_residual_refresh(&self, token_id: u32, precision: u32) -> Vec<ScalarStageValue> {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_pre_ffn_rms_internals_with_residual_refresh: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        let attn_out = _FHE_LLM_multi_head_attention_vec(
            &self.module,
            &self.eval_key,
            &normed_pre_attn,
            &weights.attention,
            &block_config.attention,
        );
        let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);
        let residual_1_refreshed = self.refresh_vec_at_effective_scale(&residual_1);

        let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
            Some(gamma) => LayerNormConfig::rms_norm_midrange(d).with_gamma(gamma.clone()),
            None => LayerNormConfig::rms_norm_midrange(d),
        };

        let debug = _FHE_LLM_rms_norm_vec_debug_stages(&self.module, &self.eval_key, &residual_1_refreshed, &pre_ffn_norm_cfg);
        let decode_one = |ct: &GLWE<Vec<u8>>| -> i64 {
            let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
            let mut decoded = vec![0i64; self.module.n() as usize];
            pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
            decoded[0]
        };

        vec![
            ScalarStageValue {
                stage: format!("refresh_pre_ffn_sum_sq@{precision}"),
                value: decode_one(&debug.sum_sq),
            },
            ScalarStageValue {
                stage: format!("refresh_pre_ffn_mean_sq@{precision}"),
                value: decode_one(&debug.mean_sq),
            },
            ScalarStageValue {
                stage: format!("refresh_pre_ffn_mean_sq_for_poly@{precision}"),
                value: decode_one(&debug.mean_sq_for_poly),
            },
            ScalarStageValue {
                stage: format!("refresh_pre_ffn_inv_rms_pre@{precision}"),
                value: decode_one(&debug.inv_rms_pre),
            },
            ScalarStageValue {
                stage: format!("refresh_pre_ffn_inv_rms@{precision}"),
                value: decode_one(&debug.inv_rms),
            },
        ]
    }

    fn decode_first_block_stage_for_token(&self, token_id: u32, stage: &str, precision: u32) -> Vec<f64> {
        assert!(
            !self.layer_weights.is_empty(),
            "decode_first_block_stage_for_token: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));

        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let normed_pre_attn = _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);
        let target_cts = if stage == "pre_attn_norm" {
            normed_pre_attn
        } else {
            let attn_out = _FHE_LLM_multi_head_attention_vec(
                &self.module,
                &self.eval_key,
                &normed_pre_attn,
                &weights.attention,
                &block_config.attention,
            );
            if stage == "attn_out" {
                attn_out
            } else {
                let residual_1 = self.project_and_add_diag(&encrypted, &attn_out);
                if stage == "residual_1" {
                    residual_1
                } else {
                    let pre_ffn_norm_cfg = match &weights.pre_ffn_norm_gamma {
                        Some(gamma) => block_config.pre_ffn_norm.clone().with_gamma(gamma.clone()),
                        None => block_config.pre_ffn_norm.clone(),
                    };
                    let residual_1_for_norm: Vec<_> = residual_1
                        .iter()
                        .map(|ct| crate::arithmetic::_FHE_LLM_mul_const_scaled(&self.module, ct, &[1], 1))
                        .collect();
                    let normed_pre_ffn =
                        _FHE_LLM_rms_norm_vec(&self.module, &self.eval_key, &residual_1_for_norm, &pre_ffn_norm_cfg);
                    if stage == "pre_ffn_norm" {
                        normed_pre_ffn
                    } else {
                        let ffn_out = _FHE_LLM_ffn_vec_scaled(
                            &self.module,
                            &self.eval_key,
                            &normed_pre_ffn,
                            &weights.ffn,
                            &block_config.ffn,
                            crate::layernorm::RMS_OUTPUT_SCALE_SHIFT_BITS,
                        );
                        if stage == "ffn_out" {
                            ffn_out
                        } else if stage == "block_out" {
                            self.project_and_add_diag(&residual_1, &ffn_out)
                        } else {
                            panic!("unknown stage: {stage}");
                        }
                    }
                }
            }
        };

        target_cts
            .iter()
            .map(|ct| {
                let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
                let mut decoded = vec![0i64; self.module.n() as usize];
                pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
                decoded[0] as f64
            })
            .collect()
    }

    pub fn diagnose_pre_attn_norm_variants_for_token(&self, token_id: u32) -> Vec<VariantRangeStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_pre_attn_norm_variants_for_token: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let weights = &self.layer_weights[0];
        let encrypted = self.encrypt_embedding(&self.embed_token(token_id));
        let pre_attn_norm_cfg = match &weights.pre_attn_norm_gamma {
            Some(gamma) => block_config.pre_attn_norm.clone().with_gamma(gamma.clone()),
            None => block_config.pre_attn_norm.clone(),
        };
        let debug = _FHE_LLM_rms_norm_vec_debug_stages(&self.module, &self.eval_key, &encrypted, &pre_attn_norm_cfg);

        let inv_layout = poulpy_core::layouts::GLWELayout {
            n: debug.inv_rms_pre.n(),
            base2k: debug.inv_rms_pre.base2k(),
            k: debug.inv_rms_pre.k(),
            rank: debug.inv_rms_pre.rank(),
        };

        let apply_variant = |label: &str, inv_ct: &GLWE<Vec<u8>>, res_offset: usize| -> Vec<GLWE<Vec<u8>>> {
            let mut outputs = Vec::with_capacity(encrypted.len());
            for (i, xi) in encrypted.iter().enumerate() {
                let xi_proj = if xi.base2k() == inv_ct.base2k() && xi.k() == inv_ct.k() {
                    xi.clone()
                } else {
                    _FHE_LLM_align_layout(&self.module, xi, &inv_layout)
                };
                let mut out = crate::activations::_FHE_LLM_ct_mul_with_res_offset(
                    &self.module,
                    &self.eval_key,
                    &xi_proj,
                    inv_ct,
                    res_offset,
                );
                if let Some(gamma) = &pre_attn_norm_cfg.gamma {
                    if i < gamma.len() {
                        out = crate::arithmetic::_FHE_LLM_mul_const_scaled(
                            &self.module,
                            &out,
                            &[gamma[i]],
                            crate::activations::COEFF_SCALE_BITS,
                        );
                    }
                }
                outputs.push(out);
            }
            let _ = label;
            outputs
        };

        let variants = vec![
            (
                "current_inv_comp_res2",
                apply_variant("current_inv_comp_res2", &debug.inv_rms, 2),
            ),
            (
                "current_inv_comp_res10",
                apply_variant("current_inv_comp_res10", &debug.inv_rms, 10),
            ),
            (
                "current_inv_comp_res18",
                apply_variant("current_inv_comp_res18", &debug.inv_rms, 18),
            ),
            (
                "current_inv_comp_res26",
                apply_variant("current_inv_comp_res26", &debug.inv_rms, 26),
            ),
            ("inv_pre_res2", apply_variant("inv_pre_res2", &debug.inv_rms_pre, 2)),
            ("inv_pre_res10", apply_variant("inv_pre_res10", &debug.inv_rms_pre, 10)),
            ("inv_pre_res18", apply_variant("inv_pre_res18", &debug.inv_rms_pre, 18)),
        ];

        let mut stats = Vec::new();
        for (name, cts) in variants {
            for precision in [26u32, 18u32, 10u32, 8u32, 4u32, 2u32] {
                let raw: Vec<i64> = cts
                    .iter()
                    .map(|ct| {
                        let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
                        let mut decoded = vec![0i64; self.module.n() as usize];
                        pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
                        decoded[0]
                    })
                    .collect();
                stats.push(VariantRangeStats {
                    variant: name.to_string(),
                    decode_precision: precision,
                    min: raw.iter().copied().min().unwrap_or(0),
                    max: raw.iter().copied().max().unwrap_or(0),
                    overflow_dims: raw.iter().filter(|&&v| !(-128..=127).contains(&v)).count(),
                    total_dims: raw.len(),
                });
            }
        }

        stats
    }

    pub fn diagnose_plaintext_rms_ranges_for_token(&self, token_id: u32) -> Vec<PlaintextRmsStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_plaintext_rms_ranges_for_token: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let x_i8 = self.embed_token(token_id);
        let x: Vec<f64> = x_i8.iter().map(|&v| v as f64).collect();
        let weights = &self.layer_weights[0];

        let mean_sq = |vals: &[f64]| -> f64 { vals.iter().map(|v| v * v).sum::<f64>() / vals.len() as f64 };

        let mut stats = vec![PlaintextRmsStats {
            stage: "input".to_string(),
            mean_sq: mean_sq(&x),
        }];

        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let normed_pre_attn =
            crate::plaintext_forward::rms_norm(&x, weights.pre_attn_norm_gamma.as_deref(), block_config.pre_attn_norm.epsilon);
        stats.push(PlaintextRmsStats {
            stage: "pre_attn_norm".to_string(),
            mean_sq: mean_sq(&normed_pre_attn),
        });

        let attn_out =
            crate::plaintext_forward::multi_head_attention(&normed_pre_attn, &weights.attention, &block_config.attention);
        let residual_1: Vec<f64> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
        stats.push(PlaintextRmsStats {
            stage: "residual_1".to_string(),
            mean_sq: mean_sq(&residual_1),
        });

        let normed_pre_ffn = crate::plaintext_forward::rms_norm(
            &residual_1,
            weights.pre_ffn_norm_gamma.as_deref(),
            block_config.pre_ffn_norm.epsilon,
        );
        stats.push(PlaintextRmsStats {
            stage: "pre_ffn_norm".to_string(),
            mean_sq: mean_sq(&normed_pre_ffn),
        });

        stats
    }

    pub fn diagnose_refreshed_plaintext_rms_ranges_for_token(&self, token_id: u32) -> Vec<PlaintextRmsStats> {
        assert!(
            !self.layer_weights.is_empty(),
            "diagnose_refreshed_plaintext_rms_ranges_for_token: no layers loaded"
        );

        let d = self.effective_dims.d_model;
        let x_i8 = self.embed_token(token_id);
        let x: Vec<f64> = x_i8.iter().map(|&v| v as f64).collect();
        let weights = &self.layer_weights[0];
        let mean_sq = |vals: &[f64]| -> f64 { vals.iter().map(|v| v * v).sum::<f64>() / vals.len() as f64 };

        let mut stats = vec![PlaintextRmsStats {
            stage: "input".to_string(),
            mean_sq: mean_sq(&x),
        }];

        let block_config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: self.effective_dims.clone(),
                params: self.params.clone(),
                softmax_approx: self.config.softmax_strategy.clone(),
                causal: true,
                rope: None,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(d),
            pre_ffn_norm: LayerNormConfig::rms_norm_midrange(d),
            ffn: FFNConfig::SwiGLU,
            residual: true,
        };

        let normed_pre_attn =
            crate::plaintext_forward::rms_norm(&x, weights.pre_attn_norm_gamma.as_deref(), block_config.pre_attn_norm.epsilon);
        stats.push(PlaintextRmsStats {
            stage: "pre_attn_norm".to_string(),
            mean_sq: mean_sq(&normed_pre_attn),
        });

        let attn_out =
            crate::plaintext_forward::multi_head_attention(&normed_pre_attn, &weights.attention, &block_config.attention);
        let residual_1: Vec<f64> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();
        let residual_1_refreshed: Vec<f64> = residual_1.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        stats.push(PlaintextRmsStats {
            stage: "residual_1_refreshed".to_string(),
            mean_sq: mean_sq(&residual_1_refreshed),
        });

        let normed_pre_ffn = crate::plaintext_forward::rms_norm(
            &residual_1_refreshed,
            weights.pre_ffn_norm_gamma.as_deref(),
            block_config.pre_ffn_norm.epsilon,
        );
        stats.push(PlaintextRmsStats {
            stage: "pre_ffn_norm_refreshed".to_string(),
            mean_sq: mean_sq(&normed_pre_ffn),
        });

        let w_up = weights.ffn.w3.as_ref().expect("SwiGLU requires w3");
        let gate = crate::plaintext_forward::matvec(&weights.ffn.w1, &normed_pre_ffn);
        let gate_activated: Vec<f64> = gate.iter().map(|&v| crate::plaintext_forward::exact_silu(v)).collect();
        let up = crate::plaintext_forward::matvec(w_up, &normed_pre_ffn);
        let hidden: Vec<f64> = gate_activated.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
        let hidden_refreshed: Vec<f64> = hidden.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        stats.push(PlaintextRmsStats {
            stage: "hidden_refreshed".to_string(),
            mean_sq: mean_sq(&hidden_refreshed),
        });

        let down = crate::plaintext_forward::matvec(&weights.ffn.w2, &hidden_refreshed);
        let block_out: Vec<f64> = residual_1_refreshed.iter().zip(down.iter()).map(|(a, b)| a + b).collect();
        let block_out_refreshed: Vec<f64> = block_out.iter().map(|&v| v.round().clamp(-128.0, 127.0)).collect();
        stats.push(PlaintextRmsStats {
            stage: "block_out_refreshed".to_string(),
            mean_sq: mean_sq(&block_out_refreshed),
        });

        if let Some(gamma) = self.final_norm_gamma.as_deref() {
            let final_plain = crate::plaintext_forward::rms_norm(&block_out_refreshed, Some(gamma), 1e-5);
            stats.push(PlaintextRmsStats {
                stage: "final_norm_refreshed".to_string(),
                mean_sq: mean_sq(&final_plain),
            });
        }

        stats
    }

    fn hidden_range_stats(&self, stage: &str, cts: &[GLWE<Vec<u8>>]) -> HiddenRangeStats {
        let raw = self.decrypt_hidden_state_raw(cts);
        HiddenRangeStats {
            stage: stage.to_string(),
            min: raw.iter().copied().min().unwrap_or(0),
            max: raw.iter().copied().max().unwrap_or(0),
            overflow_dims: raw.iter().filter(|&&v| !(-128..=127).contains(&v)).count(),
            total_dims: raw.len(),
        }
    }

    pub fn hidden_range_stats_at_precision(&self, stage: &str, cts: &[GLWE<Vec<u8>>], precision: u32) -> HiddenRangeStats {
        let raw: Vec<i64> = cts
            .iter()
            .map(|ct| {
                let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
                let mut decoded = vec![0i64; self.module.n() as usize];
                pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
                decoded[0]
            })
            .collect();

        HiddenRangeStats {
            stage: format!("{stage}@{precision}"),
            min: raw.iter().copied().min().unwrap_or(0),
            max: raw.iter().copied().max().unwrap_or(0),
            overflow_dims: raw.iter().filter(|&&v| !(-128..=127).contains(&v)).count(),
            total_dims: raw.len(),
        }
    }

    fn project_and_add_diag(&self, a: &[GLWE<Vec<u8>>], b: &[GLWE<Vec<u8>>]) -> Vec<GLWE<Vec<u8>>> {
        assert_eq!(a.len(), b.len(), "project_and_add_diag: length mismatch");
        let mut result = Vec::with_capacity(a.len());

        for i in 0..a.len() {
            let ai_proj = if a[i].base2k() == b[i].base2k() && a[i].k() == b[i].k() {
                a[i].clone()
            } else {
                let target = poulpy_core::layouts::GLWELayout {
                    n: b[i].n(),
                    base2k: b[i].base2k(),
                    k: b[i].k(),
                    rank: b[i].rank(),
                };
                _FHE_LLM_align_layout(&self.module, &a[i], &target)
            };
            result.push(_FHE_LLM_add(&self.module, &ai_proj, &b[i]));
        }

        result
    }

    fn refresh_vec_at_effective_scale(&self, cts: &[GLWE<Vec<u8>>]) -> Vec<GLWE<Vec<u8>>> {
        self.refresh_vec_at_precision(cts, VEC_EFFECTIVE_DECODE_SCALE)
    }

    fn refresh_vec_at_precision(&self, cts: &[GLWE<Vec<u8>>], precision: u32) -> Vec<GLWE<Vec<u8>>> {
        let vals: Vec<i8> = cts
            .iter()
            .map(|ct| {
                let pt = _FHE_LLM_decrypt(&self.module, &self.key, ct, &self.params);
                let mut decoded = vec![0i64; self.module.n() as usize];
                pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(precision));
                decoded[0].clamp(-128, 127) as i8
            })
            .collect();
        self.encrypt_embedding(&vals)
    }

    /// Applies the LM head to a decrypted hidden state and returns logits.
    ///
    /// If the model was truncated, this uses a truncated LM head
    /// (first `d_model` dimensions only).
    pub fn refresh_with_config(&self, cts: &[GLWE<Vec<u8>>], is_silu: bool) -> Vec<GLWE<Vec<u8>>> {
        let log_msg_mod = if is_silu {
            self.config.fhe_silu_log_msg_mod.unwrap_or(16) as u32
        } else {
            self.config.fhe_identity_log_msg_mod.unwrap_or(7) as u32
        };
        self.refresh_vec_at_precision(cts, log_msg_mod)
    }

    fn lm_head_forward(&self, hidden: &[i8]) -> Vec<i64> {
        let d = self.effective_dims.d_model;
        let hidden_i64: Vec<i64> = hidden.iter().map(|&v| v as i64).collect();

        // If truncated, only use first d dimensions of each LM head row
        if d < self.lm_head.d_model {
            self.lm_head
                .weights
                .iter()
                .map(|w_row| w_row[..d].iter().zip(hidden_i64.iter()).map(|(&w, &h)| w * h).sum())
                .collect()
        } else {
            self.lm_head.forward(&hidden_i64)
        }
    }

    fn apply_cleartext_final_norm_quantized(&self, hidden: &[f64]) -> Vec<i8> {
        if let Some(gamma) = self.final_norm_gamma.as_deref() {
            crate::plaintext_forward::rms_norm(hidden, Some(gamma), 1e-5)
                .iter()
                .map(|&v| v.round().clamp(-128.0, 127.0) as i8)
                .collect()
        } else {
            hidden.iter().map(|&v| v.round().clamp(-128.0, 127.0) as i8).collect()
        }
    }

    /// Returns the top-k token IDs by logit value.
    fn top_k_tokens(&self, logits: &[i64], k: usize) -> Vec<(u32, i64)> {
        let mut indexed: Vec<(u32, i64)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.truncate(k);
        indexed
    }

    /// Runs a single inference step: embed + encrypt + FHE forward + decrypt
    /// + LM head → next token.
    ///
    /// Takes a single token ID (the last token in the sequence) and produces
    /// the next predicted token.
    ///
    /// Note: This is a simplified single-token-at-a-time approach without
    /// KV cache. Each step processes only the last token.
    pub fn step(&self, token_id: u32) -> Result<InferenceStepResult, InferenceError> {
        let total_start = Instant::now();

        // 1. Embed token (cleartext, user-side)
        let embedding = self.embed_token(token_id);
        eprintln!(
            "[inference] Token {} embedded ({} dims, first 4: {:?})",
            token_id,
            embedding.len(),
            &embedding[..4.min(embedding.len())]
        );

        // 2. Encrypt embedding (user-side)
        let encrypt_start = Instant::now();
        let encrypted = self.encrypt_embedding(&embedding);
        eprintln!(
            "[inference] Embedding encrypted: {} cts in {:.2?}",
            encrypted.len(),
            encrypt_start.elapsed()
        );

        // 3. FHE forward pass (provider-side)
        let fhe_start = Instant::now();
        let encrypted_output = self.fhe_forward_refreshed(&encrypted);
        let fhe_time = fhe_start.elapsed();
        eprintln!("[inference] FHE forward pass done in {:.2?}", fhe_time);

        // 4. Decrypt hidden state (user-side)
        let decrypt_start = Instant::now();
        let hidden = self.decode_refreshed_hidden_from_output(&encrypted_output, REFRESHED_EFFECTIVE_DECODE_SCALE);
        eprintln!("[inference] Decrypted in {:.2?}", decrypt_start.elapsed());

        // 5. LM head + argmax (cleartext, user-side)
        let lm_start = Instant::now();
        let logits = self.lm_head_forward(&hidden);
        let top_k = self.top_k_tokens(&logits, 5);
        let next_token_id = top_k[0].0;
        eprintln!(
            "[inference] LM head done in {:.2?}, next_token={}",
            lm_start.elapsed(),
            next_token_id
        );

        // 6. Decode token
        let token_text = self.decode_token(next_token_id)?;
        eprintln!("[inference] Decoded token: {:?}", token_text);

        let total_time = total_start.elapsed();

        // Log top-5 for diagnostics
        for (i, (tid, logit)) in top_k.iter().enumerate() {
            let text = self.decode_token(*tid).unwrap_or_else(|_| "???".to_string());
            eprintln!("[inference]   top-{}: token={} logit={} text={:?}", i + 1, tid, logit, text);
        }

        Ok(InferenceStepResult {
            token_id: next_token_id,
            token_text,
            hidden_state: hidden,
            top_logits: top_k,
            fhe_time,
            total_time,
        })
    }

    /// Generates `max_tokens` new tokens from a text prompt.
    ///
    /// This is the main entry point for end-to-end inference.
    ///
    /// Note: Current implementation is single-token-at-a-time without
    /// KV cache. Each step processes only the last token, which means
    /// the model cannot attend to the full prompt. This is a known
    /// limitation that will be addressed by implementing KV cache support.
    ///
    /// For a more faithful (but much more expensive) approach, all prompt
    /// tokens would need to be processed through attention with proper
    /// position encodings.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<GenerationResult, InferenceError> {
        let gen_start = Instant::now();

        // Tokenize prompt
        let prompt_tokens = self.tokenize(prompt)?;
        eprintln!(
            "[inference] Prompt: {:?} → {} tokens: {:?}",
            prompt,
            prompt_tokens.len(),
            prompt_tokens
        );

        if prompt_tokens.is_empty() {
            return Err(InferenceError::Tokenizer("empty prompt produces no tokens".to_string()));
        }

        // Use the last prompt token as the starting point for generation
        // (simplified: without KV cache, we can only condition on the last token)
        let mut current_token = *prompt_tokens.last().unwrap();
        let mut generated_tokens = Vec::new();
        let mut steps = Vec::new();

        let eos_id = self.model_spec.eos_token_id;

        for step_idx in 0..max_tokens {
            eprintln!("\n[inference] === Generation step {}/{} ===", step_idx + 1, max_tokens);

            let step_result = self.step(current_token)?;
            current_token = step_result.token_id;
            generated_tokens.push(step_result.token_id);
            steps.push(step_result);

            // Stop on EOS
            if current_token == eos_id {
                eprintln!("[inference] EOS token generated, stopping");
                break;
            }
        }

        // Decode all generated tokens
        let generated_text = self.decode(&generated_tokens)?;
        let all_token_ids: Vec<u32> = prompt_tokens.iter().chain(generated_tokens.iter()).copied().collect();
        let full_text = self.decode(&all_token_ids)?;

        let total_time = gen_start.elapsed();

        eprintln!("\n[inference] === Generation complete ===");
        eprintln!("[inference] Prompt: {:?}", prompt);
        eprintln!("[inference] Generated: {:?}", generated_text);
        eprintln!("[inference] Full: {:?}", full_text);
        eprintln!("[inference] Total time: {:.2?}", total_time);
        eprintln!(
            "[inference] Tokens generated: {} ({:.2?} per token)",
            generated_tokens.len(),
            if generated_tokens.is_empty() {
                std::time::Duration::ZERO
            } else {
                total_time / generated_tokens.len() as u32
            }
        );

        Ok(GenerationResult {
            prompt: prompt.to_string(),
            prompt_tokens,
            generated_tokens,
            generated_text,
            full_text,
            steps,
            total_time,
        })
    }

    /// Returns a reference to the effective model dimensions.
    pub fn effective_dims(&self) -> &ModelDims {
        &self.effective_dims
    }

    /// Returns a reference to the FHE parameters.
    pub fn params(&self) -> &FHE_LLMParams {
        &self.params
    }

    /// Returns a reference to the model specification.
    pub fn model_spec(&self) -> &ModelSpec {
        &self.model_spec
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Loads a TinyLlama inference pipeline with default configuration.
///
/// This is a convenience wrapper around [`InferencePipeline::load`] that
/// uses the TinyLlama model specification and default inference config.
///
/// # Arguments
///
/// * `model_path` - Path to `model.safetensors`.
/// * `tokenizer_path` - Path to `tokenizer.json`.
pub fn load_tinyllama(
    model_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
) -> Result<InferencePipeline, InferenceError> {
    InferencePipeline::load(
        model_path,
        tokenizer_path,
        ModelSpec::tinyllama_1_1b(),
        InferenceConfig::default(),
    )
}

/// Loads a TinyLlama inference pipeline with custom configuration.
pub fn load_tinyllama_with_config(
    model_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
    config: InferenceConfig,
) -> Result<InferencePipeline, InferenceError> {
    InferencePipeline::load(model_path, tokenizer_path, ModelSpec::tinyllama_1_1b(), config)
}
