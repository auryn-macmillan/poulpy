//! End-to-end encrypted inference pipeline.
//!
//! This module ties together all CHIMERA components into a complete
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
//! 6. Apply final RMSNorm
//! 7. Return encrypted hidden state to user
//!
//! **User side (cleartext):**
//! 8. Decrypt hidden state → `Vec<i8>`
//! 9. Apply LM head (cleartext matmul) → logits
//! 10. Argmax → next token ID
//! 11. Decode token ID → text
//!
//! ## Usage
//!
//! ```ignore
//! use poulpy_chimera::inference::{InferencePipeline, InferenceConfig};
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

use poulpy_core::layouts::GLWE;
use poulpy_hal::api::ModuleNew;
use poulpy_hal::layouts::Module;
use tokenizers::Tokenizer;

use crate::attention::{AttentionConfig, AttentionWeights, SoftmaxStrategy};
use crate::encoding::{decode_int8, encode_int8};
use crate::encrypt::{chimera_decrypt, chimera_encrypt, ChimeraEvalKey, ChimeraKey};
use crate::layernorm::LayerNormConfig;
use crate::model_loader::{
    load_embedding_from_file, load_final_norm, load_layer_from_file, load_lm_head_from_file, EmbeddingTable, LMHead,
    LoaderConfig, ModelLoadError,
};
use crate::params::{ChimeraParams, ModelDims, Precision, SecurityLevel};
use crate::transformer::{chimera_transformer_block_vec, FFNConfig, TransformerBlockConfig, TransformerBlockWeights};

// ---------------------------------------------------------------------------
// Backend selection (mirrors tests.rs)
// ---------------------------------------------------------------------------

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
type BE = poulpy_cpu_ref::FFT64Ref;
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
type BE = poulpy_cpu_avx::FFT64Avx;

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
    params: ChimeraParams,
    /// Backend module.
    module: Module<BE>,
    /// FHE secret key (user-side).
    key: ChimeraKey<BE>,
    /// FHE evaluation key (sent to provider).
    eval_key: ChimeraEvalKey<BE>,
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
        let params = ChimeraParams::new(config.security, config.precision);
        eprintln!(
            "[inference] Generating FHE keys ({:?} security, N={})...",
            config.security,
            params.n()
        );
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, config.key_seed);
        let eval_key = ChimeraEvalKey::generate(&module, &key, &params, config.eval_seed_a, config.eval_seed_b);
        eprintln!("[inference] FHE keys generated");

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
                chimera_encrypt(&self.module, &self.key, &pt, seed_a, seed_e)
            })
            .collect()
    }

    /// Decrypts a vector of per-dimension ciphertexts back to i8 values.
    fn decrypt_hidden_state(&self, cts: &[GLWE<Vec<u8>>]) -> Vec<i8> {
        cts.iter()
            .map(|ct| {
                let pt = chimera_decrypt(&self.module, &self.key, ct, &self.params);
                let vals = decode_int8(&self.module, &self.params, &pt, 1);
                vals[0]
            })
            .collect()
    }

    /// Runs the FHE forward pass: transformer layers + optional final norm.
    ///
    /// This is the provider-side computation. Returns encrypted hidden state.
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
            current = chimera_transformer_block_vec(&self.module, &self.eval_key, &current, &block_config, weights);
            eprintln!("[inference]   Layer {} done in {:.2?}", layer_idx, layer_start.elapsed());
        }

        // Apply final RMSNorm if configured
        if let Some(ref gamma) = self.final_norm_gamma {
            eprintln!("[inference]   Applying final RMSNorm...");
            let norm_cfg = LayerNormConfig::rms_norm(d).with_gamma(gamma.clone());
            current = crate::layernorm::chimera_rms_norm_vec(&self.module, &self.eval_key, &current, &norm_cfg);
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

        crate::plaintext_forward::three_way_comparison(
            fhe_hidden,
            &embedding,
            &block_config,
            &self.layer_weights,
            self.final_norm_gamma.as_deref(),
            block_config.pre_attn_norm.epsilon,
        )
    }

    /// Applies the LM head to a decrypted hidden state and returns logits.
    ///
    /// If the model was truncated, this uses a truncated LM head
    /// (first `d_model` dimensions only).
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
        let encrypted_output = self.fhe_forward(&encrypted);
        let fhe_time = fhe_start.elapsed();
        eprintln!("[inference] FHE forward pass done in {:.2?}", fhe_time);

        // 4. Decrypt hidden state (user-side)
        let decrypt_start = Instant::now();
        let hidden = self.decrypt_hidden_state(&encrypted_output);
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
    pub fn params(&self) -> &ChimeraParams {
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
