//! Model weight loading from safetensors format.
//!
//! This module provides utilities for loading pre-trained transformer model
//! weights from the safetensors format and converting them into FHE_LLM's
//! internal weight representation ([`TransformerBlockWeights`]).
//!
//! ## Supported Weight Formats
//!
//! - **INT8** (`torch.int8`): Loaded directly as signed 8-bit integers, cast to `i64`.
//! - **FP16** (`torch.float16`): Loaded as IEEE 754 half-precision floats, then
//!   quantized to INT8 range `[-127, 127]` using per-tensor symmetric quantization.
//! - **BF16** (`torch.bfloat16`): Loaded as brain floating-point, then quantized
//!   to INT8 similarly to FP16.
//! - **FP32** (`torch.float32`): Loaded as single-precision floats, then quantized.
//!
//! ## Weight Naming Conventions
//!
//! The loader supports two common naming conventions:
//!
//! ### LLaMA / Mistral / Mixtral style
//! ```text
//! model.layers.{i}.self_attn.q_proj.weight   → AttentionWeights.w_q
//! model.layers.{i}.self_attn.k_proj.weight   → AttentionWeights.w_k
//! model.layers.{i}.self_attn.v_proj.weight   → AttentionWeights.w_v
//! model.layers.{i}.self_attn.o_proj.weight   → AttentionWeights.w_o
//! model.layers.{i}.mlp.gate_proj.weight      → FFNWeights.w1 (W_gate)
//! model.layers.{i}.mlp.down_proj.weight      → FFNWeights.w2 (W_down)
//! model.layers.{i}.mlp.up_proj.weight        → FFNWeights.w3 (W_up)
//! model.layers.{i}.input_layernorm.weight     → pre_attn_norm_gamma
//! model.layers.{i}.post_attention_layernorm.weight → pre_ffn_norm_gamma
//! ```
//!
//! ### GPT-2 / GPT-NeoX style (partial support)
//! ```text
//! transformer.h.{i}.attn.c_attn.weight       → fused QKV (split into Q, K, V)
//! transformer.h.{i}.attn.c_proj.weight       → AttentionWeights.w_o
//! transformer.h.{i}.mlp.c_fc.weight          → FFNWeights.w1
//! transformer.h.{i}.mlp.c_proj.weight        → FFNWeights.w2
//! ```
//!
//! ## Transpose Convention
//!
//! safetensors (PyTorch) stores linear layer weights as `[out_features, in_features]`.
//! FHE_LLM's weight matrices use the convention `[in_features, out_features]` for
//! the matmul `y = W · x` where each row of W corresponds to one input dimension.
//! The loader transposes automatically.
//!
//! ## Memory Mapping
//!
//! For large models, weights are memory-mapped from disk rather than fully loaded
//! into RAM. This is done via the `memmap2` crate and the safetensors mmap API.

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::Dtype;
use safetensors::SafeTensors;

use crate::attention::AttentionWeights;
use crate::params::ModelDims;
use crate::transformer::{FFNWeights, TransformerBlockWeights};

/// Errors that can occur during model loading.
#[derive(Debug)]
pub enum ModelLoadError {
    /// I/O error reading the file.
    Io(std::io::Error),
    /// safetensors parsing error.
    SafeTensors(String),
    /// A required tensor was not found in the file.
    MissingTensor(String),
    /// Tensor has an unexpected shape.
    ShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Unsupported tensor dtype.
    UnsupportedDtype { tensor_name: String, dtype: String },
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelLoadError::Io(e) => write!(f, "I/O error: {e}"),
            ModelLoadError::SafeTensors(e) => write!(f, "safetensors error: {e}"),
            ModelLoadError::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            ModelLoadError::ShapeMismatch {
                tensor_name,
                expected,
                actual,
            } => {
                write!(f, "shape mismatch for {tensor_name}: expected {expected:?}, got {actual:?}")
            }
            ModelLoadError::UnsupportedDtype { tensor_name, dtype } => {
                write!(f, "unsupported dtype for {tensor_name}: {dtype}")
            }
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<std::io::Error> for ModelLoadError {
    fn from(e: std::io::Error) -> Self {
        ModelLoadError::Io(e)
    }
}

/// Configuration for the model weight loader.
#[derive(Clone, Debug)]
pub struct LoaderConfig {
    /// Naming convention prefix for layers.
    /// Default: `"model.layers"` (LLaMA/Mistral convention).
    pub layer_prefix: String,
    /// Naming convention for attention sub-module.
    /// Default: `"self_attn"`.
    pub attn_prefix: String,
    /// Naming convention for MLP sub-module.
    /// Default: `"mlp"`.
    pub mlp_prefix: String,
    /// Whether the model uses SwiGLU FFN (has gate_proj/up_proj/down_proj).
    /// If false, expects c_fc/c_proj naming (standard FFN).
    pub swiglu: bool,
    /// Whether to transpose weight matrices from `[out, in]` to `[in, out]`.
    /// Default: `true` (PyTorch convention → FHE_LLM convention).
    pub transpose: bool,
    /// Model dimensions (used for shape validation).
    pub dims: ModelDims,
}

impl LoaderConfig {
    /// Creates a LLaMA-style loader configuration.
    pub fn llama(dims: ModelDims) -> Self {
        LoaderConfig {
            layer_prefix: "model.layers".to_string(),
            attn_prefix: "self_attn".to_string(),
            mlp_prefix: "mlp".to_string(),
            swiglu: true,
            transpose: true,
            dims,
        }
    }

    /// Creates a GPT-2/NeoX-style loader configuration.
    pub fn gpt2(dims: ModelDims) -> Self {
        LoaderConfig {
            layer_prefix: "transformer.h".to_string(),
            attn_prefix: "attn".to_string(),
            mlp_prefix: "mlp".to_string(),
            swiglu: false,
            transpose: true,
            dims,
        }
    }
}

/// Quantization parameters for a single tensor.
#[derive(Clone, Debug)]
pub struct QuantInfo {
    /// Scale factor: float_value = quantized_value * scale.
    pub scale: f64,
    /// The maximum absolute value in the original tensor.
    pub abs_max: f64,
}

/// Result of loading a single layer's weights, including quantization info.
#[derive(Clone, Debug)]
pub struct LayerLoadResult {
    /// The loaded transformer block weights.
    pub weights: TransformerBlockWeights,
    /// Per-tensor quantization info (tensor name → quant params).
    pub quant_info: HashMap<String, QuantInfo>,
}

impl LayerLoadResult {
    /// Converts loader-oriented weights into the orientation expected by the
    /// vector pipeline.
    ///
    /// The model loader returns attention weights in the standard loader
    /// orientation and FFN weights in the same orientation used by the single-
    /// ciphertext path. The vector pipeline expects FFN matrices with rows as
    /// output dimensions:
    /// - `w1`: `[d_ffn][d_model]`
    /// - `w2`: `[d_model][d_ffn]`
    /// - `w3`: `[d_ffn][d_model]` (if present)
    pub fn into_vec_pipeline_weights(self) -> TransformerBlockWeights {
        TransformerBlockWeights {
            attention: self.weights.attention,
            ffn: self.weights.ffn.into_vec_pipeline_weights(),
            pre_attn_norm_gamma: self.weights.pre_attn_norm_gamma,
            pre_ffn_norm_gamma: self.weights.pre_ffn_norm_gamma,
        }
    }
}

fn transpose_matrix(matrix: &[Vec<i64>]) -> Vec<Vec<i64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut out = vec![vec![0i64; rows]; cols];
    for (r, row) in matrix.iter().enumerate() {
        assert_eq!(
            row.len(),
            cols,
            "transpose_matrix: ragged input row {} has len {}, expected {}",
            r,
            row.len(),
            cols,
        );
        for (c, &value) in row.iter().enumerate() {
            out[c][r] = value;
        }
    }
    out
}

impl FFNWeights {
    /// Converts FFN weights from loader orientation into the orientation used
    /// by the vector pipeline.
    pub fn into_vec_pipeline_weights(self) -> FFNWeights {
        FFNWeights {
            w1: transpose_matrix(&self.w1),
            w2: transpose_matrix(&self.w2),
            w3: self.w3.map(|w3| transpose_matrix(&w3)),
        }
    }
}

// ---------------------------------------------------------------------------
// Core tensor conversion functions
// ---------------------------------------------------------------------------

/// Converts a raw byte buffer with the given dtype into a `Vec<f64>`.
///
/// Handles INT8, FP16, BF16, and FP32 dtypes.
fn tensor_to_f64(data: &[u8], dtype: Dtype) -> Result<Vec<f64>, String> {
    match dtype {
        Dtype::I8 => Ok(data.iter().map(|&b| (b as i8) as f64).collect()),
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err("F16 data length not a multiple of 2".to_string());
            }
            let values: Vec<f64> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let h = half::f16::from_bits(bits);
                    f64::from(f32::from(h))
                })
                .collect();
            Ok(values)
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err("BF16 data length not a multiple of 2".to_string());
            }
            let values: Vec<f64> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    let h = half::bf16::from_bits(bits);
                    f64::from(f32::from(h))
                })
                .collect();
            Ok(values)
        }
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err("F32 data length not a multiple of 4".to_string());
            }
            let values: Vec<f64> = data
                .chunks_exact(4)
                .map(|chunk| {
                    let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f32::from_bits(bits) as f64
                })
                .collect();
            Ok(values)
        }
        other => Err(format!("unsupported dtype: {other:?}")),
    }
}

/// Quantizes a vector of f64 values to INT8 range `[-127, 127]` using
/// per-tensor symmetric quantization.
///
/// Returns the quantized i64 values and the quantization info.
fn quantize_to_int8(values: &[f64]) -> (Vec<i64>, QuantInfo) {
    let abs_max = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    if abs_max < 1e-30 {
        // All-zero tensor
        return (
            vec![0i64; values.len()],
            QuantInfo {
                scale: 1.0,
                abs_max: 0.0,
            },
        );
    }

    let scale = abs_max / 127.0;
    let quantized: Vec<i64> = values
        .iter()
        .map(|&v| {
            let q = (v / scale).round();
            q.clamp(-127.0, 127.0) as i64
        })
        .collect();

    (quantized, QuantInfo { scale, abs_max })
}

/// Loads a 2D weight tensor from safetensors, converting to `Vec<Vec<i64>>`.
///
/// If the original dtype is not INT8, the values are quantized to INT8 range.
/// If `transpose` is true, the matrix is transposed from `[out, in]` to `[in, out]`.
///
/// Returns the weight matrix and quantization info.
fn load_weight_matrix(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_shape: [usize; 2],
    transpose: bool,
) -> Result<(Vec<Vec<i64>>, QuantInfo), ModelLoadError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| ModelLoadError::MissingTensor(name.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Err(ModelLoadError::ShapeMismatch {
            tensor_name: name.to_string(),
            expected: expected_shape.to_vec(),
            actual: shape.to_vec(),
        });
    }

    // safetensors shape is [out_features, in_features] (PyTorch convention)
    let (rows_file, cols_file) = (shape[0], shape[1]);

    // Validate shape against expected dimensions
    let (exp_rows, exp_cols) = if transpose {
        // Expected shape is the post-transpose shape: [in, out]
        // File shape should be [out, in] = [exp_cols, exp_rows]
        (expected_shape[1], expected_shape[0])
    } else {
        (expected_shape[0], expected_shape[1])
    };

    if rows_file != exp_rows || cols_file != exp_cols {
        return Err(ModelLoadError::ShapeMismatch {
            tensor_name: name.to_string(),
            expected: vec![exp_rows, exp_cols],
            actual: vec![rows_file, cols_file],
        });
    }

    let dtype = view.dtype();
    let data = view.data();

    // Convert to f64 values
    let flat_values = if dtype == Dtype::I8 {
        // INT8: direct conversion, no quantization needed
        let values: Vec<i64> = data.iter().map(|&b| (b as i8) as i64).collect();
        let quant = QuantInfo {
            scale: 1.0,
            abs_max: 127.0,
        };

        // Reshape and optionally transpose
        let matrix = if transpose {
            // File is [out, in] = [rows_file, cols_file]
            // We want [in, out] = [cols_file, rows_file]
            let mut result = vec![vec![0i64; rows_file]; cols_file];
            for r in 0..rows_file {
                for c in 0..cols_file {
                    result[c][r] = values[r * cols_file + c];
                }
            }
            result
        } else {
            let mut result = vec![vec![0i64; cols_file]; rows_file];
            for r in 0..rows_file {
                result[r] = values[r * cols_file..(r + 1) * cols_file].to_vec();
            }
            result
        };

        return Ok((matrix, quant));
    } else {
        tensor_to_f64(data, dtype).map_err(|e| ModelLoadError::UnsupportedDtype {
            tensor_name: name.to_string(),
            dtype: e,
        })?
    };

    // Quantize to INT8
    let (quantized, quant) = quantize_to_int8(&flat_values);

    // Reshape and optionally transpose
    let matrix = if transpose {
        // File is [out, in] = [rows_file, cols_file]
        // We want [in, out] = [cols_file, rows_file]
        let mut result = vec![vec![0i64; rows_file]; cols_file];
        for r in 0..rows_file {
            for c in 0..cols_file {
                result[c][r] = quantized[r * cols_file + c];
            }
        }
        result
    } else {
        let mut result = vec![vec![0i64; cols_file]; rows_file];
        for r in 0..rows_file {
            result[r] = quantized[r * cols_file..(r + 1) * cols_file].to_vec();
        }
        result
    };

    Ok((matrix, quant))
}

/// Loads a 1D tensor (e.g., RMSNorm gamma) from safetensors, converting to `Vec<i64>`.
///
/// For non-INT8 dtypes, values are quantized to INT8 range.
fn load_norm_weights(
    tensors: &SafeTensors<'_>,
    name: &str,
    expected_len: usize,
) -> Result<(Vec<i64>, QuantInfo), ModelLoadError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| ModelLoadError::MissingTensor(name.to_string()))?;

    let shape = view.shape();
    let total: usize = shape.iter().product();
    if total != expected_len {
        return Err(ModelLoadError::ShapeMismatch {
            tensor_name: name.to_string(),
            expected: vec![expected_len],
            actual: shape.to_vec(),
        });
    }

    let dtype = view.dtype();
    let data = view.data();

    let float_values = tensor_to_f64(data, dtype).map_err(|e| ModelLoadError::UnsupportedDtype {
        tensor_name: name.to_string(),
        dtype: e,
    })?;

    let fixed_scale = (1u64 << crate::activations::COEFF_SCALE_BITS) as f64;
    let fixed_point: Vec<i64> = float_values.iter().map(|&v| (v * fixed_scale).round() as i64).collect();
    let abs_max = float_values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    Ok((
        fixed_point,
        QuantInfo {
            scale: 1.0 / fixed_scale,
            abs_max,
        },
    ))
}

// ---------------------------------------------------------------------------
// High-level model loading API
// ---------------------------------------------------------------------------

/// Loads a single transformer layer's weights from a safetensors buffer.
///
/// The `layer_idx` is used to construct tensor names according to the
/// loader configuration's naming convention.
///
/// # Example
///
/// ```ignore
/// use poulpy_FHE_LLM::model_loader::{LoaderConfig, load_layer};
/// use poulpy_FHE_LLM::params::ModelDims;
///
/// let dims = ModelDims::dense_7b();
/// let config = LoaderConfig::llama(dims);
/// let data = std::fs::read("model.safetensors")?;
/// let tensors = safetensors::SafeTensors::deserialize(&data)?;
/// let result = load_layer(&tensors, 0, &config)?;
/// ```
pub fn load_layer(tensors: &SafeTensors<'_>, layer_idx: usize, config: &LoaderConfig) -> Result<LayerLoadResult, ModelLoadError> {
    let d = config.dims.d_model;
    let d_kv = config.dims.d_kv();
    let d_ffn = config.dims.d_ffn;
    let prefix = format!("{}.{layer_idx}", config.layer_prefix);
    let attn = &config.attn_prefix;
    let mlp = &config.mlp_prefix;

    let mut quant_info = HashMap::new();

    // ----- Attention weights -----
    // Q and O projections are always [d_model, d_model].
    // K and V projections are [d_kv, d_model] where d_kv = n_kv_heads * d_head.
    // For standard MHA (n_kv_heads == n_heads), d_kv == d_model.
    // For GQA (n_kv_heads < n_heads), d_kv < d_model.
    let (w_q, qi) = load_weight_matrix(tensors, &format!("{prefix}.{attn}.q_proj.weight"), [d, d], config.transpose)?;
    quant_info.insert("w_q".to_string(), qi);

    // K projection: file has [d_kv, d_model] (PyTorch: out_features × in_features).
    // For GQA, d_kv < d_model so this is a non-square matrix.
    // We use transpose=false to load as [d_kv][d_model] directly.
    let kv_transpose = if d_kv == d { config.transpose } else { false };
    let (w_k, qi) = load_weight_matrix(tensors, &format!("{prefix}.{attn}.k_proj.weight"), [d_kv, d], kv_transpose)?;
    quant_info.insert("w_k".to_string(), qi);

    let (w_v, qi) = load_weight_matrix(tensors, &format!("{prefix}.{attn}.v_proj.weight"), [d_kv, d], kv_transpose)?;
    quant_info.insert("w_v".to_string(), qi);

    let (w_o, qi) = load_weight_matrix(tensors, &format!("{prefix}.{attn}.o_proj.weight"), [d, d], config.transpose)?;
    quant_info.insert("w_o".to_string(), qi);

    let attention = AttentionWeights { w_q, w_k, w_v, w_o };

    // ----- FFN weights -----
    let ffn = if config.swiglu {
        // SwiGLU: gate_proj, down_proj, up_proj
        let (w1, qi) = load_weight_matrix(
            tensors,
            &format!("{prefix}.{mlp}.gate_proj.weight"),
            [d, d_ffn],
            config.transpose,
        )?;
        quant_info.insert("w1_gate".to_string(), qi);

        let (w2, qi) = load_weight_matrix(
            tensors,
            &format!("{prefix}.{mlp}.down_proj.weight"),
            [d_ffn, d],
            config.transpose,
        )?;
        quant_info.insert("w2_down".to_string(), qi);

        let (w3, qi) = load_weight_matrix(
            tensors,
            &format!("{prefix}.{mlp}.up_proj.weight"),
            [d, d_ffn],
            config.transpose,
        )?;
        quant_info.insert("w3_up".to_string(), qi);

        FFNWeights { w1, w2, w3: Some(w3) }
    } else {
        // Standard FFN: c_fc (or fc1), c_proj (or fc2)
        let w1_name = format!("{prefix}.{mlp}.c_fc.weight");
        let w2_name = format!("{prefix}.{mlp}.c_proj.weight");

        let (w1, qi) = load_weight_matrix(tensors, &w1_name, [d, d_ffn], config.transpose)?;
        quant_info.insert("w1".to_string(), qi);

        let (w2, qi) = load_weight_matrix(tensors, &w2_name, [d_ffn, d], config.transpose)?;
        quant_info.insert("w2".to_string(), qi);

        FFNWeights { w1, w2, w3: None }
    };

    // ----- Norm weights (optional — may not exist for all models) -----
    let pre_attn_norm_gamma = load_norm_weights(tensors, &format!("{prefix}.input_layernorm.weight"), d)
        .ok()
        .map(|(gamma, qi)| {
            quant_info.insert("pre_attn_norm_gamma".to_string(), qi);
            gamma
        });

    let pre_ffn_norm_gamma = load_norm_weights(tensors, &format!("{prefix}.post_attention_layernorm.weight"), d)
        .ok()
        .map(|(gamma, qi)| {
            quant_info.insert("pre_ffn_norm_gamma".to_string(), qi);
            gamma
        });

    let weights = TransformerBlockWeights {
        attention,
        ffn,
        pre_attn_norm_gamma,
        pre_ffn_norm_gamma,
    };

    Ok(LayerLoadResult { weights, quant_info })
}

/// Loads all transformer layers from a single safetensors file.
///
/// Returns one [`LayerLoadResult`] per layer, in layer order.
pub fn load_all_layers(tensors: &SafeTensors<'_>, config: &LoaderConfig) -> Result<Vec<LayerLoadResult>, ModelLoadError> {
    let mut results = Vec::with_capacity(config.dims.n_layers);
    for i in 0..config.dims.n_layers {
        let result = load_layer(tensors, i, config)?;
        results.push(result);
    }
    Ok(results)
}

/// Loads all transformer layers from a safetensors file on disk using memory mapping.
///
/// This is the recommended entry point for loading large models, as it avoids
/// reading the entire file into memory.
///
/// # Example
///
/// ```ignore
/// use poulpy_FHE_LLM::model_loader::{LoaderConfig, load_model_from_file};
/// use poulpy_FHE_LLM::params::ModelDims;
///
/// let dims = ModelDims::dense_7b();
/// let config = LoaderConfig::llama(dims);
/// let layers = load_model_from_file("model.safetensors", &config)?;
/// println!("Loaded {} layers", layers.len());
/// ```
pub fn load_model_from_file<P: AsRef<Path>>(path: P, config: &LoaderConfig) -> Result<Vec<LayerLoadResult>, ModelLoadError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| ModelLoadError::SafeTensors(e.to_string()))?;
    load_all_layers(&tensors, config)
}

/// Loads a single layer from a safetensors file on disk using memory mapping.
pub fn load_layer_from_file<P: AsRef<Path>>(
    path: P,
    layer_idx: usize,
    config: &LoaderConfig,
) -> Result<LayerLoadResult, ModelLoadError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| ModelLoadError::SafeTensors(e.to_string()))?;
    load_layer(&tensors, layer_idx, config)
}

/// Lists all tensor names and their shapes/dtypes from a safetensors file.
///
/// Useful for debugging and verifying that the naming convention matches
/// the expected format.
pub fn inspect_model<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Vec<usize>, String)>, ModelLoadError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| ModelLoadError::SafeTensors(e.to_string()))?;

    let mut info = Vec::new();
    for (name, view) in tensors.tensors() {
        info.push((name.to_string(), view.shape().to_vec(), format!("{:?}", view.dtype())));
    }
    info.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(info)
}

// ---------------------------------------------------------------------------
// Sharded model loading (multiple safetensors files)
// ---------------------------------------------------------------------------

/// Loads transformer layers from a sharded model (multiple safetensors files).
///
/// Many large models are split across multiple files like:
/// ```text
/// model-00001-of-00003.safetensors
/// model-00002-of-00003.safetensors
/// model-00003-of-00003.safetensors
/// ```
///
/// This function tries each file for each layer and returns the first match.
/// Layers not found in any file will produce a `MissingTensor` error.
pub fn load_model_sharded<P: AsRef<Path>>(
    shard_paths: &[P],
    config: &LoaderConfig,
) -> Result<Vec<LayerLoadResult>, ModelLoadError> {
    // Memory-map all shards
    let mut mmaps = Vec::with_capacity(shard_paths.len());
    for path in shard_paths {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        mmaps.push(mmap);
    }

    let safetensors_vec: Vec<SafeTensors<'_>> = mmaps
        .iter()
        .map(|mmap| SafeTensors::deserialize(mmap))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ModelLoadError::SafeTensors(e.to_string()))?;

    let mut results = Vec::with_capacity(config.dims.n_layers);

    for layer_idx in 0..config.dims.n_layers {
        let mut loaded = None;
        for st in &safetensors_vec {
            match load_layer(st, layer_idx, config) {
                Ok(result) => {
                    loaded = Some(result);
                    break;
                }
                Err(ModelLoadError::MissingTensor(_)) => continue,
                Err(e) => return Err(e),
            }
        }
        match loaded {
            Some(result) => results.push(result),
            None => {
                return Err(ModelLoadError::MissingTensor(format!(
                    "layer {layer_idx} not found in any shard"
                )));
            }
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Embedding and LM head loading
// ---------------------------------------------------------------------------

/// Loaded embedding table: `[vocab_size, d_model]` as INT8 values.
///
/// The embedding table is not encrypted — it lives on the user's side.
/// The user looks up a token's embedding row (cleartext), then encrypts
/// each dimension as a separate ciphertext before sending to the provider.
#[derive(Clone, Debug)]
pub struct EmbeddingTable {
    /// Embedding vectors: `embeddings[token_id][dim]`.
    pub embeddings: Vec<Vec<i64>>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub d_model: usize,
    /// Quantization parameters for the embedding table.
    pub quant_info: QuantInfo,
}

impl EmbeddingTable {
    /// Looks up the embedding for a given token ID.
    ///
    /// Returns a `Vec<i64>` of length `d_model`, suitable for encrypting
    /// one dimension at a time via `_FHE_LLM_encrypt`.
    ///
    /// # Panics
    ///
    /// Panics if `token_id >= vocab_size`.
    pub fn lookup(&self, token_id: usize) -> &[i64] {
        assert!(
            token_id < self.vocab_size,
            "token_id {token_id} out of range (vocab_size = {})",
            self.vocab_size
        );
        &self.embeddings[token_id]
    }
}

/// Loads the token embedding table from a safetensors buffer.
///
/// Looks for the tensor named `embed_name` (e.g. `"model.embed_tokens.weight"`),
/// which should have shape `[vocab_size, d_model]`.
///
/// The tensor is quantized to INT8 if not already.
pub fn load_embedding_table(
    tensors: &SafeTensors<'_>,
    embed_name: &str,
    d_model: usize,
) -> Result<EmbeddingTable, ModelLoadError> {
    let view = tensors
        .tensor(embed_name)
        .map_err(|_| ModelLoadError::MissingTensor(embed_name.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 || shape[1] != d_model {
        return Err(ModelLoadError::ShapeMismatch {
            tensor_name: embed_name.to_string(),
            expected: vec![0, d_model], // 0 = any vocab_size
            actual: shape.to_vec(),
        });
    }

    let vocab_size = shape[0];
    let dtype = view.dtype();
    let data = view.data();

    let (flat_quantized, quant_info) = if dtype == Dtype::I8 {
        let values: Vec<i64> = data.iter().map(|&b| (b as i8) as i64).collect();
        (
            values,
            QuantInfo {
                scale: 1.0,
                abs_max: 127.0,
            },
        )
    } else {
        let float_values = tensor_to_f64(data, dtype).map_err(|e| ModelLoadError::UnsupportedDtype {
            tensor_name: embed_name.to_string(),
            dtype: e,
        })?;
        let (quantized, qi) = quantize_to_int8(&float_values);
        (quantized, qi)
    };

    // Reshape flat [vocab_size * d_model] → [vocab_size][d_model]
    let mut embeddings = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let start = i * d_model;
        let end = start + d_model;
        embeddings.push(flat_quantized[start..end].to_vec());
    }

    Ok(EmbeddingTable {
        embeddings,
        vocab_size,
        d_model,
        quant_info,
    })
}

/// Loads the token embedding table from a safetensors file on disk.
pub fn load_embedding_from_file<P: AsRef<Path>>(
    path: P,
    embed_name: &str,
    d_model: usize,
) -> Result<EmbeddingTable, ModelLoadError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| ModelLoadError::SafeTensors(e.to_string()))?;
    load_embedding_table(&tensors, embed_name, d_model)
}

/// Loaded LM head weights: `[vocab_size, d_model]` as INT8 values.
///
/// Like the embedding table, the LM head lives on the user's side.
/// After the provider returns encrypted hidden states, the user decrypts
/// locally and computes `logits[v] = Σ_d W_lmhead[v][d] * hidden[d]` in
/// cleartext.
#[derive(Clone, Debug)]
pub struct LMHead {
    /// Weight matrix: `weights[vocab_token][dim]`.
    pub weights: Vec<Vec<i64>>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub d_model: usize,
    /// Quantization parameters.
    pub quant_info: QuantInfo,
}

impl LMHead {
    /// Computes output logits from a decrypted hidden state (cleartext).
    ///
    /// `hidden` should be a `Vec<i64>` of length `d_model` (the decrypted
    /// output of the final transformer layer).
    ///
    /// Returns `Vec<i64>` of length `vocab_size` where `logits[v] = Σ_d W[v][d] * hidden[d]`.
    ///
    /// # Panics
    ///
    /// Panics if `hidden.len() != d_model`.
    pub fn forward(&self, hidden: &[i64]) -> Vec<i64> {
        assert_eq!(
            hidden.len(),
            self.d_model,
            "LMHead: hidden state has {} dims, expected {}",
            hidden.len(),
            self.d_model
        );
        self.weights
            .iter()
            .map(|w_row| w_row.iter().zip(hidden.iter()).map(|(&w, &h)| w * h).sum())
            .collect()
    }

    /// Returns the token ID with the highest logit (greedy decoding).
    pub fn argmax(&self, hidden: &[i64]) -> usize {
        let logits = self.forward(hidden);
        logits
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

/// Loads the LM head weight matrix from a safetensors buffer.
///
/// Looks for the tensor named `head_name` (e.g. `"lm_head.weight"`),
/// which should have shape `[vocab_size, d_model]`.
///
/// In many LLaMA-family models, the LM head shares weights with the
/// embedding table. In that case, pass the embedding tensor name instead.
pub fn load_lm_head(tensors: &SafeTensors<'_>, head_name: &str, d_model: usize) -> Result<LMHead, ModelLoadError> {
    let view = tensors
        .tensor(head_name)
        .map_err(|_| ModelLoadError::MissingTensor(head_name.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 || shape[1] != d_model {
        return Err(ModelLoadError::ShapeMismatch {
            tensor_name: head_name.to_string(),
            expected: vec![0, d_model],
            actual: shape.to_vec(),
        });
    }

    let vocab_size = shape[0];
    let dtype = view.dtype();
    let data = view.data();

    let (flat_quantized, quant_info) = if dtype == Dtype::I8 {
        let values: Vec<i64> = data.iter().map(|&b| (b as i8) as i64).collect();
        (
            values,
            QuantInfo {
                scale: 1.0,
                abs_max: 127.0,
            },
        )
    } else {
        let float_values = tensor_to_f64(data, dtype).map_err(|e| ModelLoadError::UnsupportedDtype {
            tensor_name: head_name.to_string(),
            dtype: e,
        })?;
        let (quantized, qi) = quantize_to_int8(&float_values);
        (quantized, qi)
    };

    let mut weights = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let start = i * d_model;
        let end = start + d_model;
        weights.push(flat_quantized[start..end].to_vec());
    }

    Ok(LMHead {
        weights,
        vocab_size,
        d_model,
        quant_info,
    })
}

/// Loads the LM head from a safetensors file on disk.
pub fn load_lm_head_from_file<P: AsRef<Path>>(path: P, head_name: &str, d_model: usize) -> Result<LMHead, ModelLoadError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| ModelLoadError::SafeTensors(e.to_string()))?;
    load_lm_head(&tensors, head_name, d_model)
}

/// Loads the final RMSNorm weights (applied after all transformer layers,
/// before the LM head) from a safetensors buffer.
///
/// In LLaMA models, this is `model.norm.weight` (shape `[d_model]`).
pub fn load_final_norm(
    tensors: &SafeTensors<'_>,
    norm_name: &str,
    d_model: usize,
) -> Result<(Vec<i64>, QuantInfo), ModelLoadError> {
    load_norm_weights(tensors, norm_name, d_model)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a minimal safetensors buffer with one INT8 2×3 tensor.
    fn make_test_safetensors_int8() -> Vec<u8> {
        use safetensors::tensor::serialize;
        use std::collections::HashMap;

        let mut tensors = HashMap::new();

        // A 2×3 INT8 tensor: [[1, 2, 3], [4, 5, 6]]
        let data: Vec<u8> = vec![1u8, 2, 3, 4, 5, 6];
        tensors.insert(
            "test_weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![2, 3], &data).unwrap(),
        );

        serialize(&tensors, &None).unwrap()
    }

    /// Creates a minimal safetensors buffer with one FP16 2×2 tensor.
    fn make_test_safetensors_fp16() -> Vec<u8> {
        use safetensors::tensor::serialize;
        use std::collections::HashMap;

        let mut tensors = HashMap::new();

        // FP16 values: [1.0, -0.5, 0.25, 2.0]
        let values = [1.0_f32, -0.5, 0.25, 2.0];
        let mut data = Vec::new();
        for &v in &values {
            let h = half::f16::from_f32(v);
            data.extend_from_slice(&h.to_le_bytes());
        }
        tensors.insert(
            "test_weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::F16, vec![2, 2], &data).unwrap(),
        );

        serialize(&tensors, &None).unwrap()
    }

    #[test]
    fn test_tensor_to_f64_int8() {
        let data: Vec<u8> = vec![1u8, 255, 127, 128]; // 1, -1, 127, -128 as i8
        let result = tensor_to_f64(&data, Dtype::I8).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], -1.0);
        assert_eq!(result[2], 127.0);
        assert_eq!(result[3], -128.0);
    }

    #[test]
    fn test_tensor_to_f64_fp16() {
        let h = half::f16::from_f32(3.14);
        let data = h.to_le_bytes().to_vec();
        let result = tensor_to_f64(&data, Dtype::F16).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.14).abs() < 0.01);
    }

    #[test]
    fn test_quantize_to_int8_identity() {
        // Values already in [-127, 127] should quantize to themselves (approximately)
        let values = vec![0.0, 50.0, -50.0, 127.0, -127.0];
        let (quantized, quant) = quantize_to_int8(&values);
        assert_eq!(quantized.len(), 5);
        assert!((quant.abs_max - 127.0).abs() < 1e-10);
        assert!((quant.scale - 1.0).abs() < 1e-10);
        assert_eq!(quantized[0], 0);
        assert_eq!(quantized[1], 50);
        assert_eq!(quantized[2], -50);
        assert_eq!(quantized[3], 127);
        assert_eq!(quantized[4], -127);
    }

    #[test]
    fn test_quantize_to_int8_scaling() {
        // Values outside [-127, 127] should be scaled
        let values = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let (quantized, quant) = quantize_to_int8(&values);
        assert_eq!(quantized.len(), 5);
        assert!((quant.abs_max - 1.0).abs() < 1e-10);
        assert!((quant.scale - 1.0 / 127.0).abs() < 1e-10);
        assert_eq!(quantized[0], 0);
        assert_eq!(quantized[3], 127); // 1.0 → 127
        assert_eq!(quantized[4], -127); // -1.0 → -127
                                        // 0.5 → 0.5 / (1.0/127.0) = 63.5 → rounds to 64
        assert!((quantized[1] - 64).abs() <= 1);
    }

    #[test]
    fn test_quantize_zero_tensor() {
        let values = vec![0.0, 0.0, 0.0];
        let (quantized, quant) = quantize_to_int8(&values);
        assert_eq!(quantized, vec![0, 0, 0]);
        assert_eq!(quant.scale, 1.0);
        assert_eq!(quant.abs_max, 0.0);
    }

    #[test]
    fn test_load_weight_matrix_int8_no_transpose() {
        let buf = make_test_safetensors_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        // Load [2, 3] tensor without transpose, expecting [2, 3]
        let (matrix, quant) = load_weight_matrix(&tensors, "test_weight", [2, 3], false).unwrap();

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 3);
        assert_eq!(matrix[0], vec![1, 2, 3]);
        assert_eq!(matrix[1], vec![4, 5, 6]);
        assert_eq!(quant.scale, 1.0); // INT8 → no quantization
    }

    #[test]
    fn test_load_weight_matrix_int8_transpose() {
        let buf = make_test_safetensors_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        // File has [2, 3] (out=2, in=3). After transpose we want [3, 2].
        // Expected shape [3, 2] means file should be [2, 3] ✓
        let (matrix, _quant) = load_weight_matrix(&tensors, "test_weight", [3, 2], true).unwrap();

        assert_eq!(matrix.len(), 3); // 3 rows (in_features)
        assert_eq!(matrix[0].len(), 2); // 2 cols (out_features)
                                        // Transposed: col 0 of file = [1, 4], col 1 = [2, 5], col 2 = [3, 6]
        assert_eq!(matrix[0], vec![1, 4]);
        assert_eq!(matrix[1], vec![2, 5]);
        assert_eq!(matrix[2], vec![3, 6]);
    }

    #[test]
    fn test_load_weight_matrix_fp16() {
        let buf = make_test_safetensors_fp16();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        // [2, 2] FP16 tensor with values [1.0, -0.5, 0.25, 2.0]
        // After quantization with abs_max=2.0, scale=2.0/127.0:
        //   1.0  → round(1.0 / (2.0/127.0)) = round(63.5) = 64
        //  -0.5  → round(-0.5 / (2.0/127.0)) = round(-31.75) = -32
        //   0.25 → round(0.25 / (2.0/127.0)) = round(15.875) = 16
        //   2.0  → round(2.0 / (2.0/127.0)) = 127
        let (matrix, quant) = load_weight_matrix(&tensors, "test_weight", [2, 2], false).unwrap();

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert!((quant.abs_max - 2.0).abs() < 0.01);
        assert_eq!(matrix[0][0], 64); // 1.0
        assert_eq!(matrix[0][1], -32); // -0.5
        assert_eq!(matrix[1][0], 16); // 0.25
        assert_eq!(matrix[1][1], 127); // 2.0
    }

    #[test]
    fn test_load_weight_matrix_shape_mismatch() {
        let buf = make_test_safetensors_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        // Expect [4, 4] but file has [2, 3]
        let result = load_weight_matrix(&tensors, "test_weight", [4, 4], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_weight_matrix_missing() {
        let buf = make_test_safetensors_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let result = load_weight_matrix(&tensors, "nonexistent", [2, 3], false);
        assert!(matches!(result, Err(ModelLoadError::MissingTensor(_))));
    }

    #[test]
    fn test_loader_config_constructors() {
        let dims = ModelDims::dense_7b();
        let llama_cfg = LoaderConfig::llama(dims.clone());
        assert_eq!(llama_cfg.layer_prefix, "model.layers");
        assert!(llama_cfg.swiglu);
        assert!(llama_cfg.transpose);

        let gpt2_cfg = LoaderConfig::gpt2(dims);
        assert_eq!(gpt2_cfg.layer_prefix, "transformer.h");
        assert!(!gpt2_cfg.swiglu);
    }

    /// Creates a safetensors buffer simulating a single LLaMA-style layer
    /// with tiny dimensions (d_model=2, d_ffn=3).
    fn make_test_llama_layer() -> Vec<u8> {
        use safetensors::tensor::serialize;
        use std::collections::HashMap;

        let mut tensors = HashMap::new();

        // Helper: create INT8 tensor data
        let make_2d = |rows: usize, cols: usize, base: i8| -> Vec<u8> {
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    data.push((base.wrapping_add((r * cols + c) as i8)) as u8);
                }
            }
            data
        };

        let make_1d = |len: usize, base: i8| -> Vec<u8> { (0..len).map(|i| (base.wrapping_add(i as i8)) as u8).collect() };

        let prefix = "model.layers.0";

        // Pre-allocate all data buffers so they live long enough for TensorView borrows
        let attn_names = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ];
        let attn_data: Vec<Vec<u8>> = attn_names.iter().map(|_| make_2d(2, 2, 1)).collect();

        for (name, data) in attn_names.iter().zip(attn_data.iter()) {
            tensors.insert(
                format!("{prefix}.{name}"),
                safetensors::tensor::TensorView::new(Dtype::I8, vec![2, 2], data).unwrap(),
            );
        }

        // MLP: gate_proj [out=3, in=2], down_proj [out=2, in=3], up_proj [out=3, in=2]
        let gate_data = make_2d(3, 2, 10);
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![3, 2], &gate_data).unwrap(),
        );
        let down_data = make_2d(2, 3, 20);
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![2, 3], &down_data).unwrap(),
        );
        let up_data = make_2d(3, 2, 30);
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![3, 2], &up_data).unwrap(),
        );

        // Norms: 1D [2]
        let norm1_data = make_1d(2, 50);
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![2], &norm1_data).unwrap(),
        );
        let norm2_data = make_1d(2, 60);
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![2], &norm2_data).unwrap(),
        );

        serialize(&tensors, &None).unwrap()
    }

    #[test]
    fn test_load_layer_llama_style() {
        let buf = make_test_llama_layer();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let dims = ModelDims {
            d_model: 2,
            d_head: 2,
            n_heads: 1,
            n_kv_heads: 1,
            d_ffn: 3,
            n_layers: 1,
            n_experts: 1,
            n_active_experts: 1,
        };
        let config = LoaderConfig::llama(dims);

        let result = load_layer(&tensors, 0, &config).unwrap();
        let w = &result.weights;

        // Attention weights should be 2×2 (transposed from file's [2,2])
        assert_eq!(w.attention.w_q.len(), 2);
        assert_eq!(w.attention.w_q[0].len(), 2);

        // FFN gate should be [in=2, out=3] (transposed from file's [3,2])
        assert_eq!(w.ffn.w1.len(), 2);
        assert_eq!(w.ffn.w1[0].len(), 3);

        // FFN down should be [in=3, out=2] (transposed from file's [2,3])
        assert_eq!(w.ffn.w2.len(), 3);
        assert_eq!(w.ffn.w2[0].len(), 2);

        // FFN up (w3) should exist and be [in=2, out=3]
        assert!(w.ffn.w3.is_some());
        let w3 = w.ffn.w3.as_ref().unwrap();
        assert_eq!(w3.len(), 2);
        assert_eq!(w3[0].len(), 3);

        // Norm weights should exist
        assert!(w.pre_attn_norm_gamma.is_some());
        assert_eq!(w.pre_attn_norm_gamma.as_ref().unwrap().len(), 2);
        assert!(w.pre_ffn_norm_gamma.is_some());
        assert_eq!(w.pre_ffn_norm_gamma.as_ref().unwrap().len(), 2);

        // Quant info should have entries
        assert!(result.quant_info.contains_key("w_q"));
        assert!(result.quant_info.contains_key("w1_gate"));
        assert!(result.quant_info.contains_key("pre_attn_norm_gamma"));
    }

    // -----------------------------------------------------------------------
    // Helpers for EmbeddingTable / LMHead / final norm tests
    // -----------------------------------------------------------------------

    /// Creates a safetensors buffer with an INT8 embedding table.
    /// vocab_size=3, d_model=2.  Rows: [1, 2], [3, 4], [5, 6].
    fn make_test_embedding_int8() -> Vec<u8> {
        use safetensors::tensor::serialize;
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        // 3×2 INT8: [[1,2],[3,4],[5,6]]
        let data: Vec<u8> = vec![1u8, 2, 3, 4, 5, 6];
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![3, 2], &data).unwrap(),
        );
        serialize(&tensors, &None).unwrap()
    }

    /// Creates a safetensors buffer with an FP16 embedding table.
    /// vocab_size=2, d_model=2.  Rows: [1.0, -0.5], [0.25, 2.0].
    fn make_test_embedding_fp16() -> Vec<u8> {
        use safetensors::tensor::serialize;
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let values = [1.0_f32, -0.5, 0.25, 2.0];
        let mut data = Vec::new();
        for &v in &values {
            let h = half::f16::from_f32(v);
            data.extend_from_slice(&h.to_le_bytes());
        }
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::F16, vec![2, 2], &data).unwrap(),
        );
        serialize(&tensors, &None).unwrap()
    }

    /// Creates a safetensors buffer with a small INT8 LM head weight.
    /// vocab_size=3, d_model=2.  Rows: [10, 20], [30, 40], [50, 60].
    fn make_test_lm_head_int8() -> Vec<u8> {
        use safetensors::tensor::serialize;
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let data: Vec<u8> = vec![10u8, 20, 30, 40, 50, 60];
        tensors.insert(
            "lm_head.weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::I8, vec![3, 2], &data).unwrap(),
        );
        serialize(&tensors, &None).unwrap()
    }

    /// Creates a safetensors buffer with a 1D FP16 norm tensor.
    /// Shape [4], values [1.0, -0.5, 0.25, 2.0].
    fn make_test_final_norm() -> Vec<u8> {
        use safetensors::tensor::serialize;
        use std::collections::HashMap;

        let mut tensors = HashMap::new();
        let values = [1.0_f32, -0.5, 0.25, 2.0];
        let mut data = Vec::new();
        for &v in &values {
            let h = half::f16::from_f32(v);
            data.extend_from_slice(&h.to_le_bytes());
        }
        tensors.insert(
            "model.norm.weight".to_string(),
            safetensors::tensor::TensorView::new(Dtype::F16, vec![4], &data).unwrap(),
        );
        serialize(&tensors, &None).unwrap()
    }

    // -----------------------------------------------------------------------
    // EmbeddingTable tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_embedding_table_basic() {
        let buf = make_test_embedding_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let table = load_embedding_table(&tensors, "model.embed_tokens.weight", 2).unwrap();

        assert_eq!(table.vocab_size, 3);
        assert_eq!(table.d_model, 2);
        assert_eq!(table.lookup(0), &[1i64, 2]);
        assert_eq!(table.lookup(1), &[3i64, 4]);
        assert_eq!(table.lookup(2), &[5i64, 6]);
    }

    #[test]
    fn test_embedding_table_fp16() {
        let buf = make_test_embedding_fp16();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let table = load_embedding_table(&tensors, "model.embed_tokens.weight", 2).unwrap();

        assert_eq!(table.vocab_size, 2);
        assert_eq!(table.d_model, 2);
        // FP16 values [1.0, -0.5, 0.25, 2.0], abs_max=2.0, scale=2.0/127.0
        // 1.0  → round(1.0 / (2.0/127.0)) = round(63.5) = 64
        // -0.5 → round(-0.5 / (2.0/127.0)) = round(-31.75) = -32
        // 0.25 → round(0.25 / (2.0/127.0)) = round(15.875) = 16
        // 2.0  → 127
        assert!((table.quant_info.abs_max - 2.0).abs() < 0.01);
        assert_eq!(table.lookup(0), &[64i64, -32]);
        assert_eq!(table.lookup(1), &[16i64, 127]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_embedding_lookup_panic() {
        let buf = make_test_embedding_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let table = load_embedding_table(&tensors, "model.embed_tokens.weight", 2).unwrap();
        // vocab_size=3, so lookup(3) should panic
        let _ = table.lookup(3);
    }

    // -----------------------------------------------------------------------
    // LMHead tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lm_head_forward() {
        let buf = make_test_lm_head_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let head = load_lm_head(&tensors, "lm_head.weight", 2).unwrap();

        assert_eq!(head.vocab_size, 3);
        assert_eq!(head.d_model, 2);

        // Weights: [[10,20],[30,40],[50,60]]
        // forward([1, 0]):
        //   logit[0] = 10*1 + 20*0 = 10
        //   logit[1] = 30*1 + 40*0 = 30
        //   logit[2] = 50*1 + 60*0 = 50
        let logits = head.forward(&[1, 0]);
        assert_eq!(logits, vec![10, 30, 50]);

        // forward([0, 1]):
        //   logit[0] = 10*0 + 20*1 = 20
        //   logit[1] = 30*0 + 40*1 = 40
        //   logit[2] = 50*0 + 60*1 = 60
        let logits = head.forward(&[0, 1]);
        assert_eq!(logits, vec![20, 40, 60]);

        // forward([1, 1]):
        //   logit[0] = 10 + 20 = 30
        //   logit[1] = 30 + 40 = 70
        //   logit[2] = 50 + 60 = 110
        let logits = head.forward(&[1, 1]);
        assert_eq!(logits, vec![30, 70, 110]);
    }

    #[test]
    fn test_lm_head_argmax() {
        let buf = make_test_lm_head_int8();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let head = load_lm_head(&tensors, "lm_head.weight", 2).unwrap();

        // Weights: [[10,20],[30,40],[50,60]]
        // forward([1, 0]) = [10, 30, 50] → argmax = 2
        assert_eq!(head.argmax(&[1, 0]), 2);

        // forward([0, 1]) = [20, 40, 60] → argmax = 2
        assert_eq!(head.argmax(&[0, 1]), 2);

        // forward([1, -1]) = [10-20, 30-40, 50-60] = [-10, -10, -10]
        // All tied; max_by_key returns last max index = 2
        assert_eq!(head.argmax(&[1, -1]), 2);

        // forward([2, 1]) = [20+20, 60+40, 100+60] = [40, 100, 160] → argmax = 2
        assert_eq!(head.argmax(&[2, 1]), 2);
    }

    // -----------------------------------------------------------------------
    // load_final_norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_final_norm() {
        let buf = make_test_final_norm();
        let tensors = SafeTensors::deserialize(&buf).unwrap();

        let (weights, quant) = load_final_norm(&tensors, "model.norm.weight", 4).unwrap();

        assert_eq!(weights.len(), 4);
        assert_eq!(weights[0], 256);
        assert_eq!(weights[1], -128);
        assert_eq!(weights[2], 64);
        assert_eq!(weights[3], 512);
        assert!((quant.scale - (1.0 / 256.0)).abs() < 1e-12);
        assert!((quant.abs_max - 2.0).abs() < 1e-12);
    }
}
