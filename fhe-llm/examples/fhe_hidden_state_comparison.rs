//! Layer-by-layer FHE vs plaintext comparison.
//!
//! This example compares the hidden states produced by:
//! 1. Exact plaintext inference (using standard f64 arithmetic)
//! 2. FHE encrypted inference (with per-layer bootstrapping)
//!
//! The goal is to validate that FHE arithmetic produces correct results.

use std::process;
use std::time::Instant;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut args_iter = args.into_iter();
    let user_prompt = args_iter.next().unwrap_or_else(|| "What is 2+2?".to_string());
    let d_model: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_layers: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let d_ffn = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(d_model * 2);

    // For truncated config, ensure d_model = n_heads * d_head (where d_head=64 from original model)
    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);
    // Recalculate d_model to match n_heads * d_head for GQA consistency
    let effective_d_model = n_heads * 64;

    eprintln!("=== SmolLM2 Layer-by-Layer Hidden State Comparison ===");
    eprintln!("effective_d_model={effective_d_model}, d_ffn={d_ffn}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, d_head=64, n_layers={n_layers}");
    eprintln!("prompt: {:?}\n", user_prompt);

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Fp16, // FP16 mode: scale_bits=14, 13-bit bootstrap
        num_layers: Some(n_layers),
        trunc_d_model: Some(effective_d_model),
        trunc_d_ffn: Some(d_ffn),
        num_heads: Some(n_heads),
        num_kv_heads: Some(n_kv_heads),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
        fhe_silu_log_msg_mod: Some(13),     // 8192 entries for SiLU
        fhe_identity_log_msg_mod: Some(12), // 4096 entries for identity
    };

    eprintln!("[main] Loading model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    // Run FHE forward through all layers with per-layer capture
    eprintln!("[fhe] Running FHE encrypted forward pass with per-layer capture...");
    let fhe_start = Instant::now();

    // Use the layer comparison method to capture hidden states at each layer
    let layer_results = pipeline.compare_prompt_fhe_layer_hidden_states(&user_prompt)?;
    let fhe_time = fhe_start.elapsed().as_secs_f64();

    let fhe_hidden = layer_results.last().unwrap().1.clone();
    let fhe_hidden_f64: Vec<f64> = fhe_hidden.iter().map(|&v| v as f64).collect();

    // Decode FHE hidden state to token ID (argmax)
    let max_idx = fhe_hidden_f64
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let fhe_token_id: u32 = max_idx as u32;
    let fhe_token = pipeline.decode(&[fhe_token_id])?;
    eprintln!(
        "[fhe] Done in {:.1}s, token: {} (id: {})\n",
        fhe_time, fhe_token, fhe_token_id
    );

    // Compute error metrics
    eprintln!("=== Per-Layer Hidden State Comparison ===");
    for (layer_name, hidden_state) in &layer_results {
        let hidden_f64: Vec<f64> = hidden_state.iter().map(|&v| v as f64).collect();
        let mean: f64 = hidden_f64.iter().sum::<f64>() / hidden_f64.len() as f64;
        let max: f64 = hidden_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min: f64 = hidden_f64.iter().cloned().fold(f64::INFINITY, f64::min);
        let abs_sum: f64 = hidden_f64.iter().map(|&v| v.abs()).sum::<f64>();
        let max_abs: f64 = hidden_f64.iter().map(|&v| v.abs()).fold(0.0, f64::max);

        eprintln!(
            "{:15} | Mean: {:8.6} | Min: {:8.6} | Max: {:8.6} | SumAbs: {:8.6} | MaxAbs: {:8.6}",
            layer_name, mean, min, max, abs_sum, max_abs
        );
    }

    // Final statistics
    let fhe_mean: f64 = fhe_hidden_f64.iter().sum::<f64>() / fhe_hidden_f64.len() as f64;
    let fhe_max: f64 = fhe_hidden_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let fhe_min: f64 = fhe_hidden_f64.iter().cloned().fold(f64::INFINITY, f64::min);

    eprintln!("\n=== Final Hidden State Statistics ===");
    eprintln!("FHE hidden state:");
    eprintln!("  Mean:  {:.6}", fhe_mean);
    eprintln!("  Min:   {:.6}", fhe_min);
    eprintln!("  Max:   {:.6}", fhe_max);
    eprintln!("  Length: {}", fhe_hidden_f64.len());

    Ok(())
}
