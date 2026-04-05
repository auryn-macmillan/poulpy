//! Diagnostic: Check weight quantization and hidden state source of 127.0000

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
    let use_fp16 = args.iter().any(|s| s == "--fp16");
    let user_prompt = args
        .iter()
        .find(|s| !s.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "2+2=".to_string());
    let d_model: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_layers: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let d_ffn = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(d_model * 2);

    let precision = if use_fp16 {
        eprintln!("=== FP16 Weight Quantization Diagnostic ===");
        Precision::Fp16
    } else {
        eprintln!("=== INT8 Weight Quantization Diagnostic ===");
        Precision::Int8
    };

    eprintln!(
        "d_model={d_model}, d_ffn={d_ffn}, n_layers={n_layers}, precision={:?}\n",
        precision
    );
    eprintln!("prompt: {:?}\n", user_prompt);

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_ffn),
        num_heads: Some((d_model / 64).max(1)),
        num_kv_heads: Some(((d_model / 64) / 3).max(1)),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
        fhe_silu_log_msg_mod: if use_fp16 { Some(13) } else { Some(7) },
        fhe_identity_log_msg_mod: if use_fp16 { Some(13) } else { Some(7) },
    };

    eprintln!("\n[diag] Creating InferencePipeline with precision={:?}...", precision);
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;

    let scale_bits = pipeline.params().scale_bits;
    let log_msg_mod = scale_bits as usize - 1;
    eprintln!("\n[diag] Pipeline parameters:");
    eprintln!("  scale_bits: {}", scale_bits);
    eprintln!("  k_pt: {}", pipeline.params().k_pt.0);
    eprintln!(
        "  log_message_modulus (bootstrap): {} (derived from scale_bits - 1)",
        log_msg_mod
    );
    eprintln!("  encoding_scale: {}", pipeline.params().encoding_scale());
    eprintln!("  Expected bootstrap levels: {}", 2usize.pow(log_msg_mod as u32));

    eprintln!("\n[diag] Running FHE forward pass with layer capture...");
    let layer_results = pipeline.compare_prompt_fhe_layer_hidden_states(&user_prompt)?;

    eprintln!("\n[diag] Per-layer hidden state distribution:");
    for (layer_name, hidden) in &layer_results {
        let hidden_f64: Vec<f64> = hidden.iter().map(|&v| v as f64).collect();
        let mean: f64 = hidden_f64.iter().sum::<f64>() / hidden_f64.len() as f64;
        let max: f64 = hidden_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min: f64 = hidden_f64.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_abs: f64 = hidden_f64.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let at_127 = hidden_f64.iter().filter(|v| (v.abs() - 127.0).abs() < 0.01).count();
        let at_64 = hidden_f64.iter().filter(|v| (v.abs() - 64.0).abs() < 0.01).count();
        let at_32 = hidden_f64.iter().filter(|v| (v.abs() - 32.0).abs() < 0.01).count();

        eprintln!(
            "  {:12} | Mean: {:6.2} | Min: {:6.2} | Max: {:6.2} | MaxAbs: {:6.2} | At_±127: {:5}/{} | At_±64: {:5} | At_±32: {:5}",
            layer_name, mean, min, max, max_abs, at_127, hidden_f64.len(), at_64, at_32
        );
    }

    Ok(())
}
