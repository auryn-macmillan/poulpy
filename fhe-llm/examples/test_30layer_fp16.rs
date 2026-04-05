use std::process;

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
    let prompt = args.get(0).map(|s| s.as_str()).unwrap_or("2+2=");
    let n_layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let d_model: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    let precision = Precision::Fp16;
    let scale_bits = match precision {
        Precision::Int8 => 10,
        Precision::Fp16 => 14,
    };
    let log_msg_mod = scale_bits - 1;

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_model * 2),
        num_heads: Some((d_model / 64).max(1)),
        num_kv_heads: Some(((d_model / 64) / 3).max(1)),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 0,
        fhe_silu_log_msg_mod: Some(log_msg_mod as u32),
        fhe_identity_log_msg_mod: Some(log_msg_mod as u32),
        fhe_frequent_bootstrap: false,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("=== 30-Layer FHE Test (d={d_model}, layers={n_layers}, FP16) ===",);
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!();

    eprintln!("[main] Loading model...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", std::time::Instant::now().elapsed().as_secs_f64());

    // Run FHE inference
    eprintln!("[fhe] Running FHE inference...");
    let start = std::time::Instant::now();
    let result = pipeline.step(45)?;
    let fhe_elapsed = start.elapsed();
    eprintln!("[fhe] Complete in {:.1}s\n", fhe_elapsed.as_secs_f64());

    // Decode token
    let token_id = result.token_id;
    let logits = &result.top_logits;
    let hidden_state = result.hidden_state;
    let token_text = result.token_text;
    eprintln!("[fhe] Decoded token: \"{token_text}\" (ID: {token_id})");
    let (top1_id, top1_logit) = logits[0];
    eprintln!("  top-1: token={top1_id} logit={top1_logit} text=\"{token_text}\"");

    // Hidden state stats
    let max_val: f64 = hidden_state
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, |acc, v| f64::max(acc, v.abs()));
    let l2_norm = (hidden_state.iter().map(|v| v * v).sum::<f64>()).sqrt();
    eprintln!("\n=== Hidden State Summary ===");
    eprintln!("Final hidden state: max={:.2}, L2_norm={:.2}", max_val, l2_norm);

    Ok(())
}
