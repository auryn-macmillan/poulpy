/// Plaintext-only prompt comparison for SmolLM2-135M-Instruct using a manually constructed ModelSpec.
///
/// Usage:
///   cargo run --release --example smollm2_manual --raw "2+2=" 128 256 1
///
/// This example demonstrates how to load the model when the built‑in ModelSpec factory is missing.
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
    // Manual ModelSpec for SmolLM2-135M-Instruct
    let manual_spec = ModelSpec {
        dims: fhe_llm::params::ModelDims::smollm2_135m_instruct(),
        embed_name: "model.embed_tokens.weight".to_string(),
        lm_head_name: "lm_head.weight".to_string(),
        final_norm_name: "model.norm.weight".to_string(),
        rope_theta: 10000.0,
        max_seq_len: 2048,
        bos_token_id: 1,
        eos_token_id: 2,
    };

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let raw_mode = args
        .iter()
        .position(|a| a == "--raw")
        .map(|i| {
            args.remove(i);
            true
        })
        .unwrap_or(false);
    let mut args_iter = args.into_iter();
    let user_prompt = args_iter.next().unwrap_or_else(|| "What is 2+2?".to_string());
    let d_model: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let d_ffn: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_layers: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    eprintln!("=== SmolLM2 Manual Spec Prompt Comparison ===");
    eprintln!("d_model={d_model}, d_ffn={d_ffn}, n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}");

    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_ffn),
        num_heads: Some(n_heads),
        num_kv_heads: Some(n_kv_heads),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 1,
        ..InferenceConfig::default()
    };

    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, manual_spec, config)?;
    eprintln!("[main] Loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    let formatted = if raw_mode {
        user_prompt.clone()
    } else {
        format!("User: {}\nAssistant:", user_prompt)
    };
    let prompt_tokens = pipeline.tokenize(&formatted)?;
    eprintln!(
        "[main] prompt={:?} ({} tokens): {:?}",
        user_prompt,
        prompt_tokens.len(),
        prompt_tokens
    );

    // Exact path (single step)
    let t = Instant::now();
    let exact = pipeline.plaintext_step(prompt_tokens[0]);
    let exact_time = t.elapsed();

    // Refreshed path (same as exact for now)
    let t = Instant::now();
    let refreshed = pipeline.plaintext_step(prompt_tokens[0]);
    let refresh_time = t.elapsed();

    let exact_h: Vec<f64> = exact.hidden_state.iter().map(|&v| v as f64).collect();
    let refresh_h: Vec<f64> = refreshed.hidden_state.iter().map(|&v| v as f64).collect();
    let (linf, l2, mae) = fhe_llm::plaintext_forward::error_metrics(&refresh_h, &exact_h);

    println!("=== RESULTS ===");
    println!("Config: d={d_model} L={n_layers} T={}", prompt_tokens.len());
    println!(
        "Exact:     token={} ({:.3}s) top5={:?}",
        exact.token_id,
        exact_time.as_secs_f64(),
        exact.top_logits,
    );
    println!(
        "Refreshed: token={} ({:.3}s) top5={:?}",
        refreshed.token_id,
        refresh_time.as_secs_f64(),
        refreshed.top_logits,
    );
    println!("Error: L-inf={linf:.3} L2={l2:.3} MAE={mae:.3}");
    println!("Match: {}", exact.token_id == refreshed.token_id);

    Ok(())
}
