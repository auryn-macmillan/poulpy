/// Full 30‑layer FHE inference for SmolLM2‑135M‑Instruct (truncated to d_model=128, d_ffn=256).
///
/// This example demonstrates the FHE forward pass over all 30 layers with frequent
/// refreshes, then returns the generated token and wall‑clock time.
///
/// Usage:
///   cargo run --release --example fhe_full_30layer --raw "2+2="
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
    // Manual ModelSpec for SmolLM2‑135M‑Instruct (truncated later).
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
    let prompt = if raw_mode {
        args.get(0).map(|s| s.as_str()).unwrap_or("2+2=")
    } else {
        "User: 2+2="
    };

    // Inference configuration – 30 layers, truncated dimensions.
    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(30),
        trunc_d_model: Some(128),
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 1,
        fhe_frequent_bootstrap: true,
        fhe_extra_refresh: true,
        fhe_silu_log_msg_mod: Some(18),
        fhe_identity_log_msg_mod: Some(18),
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
        ..InferenceConfig::default()
    };

    eprintln!("=== SmolLM2 30‑layer FHE inference (truncated) ===");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!("Config: d={}, L=30, T={}", 128, 1);
    eprintln!("Starting load...");

    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, manual_spec, config)?;
    eprintln!("[main] Loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    eprintln!("[main] Tokenizing prompt...");
    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("[main] Prompt tokens: {:?} ({} tokens)", prompt_tokens, prompt_tokens.len());

    eprintln!("[fhe] Running FHE forward pass over 30 layers (this may take several minutes)...");
    let fhe_start = Instant::now();
    let result = pipeline.generate(prompt, 1)?;
    let fhe_time = fhe_start.elapsed();

    eprintln!("[fhe] FHE generation completed in {:.1}s", fhe_time.as_secs_f64());
    eprintln!(
        "Generated token ID: {} ({})",
        result.generated_tokens[0], result.generated_text
    );

    Ok(())
}
