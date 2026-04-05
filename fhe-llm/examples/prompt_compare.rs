use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "2+2 equals what?".to_string());

    let model_path = "/home/dev/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";
    let tokenizer_path = "/home/dev/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";

    if !std::path::Path::new(model_path).exists() {
        return Err(format!("model file not found: {model_path}").into());
    }
    if !std::path::Path::new(tokenizer_path).exists() {
        return Err(format!("tokenizer file not found: {tokenizer_path}").into());
    }

    let config = InferenceConfig {
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
        ..InferenceConfig::default()
    };

    eprintln!("[prompt_compare] loading pipeline...");
    let pipeline = InferencePipeline::load(model_path, tokenizer_path, ModelSpec::tinyllama_1_1b(), config)?;

    let benchmark = pipeline.benchmark_prompt_sequence_refreshed(&prompt)?;

    println!("prompt: {:?}", benchmark.prompt);
    println!("prompt_tokens: {:?}", benchmark.prompt_tokens);
    println!("last_token_used: {} {:?}", benchmark.last_token, benchmark.last_token_text);
    println!();

    println!(
        "plaintext next token: {} {:?}",
        benchmark.plaintext.token_id, benchmark.plaintext.token_text
    );
    println!("plaintext step time: {:.6}s", benchmark.plaintext.total_time.as_secs_f64());
    println!("plaintext top-5: {:?}", benchmark.plaintext.top_logits);
    println!();

    println!("fhe next token: {} {:?}", benchmark.fhe.token_id, benchmark.fhe.token_text);
    println!("fhe core time: {:.6}s", benchmark.fhe.fhe_time.as_secs_f64());
    println!("fhe total time: {:.6}s", benchmark.fhe.total_time.as_secs_f64());
    println!("fhe top-5: {:?}", benchmark.fhe.top_logits);
    println!();

    println!("token_match: {}", benchmark.token_match);
    println!(
        "hidden_error: L-inf={:.3} L2={:.3} MAE={:.3}",
        benchmark.hidden_linf, benchmark.hidden_l2, benchmark.hidden_mae
    );
    println!("total_overhead_vs_plaintext: {:.1}x", benchmark.total_overhead_ratio);
    println!("fhe_core_overhead_vs_plaintext: {:.1}x", benchmark.fhe_core_overhead_ratio);
    println!();
    println!(
        "note: this toy path now runs a short causal prompt-conditioned pass over the full prompt, but only for very small truncated settings"
    );

    Ok(())
}
