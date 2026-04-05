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

    let pipeline = InferencePipeline::load(model_path, tokenizer_path, ModelSpec::tinyllama_1_1b(), config)?;
    let result = pipeline.compare_prompt_fhe_semantics(&prompt)?;

    println!("prompt: {:?}", result.prompt);
    println!("tokenized_prompt_len: {}", result.prompt_tokens.len());
    println!(
        "refreshed: token={} {:?} top5={:?}",
        result.refreshed.token_id, result.refreshed.token_text, result.refreshed.top_logits,
    );
    println!(
        "fhe_like_plain: token={} {:?} top5={:?}",
        result.fhe_like_plain.token_id, result.fhe_like_plain.token_text, result.fhe_like_plain.top_logits,
    );
    println!(
        "fhe: token={} {:?} top5={:?}",
        result.fhe.token_id, result.fhe.token_text, result.fhe.top_logits,
    );
    println!(
        "fhe_like_vs_fhe: L-inf={:.3} L2={:.3} MAE={:.3}",
        result.fhe_like_vs_fhe.0, result.fhe_like_vs_fhe.1, result.fhe_like_vs_fhe.2,
    );
    println!(
        "refreshed_vs_fhe_like: L-inf={:.3} L2={:.3} MAE={:.3}",
        result.refreshed_vs_fhe_like.0, result.refreshed_vs_fhe_like.1, result.refreshed_vs_fhe_like.2,
    );
    println!("fhe_like_token_match: {}", result.fhe_like_token_match);
    println!("refreshed_token_match: {}", result.refreshed_token_match);

    Ok(())
}
