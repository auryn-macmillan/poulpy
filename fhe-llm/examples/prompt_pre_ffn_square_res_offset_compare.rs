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
    let mut args = std::env::args().skip(1);
    let prompt = args.next().unwrap_or_else(|| "2+2 equals what?".to_string());
    let attn_decode = args.next().map(|s| s.parse::<u32>()).transpose()?.unwrap_or(8);
    let residual_decode = args.next().map(|s| s.parse::<u32>()).transpose()?.unwrap_or(6);
    let input_shift = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(0);
    let square_res_offset = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(4);

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
    let comparison = pipeline.compare_prompt_fhe_semantics_with_pre_ffn_square_res_offset(
        &prompt,
        attn_decode,
        residual_decode,
        input_shift,
        square_res_offset,
    )?;

    println!("prompt: {:?}", comparison.prompt);
    println!("prompt_len: {}", comparison.prompt_tokens.len());
    println!(
        "config: attn_decode={} residual_decode={} input_shift={} square_res_offset={}",
        attn_decode, residual_decode, input_shift, square_res_offset
    );
    println!(
        "refreshed token: {} {:?}",
        comparison.refreshed.token_id, comparison.refreshed.token_text
    );
    println!(
        "fhe_like token: {} {:?}",
        comparison.fhe_like_plain.token_id, comparison.fhe_like_plain.token_text
    );
    println!("fhe token: {} {:?}", comparison.fhe.token_id, comparison.fhe.token_text);
    println!(
        "fhe_like_vs_fhe: L-inf={:.3} L2={:.3} MAE={:.3}",
        comparison.fhe_like_vs_fhe.0, comparison.fhe_like_vs_fhe.1, comparison.fhe_like_vs_fhe.2
    );
    println!(
        "refreshed_vs_fhe_like: L-inf={:.3} L2={:.3} MAE={:.3}",
        comparison.refreshed_vs_fhe_like.0, comparison.refreshed_vs_fhe_like.1, comparison.refreshed_vs_fhe_like.2
    );
    println!("fhe_like_token_match: {}", comparison.fhe_like_token_match);
    println!("refreshed_token_match: {}", comparison.refreshed_token_match);

    Ok(())
}
