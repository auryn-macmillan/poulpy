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
    let mut args = std::env::args().skip(1);
    let user_prompt = args
        .next()
        .unwrap_or_else(|| "What is 2+2? Answer with one token.".to_string());
    let d_model: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let d_ffn: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(d_model * 2);
    let n_layers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    // SmolLM2-135M has d_head=64, so n_heads = d_model / 64
    // GQA ratio is 9:3 = 3:1, so n_kv_heads = n_heads / 3 (min 1)
    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    println!("=== SmolLM2 FHE comparison ===");
    println!("d_model={d_model}, d_ffn={d_ffn}, n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}");

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

    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    let prompt = pipeline.format_chat_prompt(
        Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face."),
        &user_prompt,
    );

    println!("prompt: {:?}", user_prompt);
    println!("formatted prompt length: {} chars", prompt.len());

    let result = pipeline.compare_prompt_fhe_semantics(&prompt)?;

    println!("tokenized_prompt_len: {}", result.prompt_tokens.len());
    println!();
    println!(
        "refreshed: token={} {:?} top5={:?}",
        result.refreshed.token_id, result.refreshed.token_text, result.refreshed.top_logits,
    );
    println!(
        "fhe_like_plain: token={} {:?} top5={:?}",
        result.fhe_like_plain.token_id, result.fhe_like_plain.token_text, result.fhe_like_plain.top_logits,
    );
    println!(
        "fhe: token={} {:?} (fhe_time={:.2?}) top5={:?}",
        result.fhe.token_id, result.fhe.token_text, result.fhe.fhe_time, result.fhe.top_logits,
    );
    println!();
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
