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

    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(30),
        trunc_d_model: Some(576),
        trunc_d_ffn: Some(1536),
        num_heads: Some(9),
        num_kv_heads: Some(3),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 2,
        ..InferenceConfig::default()
    };

    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    let prompt = pipeline.format_chat_prompt(
        Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face."),
        &user_prompt,
    );
    let result = pipeline.compare_prompt_output_impact(&prompt)?;

    println!("prompt: {:?}", user_prompt);
    println!("tokenized_prompt_len: {}", result.prompt_tokens.len());
    println!(
        "exact: token={} text={:?} runner_up={} {:?} margin={:.6} top_logits={:?}",
        result.exact.token_id,
        result.exact.token_text,
        result.exact_top2_token_id,
        result.exact_top2_token_text,
        result.exact_top_margin_f64,
        result.exact.top_logits
    );
    for row in result.rows {
        println!(
            "variant={} match={} token={} text={:?} hidden(L-inf={:.6} L2={:.6} MAE={:.6}) exact_token_rank={} predicted_logit={:.6} exact_token_logit={:.6} top_logits={:?}",
            row.variant,
            row.token_match,
            row.token_id,
            row.token_text,
            row.hidden_linf,
            row.hidden_l2,
            row.hidden_mae,
            row.exact_token_rank,
            row.predicted_logit_f64,
            row.exact_token_logit_f64,
            row.top_logits,
        );
    }

    Ok(())
}
