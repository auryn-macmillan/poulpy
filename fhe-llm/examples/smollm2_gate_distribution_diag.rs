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
    let diag = pipeline.diagnose_prompt_gate_distribution(&prompt)?;

    println!("prompt: {:?}", diag.prompt);
    println!("tokenized_prompt_len: {}", diag.prompt_tokens.len());
    for row in diag.rows {
        println!(
            "layer={} gate[min={:.6} max={:.6} mean={:.6} std={:.6} |abs| p50={:.6} p90={:.6} p99={:.6} max={:.6}] narrow(MAE={:.6}) wide(MAE={:.6})",
            row.layer,
            row.gate_min,
            row.gate_max,
            row.gate_mean,
            row.gate_stddev,
            row.gate_abs_p50,
            row.gate_abs_p90,
            row.gate_abs_p99,
            row.gate_abs_max,
            row.exact_vs_narrow_mae,
            row.exact_vs_wide_mae,
        );
    }

    Ok(())
}
