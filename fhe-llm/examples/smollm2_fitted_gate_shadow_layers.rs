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
    let degree = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(3);
    let clip_abs = args.next().map(|s| s.parse::<f64>()).transpose()?;

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
    let result = pipeline.compare_prompt_fitted_gate_shadow_layers_with_clip(&prompt, degree, clip_abs)?;

    println!("prompt: {:?}", result.prompt);
    println!("tokenized_prompt_len: {}", result.prompt_tokens.len());
    println!(
        "fitted_deg{} variant={} coeffs={:?} range=[{:.6}, {:.6}] sample_count={} fit(L-inf={:.6} L2={:.6} MAE={:.6})",
        degree,
        result.approx.variant,
        result.approx.coeffs,
        result.approx.range.0,
        result.approx.range.1,
        result.approx.sample_count,
        result.approx.max_error,
        result.approx.l2_error,
        result.approx.mae_error,
    );
    for row in result.rows {
        println!("{}: L-inf={:.6} L2={:.6} MAE={:.6}", row.stage, row.linf, row.l2, row.mae);
    }

    Ok(())
}
