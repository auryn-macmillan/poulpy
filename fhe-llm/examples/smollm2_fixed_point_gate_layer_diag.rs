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
    let fractional_bits = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(3);
    let compress_bits = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(0);

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
    let diag = pipeline.diagnose_prompt_fixed_point_gate_layers_with_compression(&prompt, fractional_bits, compress_bits)?;

    println!("prompt: {:?}", diag.prompt);
    println!("tokenized_prompt_len: {}", diag.prompt_tokens.len());
    println!(
        "message_domain: log_message_modulus={} range=[{}, {}] fractional_bits={} compress_bits={}",
        diag.log_message_modulus, diag.message_range_min, diag.message_range_max, diag.fractional_bits, compress_bits
    );
    for row in diag.rows {
        println!(
            "layer={} compress_bits={} gate=[{:.6}, {:.6}] encoded=[{}, {}] overflow={}/{} hidden(L-inf={:.6} L2={:.6} MAE={:.6})",
            row.layer,
            row.compress_bits,
            row.gate_min,
            row.gate_max,
            row.encoded_min,
            row.encoded_max,
            row.overflow_dims,
            row.total_dims,
            row.hidden_linf,
            row.hidden_l2,
            row.hidden_mae,
        );
    }

    Ok(())
}
