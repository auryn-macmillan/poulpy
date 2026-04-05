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
    let fractional_bits: Vec<usize> = args
        .next()
        .map(|s| {
            s.split(',')
                .filter(|part| !part.trim().is_empty())
                .map(|part| part.trim().parse::<usize>().expect("invalid fractional bits"))
                .collect()
        })
        .unwrap_or_else(|| vec![0, 1, 2, 3, 4, 5, 6]);

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
    let sweep = pipeline.sweep_prompt_adaptive_fixed_point_gate(&prompt, &fractional_bits)?;

    println!("prompt: {:?}", sweep.prompt);
    println!("tokenized_prompt_len: {}", sweep.prompt_tokens.len());
    println!(
        "message_domain: log_message_modulus={} range=[{}, {}]",
        sweep.log_message_modulus, sweep.message_range_min, sweep.message_range_max
    );
    for row in sweep.rows {
        println!(
            "q{}: final_layer compress_bits={} overflow={}/{} hidden(L-inf={:.6} L2={:.6} MAE={:.6})",
            row.fractional_bits,
            row.final_layer_compress_bits,
            row.final_layer_overflow_dims,
            row.final_layer_total_dims,
            row.final_hidden_linf,
            row.final_hidden_l2,
            row.final_hidden_mae,
        );
    }

    Ok(())
}
