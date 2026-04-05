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
    let probe = pipeline.probe_prompt_attention_head0(&prompt)?;

    println!("prompt: {:?}", probe.prompt);
    println!("tokenized_prompt_len: {}", probe.prompt_tokens.len());
    println!("plaintext_scores: {:?}", probe.plaintext_scores);
    println!("plaintext_configured_weights: {:?}", probe.plaintext_configured_weights);
    println!("plaintext_exact_weights: {:?}", probe.plaintext_exact_weights);
    println!("plaintext_quantized_scores: {:?}", probe.plaintext_quantized_scores);
    println!(
        "plaintext_quantized_configured_weights: {:?}",
        probe.plaintext_quantized_configured_weights
    );
    println!(
        "pre_attn_norm_quantized_error: L-inf={:.3} L2={:.3} MAE={:.3}",
        probe.pre_attn_norm_quantized_error.0, probe.pre_attn_norm_quantized_error.1, probe.pre_attn_norm_quantized_error.2,
    );
    println!(
        "q_head_quantized_error: L-inf={:.3} L2={:.3} MAE={:.3}",
        probe.q_head_quantized_error.0, probe.q_head_quantized_error.1, probe.q_head_quantized_error.2,
    );
    println!(
        "k_head_quantized_error: L-inf={:.3} L2={:.3} MAE={:.3}",
        probe.k_head_quantized_error.0, probe.k_head_quantized_error.1, probe.k_head_quantized_error.2,
    );
    println!("fhe_scores: {:?}", probe.fhe_scores);
    println!("fhe_weight_numerators: {:?}", probe.fhe_weight_numerators);
    println!("fhe_weight_numerator_sum: {:.6}", probe.fhe_weight_numerator_sum);

    Ok(())
}
