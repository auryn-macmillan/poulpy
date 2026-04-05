use std::process;

use fhe_llm::activations::inv_sqrt_poly_approx;
use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::layernorm::LayerNormConfig;
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
    let attn_decode = args.next().map(|s| s.parse::<u32>()).transpose()?.unwrap_or(6);
    let residual_decode = args.next().map(|s| s.parse::<u32>()).transpose()?.unwrap_or(4);

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
    let d = 64usize;
    let midrange = LayerNormConfig::rms_norm_midrange(d);
    let range32 = LayerNormConfig {
        norm_size: d,
        use_rms_norm: true,
        inv_sqrt_approx: inv_sqrt_poly_approx(),
        epsilon: 1e-5,
        gamma: None,
        beta: None,
    };

    let midrange_result = pipeline.diagnose_prompt_first_layer_hidden_branch_replay_with_pre_ffn_config(
        &prompt,
        attn_decode,
        residual_decode,
        midrange,
    )?;
    let range32_result = pipeline.diagnose_prompt_first_layer_hidden_branch_replay_with_pre_ffn_config(
        &prompt,
        attn_decode,
        residual_decode,
        range32,
    )?;

    println!("prompt: {:?}", prompt);
    println!("config: attn_decode={} residual_decode={}", attn_decode, residual_decode);
    println!("midrange:");
    for row in midrange_result.rows {
        println!("  {}: L-inf={:.3} L2={:.3} MAE={:.3}", row.stage, row.linf, row.l2, row.mae);
    }
    println!("range32:");
    for row in range32_result.rows {
        println!("  {}: L-inf={:.3} L2={:.3} MAE={:.3}", row.stage, row.linf, row.l2, row.mae);
    }

    Ok(())
}
