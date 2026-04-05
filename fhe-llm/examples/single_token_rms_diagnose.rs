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
    let token_id = std::env::args().nth(1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(29889);

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

    println!("token_id: {}", token_id);
    println!("calibrated:");
    for row in pipeline.diagnose_pre_attn_rms_internals_calibrated_for_token(token_id) {
        println!("  {} = {}", row.stage, row.value);
    }
    println!("remap alternatives:");
    for row in pipeline.diagnose_pre_attn_rms_remap_alternatives_for_token(token_id) {
        println!("  {} = {}", row.stage, row.value);
    }
    println!("shift sweep @10:");
    for row in pipeline.diagnose_pre_attn_rms_shift_sweep_for_token(token_id, 10, 8) {
        println!("  {} = {}", row.stage, row.value);
    }
    println!("shift sweep @18:");
    for row in pipeline.diagnose_pre_attn_rms_shift_sweep_for_token(token_id, 18, 8) {
        println!("  {} = {}", row.stage, row.value);
    }
    println!("shift+poly sweep:");
    for row in pipeline.diagnose_pre_attn_rms_shift_poly_sweep_for_token(token_id, 8) {
        println!("  {} = {}", row.stage, row.value);
    }
    for precision in [
        2u32, 4u32, 6u32, 8u32, 10u32, 12u32, 14u32, 16u32, 18u32, 20u32, 22u32, 24u32, 26u32,
    ] {
        let rows = pipeline.diagnose_pre_attn_rms_internals_for_token(token_id, precision);
        println!("precision={}", precision);
        for row in rows {
            println!("  {} = {}", row.stage, row.value);
        }
    }

    Ok(())
}
