use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

fn parse_precisions(arg: Option<String>, default: &[u32]) -> Vec<u32> {
    arg.map(|s| {
        s.split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u32>().expect("invalid precision"))
            .collect()
    })
    .unwrap_or_else(|| default.to_vec())
}

fn format_precision(value: Option<u32>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "none".to_string())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let prompt = args.next().unwrap_or_else(|| "2+2 equals what?".to_string());
    let context_precisions = parse_precisions(args.next(), &[0, 6]);
    let attn_precisions = parse_precisions(args.next(), &[0, 6, 8]);
    let residual_precisions = parse_precisions(args.next(), &[2, 4, 6, 8]);

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
    let sweep =
        pipeline.sweep_prompt_attention_residual_bridge(&prompt, &context_precisions, &attn_precisions, &residual_precisions)?;

    println!("prompt: {:?}", sweep.prompt);
    println!("tokenized_prompt_len: {}", sweep.prompt_tokens.len());
    for row in sweep.rows {
        println!(
            "mode={} context_decode={} attn_decode={} residual_decode={}: attn_out(L-inf={:.3} L2={:.3} MAE={:.3}) residual(L-inf={:.3} L2={:.3} MAE={:.3}) pre_ffn(L-inf={:.3} L2={:.3} MAE={:.3})",
            row.projection_mode,
            format_precision(row.context_refresh_decode_precision),
            format_precision(row.attn_out_decode_precision),
            row.residual_refresh_decode_precision,
            row.attn_out_linf,
            row.attn_out_l2,
            row.attn_out_mae,
            row.residual_linf,
            row.residual_l2,
            row.residual_mae,
            row.pre_ffn_linf,
            row.pre_ffn_l2,
            row.pre_ffn_mae,
        );
    }

    Ok(())
}
