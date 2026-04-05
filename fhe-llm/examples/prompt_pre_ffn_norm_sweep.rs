use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

fn parse_u32s(arg: Option<String>, default: &[u32]) -> Vec<u32> {
    arg.map(|s| {
        s.split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u32>().expect("invalid u32"))
            .collect()
    })
    .unwrap_or_else(|| default.to_vec())
}

fn parse_usizes(arg: Option<String>, default: &[usize]) -> Vec<usize> {
    arg.map(|s| {
        s.split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<usize>().expect("invalid usize"))
            .collect()
    })
    .unwrap_or_else(|| default.to_vec())
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
    let attn_precisions = parse_u32s(args.next(), &[2, 4, 6, 8]);
    let residual_precisions = parse_u32s(args.next(), &[2, 4, 6, 8]);
    let shift_bits = parse_usizes(args.next(), &[0, 1, 2, 4]);

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
    let sweep = pipeline.sweep_prompt_pre_ffn_norm_variants(&prompt, &attn_precisions, &residual_precisions, &shift_bits)?;

    println!("prompt: {:?}", sweep.prompt);
    println!("tokenized_prompt_len: {}", sweep.prompt_tokens.len());
    for row in sweep.rows {
        println!(
            "attn_decode={} residual_decode={} shift={}: pre_ffn(L-inf={:.3} L2={:.3} MAE={:.3})",
            row.attn_out_decode_precision,
            row.residual_refresh_decode_precision,
            row.input_scale_shift_bits,
            row.pre_ffn_linf,
            row.pre_ffn_l2,
            row.pre_ffn_mae,
        );
    }

    Ok(())
}
