use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

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
    let attn_decode = args.next().map(|s| s.parse::<u32>()).transpose()?.unwrap_or(6);
    let residual_decode = args.next().map(|s| s.parse::<u32>()).transpose()?.unwrap_or(4);
    let mean_sq_shifts = parse_usizes(args.next(), &[4, 5, 6, 7, 8, 9]);
    let square_res_offset = args.next().map(|s| s.parse::<usize>()).transpose()?.unwrap_or(18);
    let final_res_offsets = parse_usizes(args.next(), &[2, 4, 10, 18, 26]);

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
    let result = pipeline.compare_prompt_pre_ffn_norm_mean_sq_remap_variants(
        &prompt,
        attn_decode,
        residual_decode,
        &mean_sq_shifts,
        square_res_offset,
        &final_res_offsets,
    )?;

    println!("prompt: {:?}", result.prompt);
    println!("tokenized_prompt_len: {}", result.prompt_tokens.len());
    println!(
        "config: attn_decode={} residual_decode={} square_res_offset={} mean_sq_shifts={:?} final_res_offsets={:?}",
        attn_decode, residual_decode, square_res_offset, mean_sq_shifts, final_res_offsets
    );
    for row in result.rows {
        println!("{}: L-inf={:.3} L2={:.3} MAE={:.3}", row.stage, row.linf, row.l2, row.mae);
    }

    Ok(())
}
