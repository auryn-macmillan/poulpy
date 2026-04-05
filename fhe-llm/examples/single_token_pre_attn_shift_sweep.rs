use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

fn parse_shifts(arg: Option<String>) -> Vec<usize> {
    arg.map(|s| {
        s.split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<usize>().expect("invalid shift"))
            .collect()
    })
    .unwrap_or_else(|| vec![3, 4, 5, 8])
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let token_id = args.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(29889);
    let shifts = parse_shifts(args.next());

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
    let sweep = pipeline.sweep_single_token_pre_attn_remap_shift(token_id, &shifts)?;

    println!("token_id: {}", sweep.token_id);
    println!(
        "refreshed: token={} {:?}",
        sweep.refreshed_token_id, sweep.refreshed_token_text
    );
    for row in sweep.rows {
        println!(
            "shift={}: token={} {:?} | L-inf={:.3} L2={:.3} MAE={:.3} | refreshed_rank={} pred_logit={:.4} refreshed_logit={:.4}",
            row.remap_shift_bits,
            row.token_id,
            row.token_text,
            row.linf,
            row.l2,
            row.mae,
            row.refreshed_token_rank,
            row.predicted_logit,
            row.refreshed_logit,
        );
    }

    Ok(())
}
