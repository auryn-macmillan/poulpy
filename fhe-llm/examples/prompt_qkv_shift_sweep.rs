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
    .unwrap_or_else(|| vec![0, 4, 8, 10])
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "What is 2+2? Answer with one token.".to_string());
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
    let sweep = pipeline.sweep_prompt_fhe_qkv_shift(&prompt, &shifts)?;

    println!("prompt: {:?}", sweep.prompt);
    println!("tokenized_prompt_len: {}", sweep.prompt_tokens.len());
    println!(
        "plaintext: token={} {:?}",
        sweep.plaintext_token_id, sweep.plaintext_token_text
    );
    for row in sweep.rows {
        println!(
            "qkv_shift={}: token={} {:?} | L-inf={:.3} L2={:.3} MAE={:.3} | plaintext_rank={} pred_logit={:.4} plaintext_logit={:.4}",
            row.qkv_shift_bits,
            row.token_id,
            row.token_text,
            row.linf,
            row.l2,
            row.mae,
            row.plaintext_token_rank,
            row.predicted_logit,
            row.plaintext_logit,
        );
    }

    Ok(())
}
