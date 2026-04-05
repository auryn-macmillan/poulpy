use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

fn parse_precisions(arg: Option<String>) -> Vec<u32> {
    arg.map(|s| {
        s.split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u32>().expect("invalid precision"))
            .collect()
    })
    .unwrap_or_else(|| vec![8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
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
    let precisions = parse_precisions(args.next());

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
    let sweep = pipeline.sweep_prompt_attention_stage_decode(&prompt, &precisions)?;

    println!("prompt: {:?}", sweep.prompt);
    println!("tokenized_prompt_len: {}", sweep.prompt_tokens.len());
    for row in sweep.rows {
        println!(
            "precision={}: pre_attn(L-inf={:.3} L2={:.3} MAE={:.3}) q(L-inf={:.3} L2={:.3} MAE={:.3}) k(L-inf={:.3} L2={:.3} MAE={:.3})",
            row.precision,
            row.pre_attn_norm.0,
            row.pre_attn_norm.1,
            row.pre_attn_norm.2,
            row.q_head.0,
            row.q_head.1,
            row.q_head.2,
            row.k_head.0,
            row.k_head.1,
            row.k_head.2,
        );
    }

    Ok(())
}
