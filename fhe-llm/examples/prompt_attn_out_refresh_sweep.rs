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
    .unwrap_or_else(|| vec![2, 4, 6, 8])
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
    for precision in precisions {
        let result = pipeline.compare_prompt_fhe_semantics_with_attn_out_local_refresh(&prompt, precision)?;
        println!(
            "attn_out_decode_precision={}: refreshed={} {:?} | fhe={} {:?} | fhe_like_vs_fhe(L-inf={:.3} L2={:.3} MAE={:.3}) | refreshed_match={}",
            precision,
            result.refreshed.token_id,
            result.refreshed.token_text,
            result.fhe.token_id,
            result.fhe.token_text,
            result.fhe_like_vs_fhe.0,
            result.fhe_like_vs_fhe.1,
            result.fhe_like_vs_fhe.2,
            result.refreshed_token_match,
        );
    }

    Ok(())
}
