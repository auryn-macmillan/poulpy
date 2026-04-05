use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str =
    "/home/dev/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";

#[derive(Clone, Copy)]
struct SweepConfig {
    d_model: usize,
    d_ffn: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "2+2=".to_string());

    if !std::path::Path::new(MODEL_PATH).exists() {
        return Err(format!("model file not found: {MODEL_PATH}").into());
    }
    if !std::path::Path::new(TOKENIZER_PATH).exists() {
        return Err(format!("tokenizer file not found: {TOKENIZER_PATH}").into());
    }

    let configs = [
        SweepConfig {
            d_model: 64,
            d_ffn: 128,
            layers: 1,
            heads: 1,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 128,
            d_ffn: 352,
            layers: 1,
            heads: 2,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 128,
            d_ffn: 352,
            layers: 2,
            heads: 2,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 256,
            d_ffn: 704,
            layers: 1,
            heads: 4,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 256,
            d_ffn: 704,
            layers: 2,
            heads: 4,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 512,
            d_ffn: 1408,
            layers: 1,
            heads: 8,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 512,
            d_ffn: 1408,
            layers: 2,
            heads: 8,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 1024,
            d_ffn: 2816,
            layers: 1,
            heads: 16,
            kv_heads: 2,
        },
        SweepConfig {
            d_model: 1024,
            d_ffn: 2816,
            layers: 2,
            heads: 16,
            kv_heads: 2,
        },
    ];

    println!("prompt: {:?}", prompt);
    println!();

    for cfg in configs {
        let config = InferenceConfig {
            security: SecurityLevel::Bits80,
            precision: Precision::Int8,
            num_layers: Some(cfg.layers),
            trunc_d_model: Some(cfg.d_model),
            trunc_d_ffn: Some(cfg.d_ffn),
            num_heads: Some(cfg.heads),
            num_kv_heads: Some(cfg.kv_heads),
            softmax_strategy: SoftmaxStrategy::ReluSquared,
            apply_final_norm: true,
            max_new_tokens: 2,
            ..InferenceConfig::default()
        };

        let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::tinyllama_1_1b(), config)?;
        let result = pipeline.generate_plaintext_prompt_refreshed(&prompt, 2)?;

        println!(
            "cfg d_model={} d_ffn={} layers={} heads={} kv_heads={} | time={:.6}s | generated={:?} | full={:?}",
            cfg.d_model,
            cfg.d_ffn,
            cfg.layers,
            cfg.heads,
            cfg.kv_heads,
            result.total_time.as_secs_f64(),
            result.generated_text,
            result.full_text
        );

        if let Some(step) = result.steps.first() {
            println!(
                "  first token: {} {:?} | top5={:?}",
                step.token_id, step.token_text, step.top_logits
            );
        }
    }

    Ok(())
}
