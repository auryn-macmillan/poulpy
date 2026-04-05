use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

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
    if !std::path::Path::new(MODEL_PATH).exists() {
        return Err(format!("model file not found: {MODEL_PATH}").into());
    }
    if !std::path::Path::new(TOKENIZER_PATH).exists() {
        return Err(format!("tokenizer file not found: {TOKENIZER_PATH}").into());
    }

    let base_config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 2,
        ..InferenceConfig::default()
    };

    let prompt = "What is 2+2? Answer with one token.";
    let configs = [
        SweepConfig {
            d_model: 64,
            d_ffn: 160,
            layers: 1,
            heads: 1,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 128,
            d_ffn: 320,
            layers: 1,
            heads: 2,
            kv_heads: 1,
        },
        SweepConfig {
            d_model: 256,
            d_ffn: 640,
            layers: 1,
            heads: 4,
            kv_heads: 2,
        },
        SweepConfig {
            d_model: 576,
            d_ffn: 1536,
            layers: 1,
            heads: 9,
            kv_heads: 3,
        },
        SweepConfig {
            d_model: 576,
            d_ffn: 1536,
            layers: 2,
            heads: 9,
            kv_heads: 3,
        },
        SweepConfig {
            d_model: 576,
            d_ffn: 1536,
            layers: 30,
            heads: 9,
            kv_heads: 3,
        },
    ];

    println!("model: SmolLM2-135M-Instruct");
    println!("user prompt: {:?}", prompt);
    println!();

    for cfg in configs {
        let config = InferenceConfig {
            num_layers: Some(cfg.layers),
            trunc_d_model: Some(cfg.d_model),
            trunc_d_ffn: Some(cfg.d_ffn),
            num_heads: Some(cfg.heads),
            num_kv_heads: Some(cfg.kv_heads),
            ..base_config.clone()
        };

        let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
        let chat_prompt = pipeline.format_chat_prompt(
            Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face."),
            prompt,
        );
        let result = pipeline.generate_plaintext_prompt_refreshed(&chat_prompt, 2)?;

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
