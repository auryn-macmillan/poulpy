use std::process;

use fhe_llm::activations::inv_sqrt_poly_approx;
use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::layernorm::LayerNormConfig;
use fhe_llm::params::{Precision, SecurityLevel};

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
    let range32 = LayerNormConfig {
        norm_size: 64,
        use_rms_norm: true,
        inv_sqrt_approx: inv_sqrt_poly_approx(),
        epsilon: 1e-5,
        gamma: None,
        beta: None,
    };

    let baseline =
        pipeline.compare_prompt_fhe_semantics_with_pre_ffn_config(&prompt, attn_decode, residual_decode, range32.clone())?;
    let refreshed = pipeline.compare_prompt_fhe_semantics_with_pre_ffn_config_and_pre_ffn_refresh(
        &prompt,
        attn_decode,
        residual_decode,
        range32,
    )?;

    println!("prompt: {:?}", prompt);
    println!("config: attn_decode={} residual_decode={}", attn_decode, residual_decode);
    println!(
        "baseline tokens: refreshed={} {:?} fhe_like={} {:?} fhe={} {:?}",
        baseline.refreshed.token_id,
        baseline.refreshed.token_text,
        baseline.fhe_like_plain.token_id,
        baseline.fhe_like_plain.token_text,
        baseline.fhe.token_id,
        baseline.fhe.token_text
    );
    println!(
        "baseline errors: fhe_like_vs_fhe L-inf={:.3} L2={:.3} MAE={:.3} | refreshed_vs_fhe_like L-inf={:.3} L2={:.3} MAE={:.3}",
        baseline.fhe_like_vs_fhe.0,
        baseline.fhe_like_vs_fhe.1,
        baseline.fhe_like_vs_fhe.2,
        baseline.refreshed_vs_fhe_like.0,
        baseline.refreshed_vs_fhe_like.1,
        baseline.refreshed_vs_fhe_like.2,
    );
    println!(
        "pre_ffn_refresh tokens: refreshed={} {:?} fhe_like={} {:?} fhe={} {:?}",
        refreshed.refreshed.token_id,
        refreshed.refreshed.token_text,
        refreshed.fhe_like_plain.token_id,
        refreshed.fhe_like_plain.token_text,
        refreshed.fhe.token_id,
        refreshed.fhe.token_text
    );
    println!(
        "pre_ffn_refresh errors: fhe_like_vs_fhe L-inf={:.3} L2={:.3} MAE={:.3} | refreshed_vs_fhe_like L-inf={:.3} L2={:.3} MAE={:.3}",
        refreshed.fhe_like_vs_fhe.0,
        refreshed.fhe_like_vs_fhe.1,
        refreshed.fhe_like_vs_fhe.2,
        refreshed.refreshed_vs_fhe_like.0,
        refreshed.refreshed_vs_fhe_like.1,
        refreshed.refreshed_vs_fhe_like.2,
    );

    Ok(())
}
