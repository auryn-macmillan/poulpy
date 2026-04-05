use fhe_llm::{
    attention::SoftmaxStrategy,
    inference::{InferenceConfig, InferencePipeline, ModelSpec},
    params::{Precision, SecurityLevel},
};
use std::path::Path;

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let prompt = args.get(0).map(|s| s.as_str()).unwrap_or("2+2=");
    let precision_str = args.get(1).map(|s| s.as_str()).unwrap_or("fp16");
    let precision = match precision_str.to_lowercase().as_str() {
        "fp16" => Precision::Fp16,
        _ => Precision::Int8,
    };

    let d_models = vec![64, 128, 256, 512, 576];
    let n_layers = 1;

    eprintln!("=== FP16 d_model sweep across {precision:?} ===");
    eprintln!("Prompt: {prompt:?}");
    eprintln!("Layers: {n_layers}");
    eprintln!();

    for &d_model in &d_models {
        eprintln!("========================================");
        eprintln!("Testing d_model={d_model}");
        eprintln!("========================================");
        test_d_model(d_model, n_layers, prompt, precision)?;
        eprintln!();
    }

    Ok(())
}

fn test_d_model(d_model: usize, n_layers: usize, prompt: &str, precision: Precision) -> Result<(), Box<dyn std::error::Error>> {
    let d_ffn = d_model * 2;
    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_ffn),
        num_heads: Some(n_heads),
        num_kv_heads: Some(n_kv_heads),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        fhe_silu_log_msg_mod: Some(13),
        fhe_identity_log_msg_mod: Some(13),
        fhe_frequent_bootstrap: false,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    let mut pipeline = InferencePipeline::load(
        Path::new(MODEL_PATH),
        Path::new(TOKENIZER_PATH),
        ModelSpec::smollm2_135m_instruct(),
        config,
    )?;

    let start = std::time::Instant::now();
    let result = pipeline.step_refreshed(33)?; // "2+2=" token
    let elapsed = start.elapsed();

    eprintln!("[FHE] Time: {:.1}s", elapsed.as_secs_f64());
    eprintln!("[FHE] Token: {} (ID: {})", result.token_text, result.token_id);
    eprintln!("[FHE] Top-5 logits:");
    for (tid, _) in result.top_logits.iter().take(5) {
        eprintln!("  {tid}: {}", pipeline.decode_token(*tid).unwrap_or_default());
    }

    // Exact plaintext for comparison
    let exact_result = pipeline.plaintext_step(33);
    eprintln!(
        "[Exact] Token: {} (ID: {})",
        pipeline.decode_token(exact_result.token_id as u32).unwrap_or_default(),
        exact_result.token_id
    );

    let match_str = if result.token_id == exact_result.token_id as u32 {
        "✅ MATCH"
    } else {
        "❌ MISMATCH"
    };
    eprintln!("[Match] {match_str}");

    Ok(())
}
