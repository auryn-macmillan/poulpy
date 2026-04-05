use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let prompt = args.get(0).map(|s| s.as_str()).unwrap_or("2+2=");
    let n_layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let d_model: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    let precision = Precision::Fp16;
    let scale_bits = match precision {
        Precision::Int8 => 10,
        Precision::Fp16 => 14,
    };
    let log_msg_mod = scale_bits - 1;

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_model * 2),
        num_heads: Some((d_model / 64).max(1)),
        num_kv_heads: Some(((d_model / 64) / 3).max(1)),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 0,
        fhe_silu_log_msg_mod: Some(log_msg_mod as u32),
        fhe_identity_log_msg_mod: Some(log_msg_mod as u32),
        fhe_frequent_bootstrap: false,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("=== Layer-by-Layer Hidden State Comparison (d={d_model}, layers={n_layers}, FP16) ===");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!();

    eprintln!("[main] Loading model...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config.clone())?;
    eprintln!("[main] Loaded in {:.1}s\n", std::time::Instant::now().elapsed().as_secs_f64());

    // Run FHE inference and capture hidden states at each layer
    eprintln!("[fhe] Running FHE inference with layer-by-layer capture...");
    let start = std::time::Instant::now();
    let fhe_hidden_states = pipeline.capture_hidden_states_at_layers(prompt)?;
    let fhe_elapsed = start.elapsed();
    eprintln!("[fhe] Complete in {:.1}s\n", fhe_elapsed.as_secs_f64());

    // Run exact HuggingFace inference for comparison
    eprintln!("[exact] Running exact HuggingFace inference for comparison...");
    let exact_pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    let exact_hidden_states = exact_pipeline.capture_hidden_states_at_layers(prompt)?;
    eprintln!(
        "[exact] Complete in {:.1}s\n",
        std::time::Instant::now().elapsed().as_secs_f64()
    );

    // Print layer-by-layer comparison
    eprintln!("\n=== Layer-by-Layer Hidden State Comparison ===");
    for (layer_idx, (fhe_hs, exact_hs)) in fhe_hidden_states.iter().zip(exact_hidden_states.iter()).enumerate() {
        let fhe_max: f64 = fhe_hs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let exact_max: f64 = exact_hs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let l_inf = fhe_hs
            .iter()
            .zip(exact_hs.iter())
            .map(|(f, e)| (f - e).abs())
            .fold(0.0f64, f64::max);
        let mae = fhe_hs.iter().zip(exact_hs.iter()).map(|(f, e)| (f - e).abs()).sum::<f64>() / fhe_hs.len() as f64;
        eprintln!(
            "Layer {}: FHE max={:.2}, Exact max={:.2}, L-inf={:.4}, MAE={:.4}",
            layer_idx, fhe_max, exact_max, l_inf, mae
        );
    }

    eprintln!("\n=== FHE Hidden State Summary ===");
    if let Some(last_layer) = fhe_hidden_states.last() {
        let max_val: f64 = last_layer
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, |acc, v| f64::max(acc, v.abs()));
        let l2_norm = (last_layer.iter().map(|v| v * v).sum::<f64>()).sqrt();
        eprintln!("Final layer hidden state: max={:.2}, L2_norm={:.2}", max_val, l2_norm);
    }

    eprintln!("\n=== FHE Hidden State Summary ===");
    if let Some(last_layer) = fhe_hidden_states.last() {
        let max_val: f64 = last_layer.iter().map(|v| v.abs()).fold(0.0f64, |acc, v| f64::max(acc, v));
        let l2_norm = (last_layer.iter().map(|v| v * v).sum::<f64>()).sqrt();
        eprintln!("Final layer hidden state: max={:.2}, L2_norm={:.2}", max_val, l2_norm);
    }

    eprintln!("\n=== FHE Hidden State Summary ===");
    if let Some(last_layer) = fhe_hidden_states.last() {
        let max_val = last_layer.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let l2_norm = (last_layer.iter().map(|v| v * v).sum::<f64>()).sqrt();
        eprintln!("Final layer hidden state: max={:.2}, L2_norm={:.2}", max_val, l2_norm);
    }

    Ok(())
}
