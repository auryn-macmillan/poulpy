/// Multi-layer plaintext shadow of the FHE forward path on SmolLM2-135M.
///
/// Runs `refreshed_plain_target_multilayer` through all 30 layers in cleartext,
/// matching the FHE quantization behaviour at each refresh point.  This produces
/// the correct reference token for comparison against the FHE run.
///
/// Usage:
///   cargo run --release --example smollm2_plain_multilayer [token_id] [n_layers]
///
/// Defaults: token_id=29889 ("."), n_layers=30
use std::process;
use std::time::Instant;

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
    let token_id: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(29889);
    let n_layers: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30);

    let d_model = 576;
    let d_ffn = 1536;
    let n_heads = 9;
    let n_kv_heads = 3;

    eprintln!("=== SmolLM2-135M Multi-Layer Plaintext Shadow ===");
    eprintln!(
        "token_id={token_id}, d_model={d_model}, d_ffn={d_ffn}, n_heads={n_heads}, \
         n_kv_heads={n_kv_heads}, n_layers={n_layers}"
    );
    eprintln!();

    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_ffn),
        num_heads: Some(n_heads),
        num_kv_heads: Some(n_kv_heads),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 1,
        ..InferenceConfig::default()
    };

    eprintln!("[main] Loading model and generating keys...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s", load_start.elapsed().as_secs_f64());
    eprintln!();

    // --- Multi-layer plaintext (with wrapping — faithful FHE shadow) ---
    eprintln!("[main] Running multi-layer plaintext shadow ({n_layers} layers, WRAPPING)...");
    let start_wrap = Instant::now();
    let result_wrap = pipeline.plaintext_step_refreshed_multilayer(token_id);
    let elapsed_wrap = start_wrap.elapsed();

    eprintln!();
    eprintln!("=== WRAPPING RESULT ===");
    eprintln!(
        "Token: {} {:?} (time={:.2}s)",
        result_wrap.token_id,
        result_wrap.token_text,
        elapsed_wrap.as_secs_f64()
    );
    eprintln!("Top logits: {:?}", result_wrap.top_logits);
    eprintln!();

    // --- Multi-layer plaintext (NO wrapping — diagnostic mode) ---
    eprintln!("[main] Running multi-layer plaintext shadow ({n_layers} layers, NO WRAP)...");
    let start_nowrap = Instant::now();
    let result_nowrap = pipeline.plaintext_step_refreshed_multilayer_no_wrap(token_id);
    let elapsed_nowrap = start_nowrap.elapsed();

    eprintln!();
    eprintln!("=== NO-WRAP RESULT ===");
    eprintln!(
        "Token: {} {:?} (time={:.2}s)",
        result_nowrap.token_id,
        result_nowrap.token_text,
        elapsed_nowrap.as_secs_f64()
    );
    eprintln!("Top logits: {:?}", result_nowrap.top_logits);
    eprintln!();

    // Also run the 1-layer path for comparison
    eprintln!("[main] Running 1-layer plaintext baseline for comparison...");
    let start_1l = Instant::now();
    let result_1l = pipeline.plaintext_step_refreshed(token_id);
    let elapsed_1l = start_1l.elapsed();
    eprintln!(
        "1-layer: token={} {:?} (time={:.2}s)",
        result_1l.token_id,
        result_1l.token_text,
        elapsed_1l.as_secs_f64()
    );
    eprintln!();

    // Summary line for grepping
    println!(
        "PLAIN_SUMMARY: d_model={d_model} n_layers={n_layers} token_id={token_id} \
         wrap_token={} wrap_text={:?} nowrap_token={} nowrap_text={:?} \
         1l_token={} 1l_text={:?} \
         wrap_time={:.2}s nowrap_time={:.2}s 1l_time={:.2}s",
        result_wrap.token_id,
        result_wrap.token_text,
        result_nowrap.token_id,
        result_nowrap.token_text,
        result_1l.token_id,
        result_1l.token_text,
        elapsed_wrap.as_secs_f64(),
        elapsed_nowrap.as_secs_f64(),
        elapsed_1l.as_secs_f64(),
    );

    Ok(())
}
