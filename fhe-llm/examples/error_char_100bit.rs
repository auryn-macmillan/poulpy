/// Error characterization test for 100-bit security FHE inference.
///
/// Compares FHE output vs plaintext shadow to measure error growth.
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let d_model: usize = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(128);
    let n_layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);

    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    eprintln!("=== 100-bit Security Error Characterization ===");
    eprintln!("d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}");
    eprintln!();

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Int8,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_model * 2),
        num_heads: Some(n_heads),
        num_kv_heads: Some(n_kv_heads),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("[main] Loading model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    // Get plaintext baseline
    eprintln!("[baseline] Running plaintext inference...");
    let baseline_start = Instant::now();
    let plaintext_result = pipeline.plaintext_step(34); // Token "2"
    let baseline_time = baseline_start.elapsed().as_secs_f64();
    eprintln!(
        "[baseline] Plaintext token: {} (ID: {})",
        pipeline
            .decode_token(plaintext_result.token_id as u32)
            .unwrap_or("??".to_string()),
        plaintext_result.token_id
    );
    eprintln!("[baseline] Done in {:.1}s\n", baseline_time);

    // Run FHE inference
    eprintln!("[fhe] Running FHE inference...");
    let fhe_start = Instant::now();
    let fhe_result = pipeline.step_refreshed(34)?; // Token "2"
    let fhe_time = fhe_start.elapsed().as_secs_f64();
    eprintln!(
        "[fhe] FHE token: {} (ID: {})",
        pipeline.decode_token(fhe_result.token_id).unwrap_or("??".to_string()),
        fhe_result.token_id
    );
    eprintln!("[fhe] Done in {:.1}s\n", fhe_time);

    // Compare outputs
    let token_match = fhe_result.token_id == (plaintext_result.token_id as u32);
    eprintln!("=== Results ===");
    eprintln!("Token match: {}", if token_match { "✅ YES" } else { "❌ NO" });
    eprintln!(
        "Plaintext token: {} (ID: {})",
        pipeline
            .decode_token(plaintext_result.token_id as u32)
            .unwrap_or("??".to_string()),
        plaintext_result.token_id
    );
    eprintln!(
        "FHE token: {} (ID: {})",
        pipeline.decode_token(fhe_result.token_id).unwrap_or("??".to_string()),
        fhe_result.token_id
    );
    eprintln!("\nError metrics:");

    // Get hidden state comparison
    let (total_err, poly_approx_err, noise_err) = pipeline.compare_fhe_vs_plaintext_refreshed(&fhe_result.hidden_state, 34);
    eprintln!("  Total L-inf error: {:.4}", total_err);
    eprintln!("  Polynomial approximation error: {:.4}", poly_approx_err);
    eprintln!("  FHE noise error: {:.4}", noise_err);

    Ok(())
}
