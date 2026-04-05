/// Full 30-layer FHE inference on SmolLM2-135M at native dimensions.
///
/// This example runs a single-token forward pass through all 30 transformer
/// layers under real FHE encryption (not plaintext simulation) at the model's
/// native d_model=576. It compares the FHE output against the refreshed
/// plaintext baseline and reports token match, error metrics, and timing.
///
/// Expected runtime: ~3-4 hours at 80-bit security on 16 cores.
/// Expected memory: ~2 GB peak.
///
/// Usage:
///   cargo run --release --example smollm2_full_fhe_30layer [token_id]
///
/// If token_id is omitted, defaults to 29889 (the "." token).
use std::process;
use std::time::Instant;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{ModelDims, Precision, SecurityLevel};

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
    let token_id: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(29889); // default: "."

    let n_layers: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30);

    // SmolLM2-135M truncated dimensions for test: d_model=256, d_ffn=1024, d_head=128,
    // n_heads=8, n_kv_heads=4
    let d_model = 256;
    let d_ffn = 1024;
    let n_heads = 8;
    let n_kv_heads = 4;

    eprintln!("=== SmolLM2-135M Full FHE Run ===");
    eprintln!(
        "token_id={token_id}, d_model={d_model}, d_ffn={d_ffn}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, n_layers={n_layers}"
    );
    eprintln!("security=80-bit, precision=Fp16");
    eprintln!();

    let config = InferenceConfig {
        security: SecurityLevel::Bits128,
        precision: Precision::Fp16,
        num_layers: Some(n_layers),
        trunc_d_model: None,
        trunc_d_ffn: None,
        num_heads: Some(9),
        num_kv_heads: Some(3),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
        fhe_frequent_bootstrap: false,
        fhe_extra_refresh: false,
        fhe_silu_log_msg_mod: None,
        fhe_identity_log_msg_mod: None,
        ..InferenceConfig::default()
    };

    eprintln!("[main] Loading model and generating FHE keys...");
    let load_start = Instant::now();
    let model_spec = ModelSpec {
        dims: ModelDims {
            d_model: 576,
            d_head: 64,
            n_heads: 9,
            n_kv_heads: 3,
            d_ffn: 1536,
            n_layers: 30,
            n_experts: 1,
            n_active_experts: 1,
        },
        embed_name: "model.embed_tokens.weight".to_string(),
        lm_head_name: "model.embed_tokens.weight".to_string(),
        final_norm_name: "model.norm.weight".to_string(),
        rope_theta: 10000.0,
        max_seq_len: 2048,
        bos_token_id: 1,
        eos_token_id: 2,
    };
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, model_spec, config)?;
    eprintln!(
        "[main] Model loaded and keys generated in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );
    eprintln!();

    // --- Multi-layer plaintext baseline (faithful shadow of FHE path) ---
    eprintln!("[main] Running multi-layer plaintext baseline ({n_layers} layers)...");
    let ml_plain_start = Instant::now();
    let plaintext_ml = pipeline.plaintext_step(token_id);
    let ml_plain_time = ml_plain_start.elapsed();
    eprintln!(
        "[main] Multi-layer plaintext done in {:.2}s: token={}",
        ml_plain_time.as_secs_f64(),
        plaintext_ml.token_id,
    );
    eprintln!("[main] Multi-layer plaintext top logits: {:?}", plaintext_ml.top_logits);
    eprintln!();

    // --- Legacy 1-layer plaintext (for reference / comparison) ---
    eprintln!("[main] Running 1-layer plaintext baseline (for reference)...");
    let plain_start = Instant::now();
    let plaintext_1l = pipeline.plaintext_step(token_id);
    let plain_time = plain_start.elapsed();
    eprintln!(
        "[main] 1-layer plaintext done in {:.2}s: token={}",
        plain_time.as_secs_f64(),
        plaintext_1l.token_id,
    );
    eprintln!();

    // --- Real FHE ---
    eprintln!("[main] Starting FHE forward pass (this will take a while)...");
    let fhe_start = Instant::now();
    let fhe_result = pipeline.step_refreshed(token_id)?;
    let fhe_total = fhe_start.elapsed();

    eprintln!();
    eprintln!("=== RESULTS ===");
    eprintln!();
    eprintln!(
        "Multi-layer plaintext ({n_layers}L): token={} (time={:.2}s)",
        plaintext_ml.token_id,
        ml_plain_time.as_secs_f64()
    );
    eprintln!("Multi-layer plaintext top logits: {:?}", plaintext_ml.top_logits);
    eprintln!();
    eprintln!(
        "1-layer plaintext (ref):           token={} (time={:.2}s)",
        plaintext_1l.token_id,
        plain_time.as_secs_f64()
    );
    eprintln!();
    eprintln!(
        "FHE ({n_layers}L):                 token={} {:?} (fhe_time={:.1}s, total={:.1}s)",
        fhe_result.token_id,
        fhe_result.token_text,
        fhe_result.fhe_time.as_secs_f64(),
        fhe_total.as_secs_f64()
    );
    eprintln!("FHE top logits: {:?}", fhe_result.top_logits);
    eprintln!();

    // Error metrics: FHE vs multi-layer plaintext (the proper comparison)
    let ml_hidden: Vec<f64> = plaintext_ml.hidden_state.iter().map(|&v| v as f64).collect();
    let fhe_hidden: Vec<f64> = fhe_result.hidden_state.iter().map(|&v| v as f64).collect();
    let (linf, l2, mae) = fhe_llm::plaintext_forward::error_metrics(&fhe_hidden, &ml_hidden);
    eprintln!("Hidden state error (FHE vs multi-layer plaintext):");
    eprintln!("  L-inf={:.3}  L2={:.3}  MAE={:.3}", linf, l2, mae);
    eprintln!();

    let token_match = plaintext_ml.token_id as u32 == fhe_result.token_id;
    eprintln!(
        "TOKEN MATCH (FHE vs {n_layers}L plaintext): {}",
        if plaintext_ml.token_id as u32 == fhe_result.token_id {
            "YES"
        } else {
            "NO"
        }
    );
    eprintln!();

    // Print summary for easy grepping
    println!(
        "SUMMARY: d_model={d_model} n_layers={n_layers} security=80bit token_id={token_id} \
             plain_ml_token={} plain_1l_token={} fhe_token={} match_ml={token_match} \
             fhe_time={:.1}s linf={linf:.3} l2={l2:.3} mae={mae:.3}",
        plaintext_ml.token_id as u32,
        plaintext_1l.token_id as u32,
        fhe_result.token_id,
        fhe_result.fhe_time.as_secs_f64()
    );

    Ok(())
}
