/// Full SmolLM2 validation test at native dimensions (d=576, 24 layers).
///
/// This test validates the 100-bit security solution at the actual model scale.
/// Expected runtime: ~2-3 hours without optimization (documented in PERFORMANCE_ANALYSIS.md).
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
    let prompt = args.get(0).map(|s| s.as_str()).unwrap_or("2+2=");
    let n_layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let precision_str = args.get(2).map(|s| s.as_str()).unwrap_or("int8");
    let precision = match precision_str.to_lowercase().as_str() {
        "fp16" => Precision::Fp16,
        _ => Precision::Int8,
    };

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: precision,
        num_layers: Some(n_layers),
        trunc_d_model: Some(64),   // Full d=576
        trunc_d_ffn: None,     // Full d=1536
        num_heads: Some(1),    // SmolLM2-135M uses 9 attention heads
        num_kv_heads: Some(1), // GQA with 3 KV heads
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        fhe_silu_log_msg_mod: Some(7),
        fhe_identity_log_msg_mod: Some(7),
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("=== Full SmolLM2 Validation (d=576, layers={n_layers}) ===");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!();

    eprintln!("[main] Loading full SmolLM2 model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    eprintln!("[fhe] Running full SmolLM2 FHE inference...");
    eprintln!("[fhe] This may take 2-3 hours without optimization");
    let fhe_start = Instant::now();
    let fhe_result = pipeline.step(prompt_tokens[0])?;
    let fhe_time = fhe_start.elapsed().as_secs_f64();

    eprintln!("\n=== Results ===");
    eprintln!("FHE time: {:.1}s ({:.1}min)", fhe_time, fhe_time / 60.0);
    eprintln!(
        "FHE token: {} (ID: {})",
        pipeline.decode_token(fhe_result.token_id).unwrap_or("??".to_string()),
        fhe_result.token_id
    );
    eprintln!("FHE logits top-5: {:?}", &fhe_result.top_logits[..5]);

    Ok(())
}
