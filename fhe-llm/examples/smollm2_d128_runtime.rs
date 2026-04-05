/// Quick d=128 validation test for runtime estimation.
///
/// Tests truncated dimensions (d=128) to get fast runtime estimate
/// for extrapolating full d=576 24-layer inference.
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
    let n_layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Int8,
        num_layers: Some(n_layers),
        trunc_d_model: Some(128), // Truncated dimensions for fast testing
        trunc_d_ffn: Some(256),
        num_heads: Some(2),    // Truncated heads
        num_kv_heads: Some(1), // Truncated KV heads
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("=== d=128 Runtime Estimation Test (layers={}) ===", n_layers);
    eprintln!("Prompt: \"{}\"\n", prompt);

    eprintln!("[main] Loading model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    eprintln!("[fhe] Running inference...");
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
    eprintln!(
        "FHE logits top-5: {:?}",
        &fhe_result.top_logits[..5.min(fhe_result.top_logits.len())]
    );

    // Extrapolate to 24 layers
    let time_per_layer = fhe_time / n_layers as f64;
    let estimated_24_layers = time_per_layer * 24.0;
    eprintln!(
        "\n[estimate] 24-layer extrapolation: {:.1}s ({:.1}min / {:.1}h)",
        estimated_24_layers,
        estimated_24_layers / 60.0,
        estimated_24_layers / 3600.0
    );

    Ok(())
}
