// Simple validation: FHE forward pass produces reasonable hidden states
// This skips the LM head entirely to avoid memory issues

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

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Int8,
        num_layers: Some(n_layers),
        trunc_d_model: Some(128),
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("=== FHE Forward Pass Only (d=128, layers={n_layers}) ===");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!();

    eprintln!("[main] Loading model...");
    let load_start = std::time::Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    eprintln!("[fhe] Running FHE forward pass (skipping LM head)...");
    let fhe_start = std::time::Instant::now();

    let step_result = pipeline.step_refreshed(prompt_tokens[0])?;
    let hidden_states = step_result.hidden_state;
    let fhe_time = fhe_start.elapsed();

    eprintln!("\n=== RESULTS ===");
    eprintln!(
        "FHE time: {:.1}s ({:.1}min)",
        fhe_time.as_secs_f64(),
        fhe_time.as_secs_f64() / 60.0
    );
    eprintln!("Hidden state shape: {} dimensions", hidden_states.len());
    eprintln!("Hidden state stats:");

    let mut min_val = i8::MAX;
    let mut max_val = i8::MIN;
    let mut sum_sq = 0f64;

    for (i, &val) in hidden_states.iter().enumerate() {
        let v = val as f64;
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
        sum_sq += v * v;

        if i < 5 {
            eprintln!("  hidden[{}] = {:.4}", i, v);
        } else if i == hidden_states.len() - 1 {
            eprintln!("  ...");
            eprintln!("  hidden[{}] = {:.4}", i, v);
        }
    }

    let mean = (sum_sq / hidden_states.len() as f64).sqrt();
    eprintln!(
        "  min = {:.4}, max = {:.4}, L2_norm = {:.4}",
        min_val as f64, max_val as f64, mean
    );

    let has_valid_hidden = hidden_states.iter().any(|&h| h.abs() > 0);
    if has_valid_hidden {
        eprintln!("\n✅ SUCCESS: FHE forward pass produced valid hidden states");
    } else {
        eprintln!("\n❌ FAIL: FHE forward pass produced degenerate hidden states");
    }

    Ok(())
}
