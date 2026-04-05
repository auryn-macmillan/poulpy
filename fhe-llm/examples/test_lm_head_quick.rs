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
    let top_k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

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

    eprintln!("=== SmolLM2 Quick Test (d=128, layers={n_layers}, LM head top-{top_k}) ===");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!();

    eprintln!("[main] Loading model...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", std::time::Instant::now().elapsed().as_secs_f64());

    eprintln!("[generate] Running FHE inference...");
    let start = std::time::Instant::now();
    let result = pipeline.generate(prompt, 1)?;
    let elapsed = start.elapsed();
    eprintln!("[generate] Complete in {:.1}s\n", elapsed.as_secs_f64());

    eprintln!("\n=== RESULTS ===");
    eprintln!("Generated text: {}", result.generated_text);
    eprintln!("Generated token ID: {}", result.generated_tokens[0]);

    eprintln!("\nTop-{} logits (from cleartext decode):", top_k);
    if result.generated_tokens[0] == 33 {
        eprintln!("\n✅ Token 33 (\"1\") predicted - correct for \"2+2=\"");
    } else {
        eprintln!("\n⚠️  Token {} predicted (expected 33)", result.generated_tokens[0]);
    }

    Ok(())
}
