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
        apply_final_norm: true,
        max_new_tokens: 0,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("=== FHE Hidden State Validation (d=128, layers={n_layers}) ===");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!();

    eprintln!("[main] Loading model...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", std::time::Instant::now().elapsed().as_secs_f64());

    eprintln!("[generate] Running FHE forward pass...");
    let start = std::time::Instant::now();

    let result = pipeline.generate(prompt, 1)?;

    let elapsed = start.elapsed();
    eprintln!("[generate] Complete in {:.1}s\n", elapsed.as_secs_f64());

    eprintln!("\n=== RESULTS ===");
    eprintln!("Full text: \"{}\"", result.full_text);
    eprintln!("Prompt tokens: {}", result.prompt_tokens.len());
    eprintln!("Generated tokens: {}", result.generated_tokens.len());

    eprintln!("\n✅ FHE forward pass completed successfully");
    eprintln!("   Hidden states encrypted through {} layer(s)", n_layers);
    eprintln!("   Next step would be LM head (49152 logits × 128 dims)");

    Ok(())
}
