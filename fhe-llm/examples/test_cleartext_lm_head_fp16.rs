use std::process;
use std::time::Instant;

use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

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
    let prompt = "2+2=";
    let n_layers: usize = 1;

    // High precision configuration
    let config = InferenceConfig {
        security: SecurityLevel::Bits128,
        precision: Precision::Fp16,
        num_layers: Some(n_layers),
        trunc_d_model: Some(128),
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: fhe_llm::attention::SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        // Use higher precision LUTs
        fhe_silu_log_msg_mod: Some(13),     // 8192 levels
        fhe_identity_log_msg_mod: Some(13), // 8192 levels
        fhe_frequent_bootstrap: true,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
        ..InferenceConfig::default()
    };

    eprintln!("=== SmolLM2 FHE Noise Validation (d=128, layers={n_layers}, HIGH PRECISION) ===");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!("Expected token: \"1\" (ID: 33)");
    eprintln!();

    eprintln!("[main] Loading model...");
    let load_start = std::time::Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    // Run FHE forward + batched LM head
    eprintln!("[fhe] Running FHE forward pass + batched LM head...");
    let fhe_start = Instant::now();
    let result = pipeline.generate(prompt, 1)?;
    let fhe_time = fhe_start.elapsed();
    eprintln!("[fhe] FHE + batched LM head done in {:.1}s", fhe_time.as_secs_f64());

    let token_id = result.generated_tokens[0];
    let token_text = result.generated_text.clone();

    eprintln!("\n=== RESULTS ===");
    eprintln!("FHE + batched LM head time: {:.1}s", fhe_time.as_secs_f64());
    eprintln!("Token ID: {} ({})", token_id, token_text);

    if token_id == 33 {
        eprintln!("\n✅ SUCCESS: Token matches expected (\"1\")!");
        eprintln!("Batched LM head works correctly with high precision.");
    } else {
        eprintln!("\n❌ FAIL: Token does not match expected.");
        eprintln!("Expected: 33 (\"1\")");
        eprintln!("Got: {} ({})", token_id, token_text);
        eprintln!("FHE noise is still too large or batched LM head has issues.");
    }

    Ok(())
}
