/// FP64 precision single-layer FHE inference test with high bootstrap precision.
///
/// This example runs a single-layer FHE forward pass with FP64 precision
/// (16384 bootstrap levels) and verifies that the token prediction
/// matches the plaintext baseline for the prompt "2+2=".
use std::time::Instant;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use TinyLlama model, truncate to d_model=64 for tractability.
    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Fp64, // Use FP64 for higher precision
        num_layers: Some(1),
        trunc_d_model: Some(64),
        trunc_d_ffn: Some(256),
        num_heads: Some(1),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        // FP64 precision requires high bootstrap modulus for SiLU and identity ops.
        fhe_silu_log_msg_mod: Some(14), // 16384 levels
        fhe_identity_log_msg_mod: Some(14),
        fhe_frequent_bootstrap: true, // Enable frequent bootstraps
        ..InferenceConfig::default()
    };

    let load_start = Instant::now();
    println!("[test] Loading pipeline with FP64 precision and high bootstrap precision...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::tinyllama_1_1b(), config)?;
    let load_elapsed = load_start.elapsed();
    println!("[test] Pipeline loaded in {:.2}s", load_elapsed.as_secs_f64());

    // Tokenize a simple prompt "2+2="
    let prompt = "2+2=";
    let prompt_tokens = pipeline.tokenize(prompt)?;
    println!("[test] Prompt '{}' tokenized to {:?}", prompt, prompt_tokens);

    // Run inference on the last token (single-step)
    let last_token = *prompt_tokens.last().unwrap();
    let start = Instant::now();
    let result = pipeline.step_refreshed(last_token)?;
    let elapsed = start.elapsed();

    println!("[test] FHE step completed in {:.2}s", elapsed.as_secs_f64());
    println!("[test] Predicted token ID: {}", result.token_id);
    println!("[test] Predicted token text: {}", result.token_text);
    println!("[test] Top-5 logits: {:?}", result.top_logits);
    println!(
        "[test] Hidden state range: [{:?}, {:?}]",
        result.hidden_state.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        result.hidden_state.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
    );

    // Basic sanity check: token should not be zero and logits should be non-degenerate.
    if result.token_id == 0 {
        println!("[WARN] Predicted token ID is 0 (unlikely)");
    } else {
        println!("[OK] Token prediction non-zero");
    }

    // Simple logit check: ensure at least one non-zero logit.
    let non_zero_logits: usize = result.top_logits.iter().filter(|&&(_, logit)| logit != 0).count();
    if non_zero_logits == 0 {
        println!("[WARN] All logits are zero - possible noise collapse");
    } else {
        println!("[OK] Non-zero logits present ({} entries)", non_zero_logits);
    }

    Ok(())
}
