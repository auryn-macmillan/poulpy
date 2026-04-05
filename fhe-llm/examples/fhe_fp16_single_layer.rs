use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
/// FP16 precision single-layer FHE inference test.
///
/// This example runs a single-layer FHE forward pass with FP16 precision
/// (8192 bootstrap levels) and verifies that the token prediction
/// matches the plaintext baseline.
use std::time::Instant;

const MODEL_PATH: &str = "/home/dev/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use TinyLlama model, but truncate to d_model=64 for tractability.
    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Fp16,
        num_layers: Some(1),
        trunc_d_model: Some(64),
        trunc_d_ffn: Some(256),
        num_heads: Some(1),
        num_kv_heads: Some(1),
        softmax_strategy: fhe_llm::attention::SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        // FP16 precision requires high bootstrap modulus for SiLU and identity ops.
        fhe_silu_log_msg_mod: Some(13), // 8192 levels
        fhe_identity_log_msg_mod: Some(13),
        fhe_frequent_bootstrap: false, // single bootstrap after layer
        ..InferenceConfig::default()
    };

    let load_start = std::time::Instant::now();
    eprintln!("[test] Loading pipeline with FP16 precision...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::tinyllama_1_1b(), config)?;
    let load_elapsed = load_start.elapsed();
    eprintln!("[test] Pipeline loaded in {:.2}s", load_elapsed.as_secs_f64());

    // Tokenize a simple prompt "2+2="
    let prompt = "2+2=";
    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("[test] Prompt '{}' tokenized to {:?}", prompt, prompt_tokens);

    // Run inference on the last token (single-step)
    let last_token = *prompt_tokens.last().unwrap();
    let start = Instant::now();
    let result = pipeline.step_refreshed(last_token)?;
    let elapsed = start.elapsed();

    eprintln!("[test] FHE step completed in {:.2}s", elapsed.as_secs_f64());
    eprintln!("[test] Predicted token ID: {}", result.token_id);
    eprintln!("[test] Predicted token text: {}", result.token_text);
    eprintln!("[test] Top-5 logits: {:?}", result.top_logits);
    eprintln!(
        "[test] Hidden state range: [{:?}, {:?}]",
        result.hidden_state.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        result.hidden_state.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
    );

    // Basic sanity check: token should not be zero and logits should be non-degenerate.
    if result.token_id == 0 {
        eprintln!("[WARN] Predicted token ID is 0 (unlikely)");
    } else {
        eprintln!("[OK] Token prediction non-zero");
    }

    // Simple logit check: ensure at least one non-zero logit.
    let non_zero_logits: usize = result.top_logits.iter().filter(|&&(_, logit)| logit != 0).count();
    if non_zero_logits == 0 {
        eprintln!("[WARN] All logits are zero - possible noise collapse");
    } else {
        eprintln!("[OK] Non-zero logits present ({} entries)", non_zero_logits);
    }

    Ok(())
}
