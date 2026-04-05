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
        security: SecurityLevel::Bits128,
        precision: Precision::Fp32,
        num_layers: Some(1),
        trunc_d_model: Some(64),
        trunc_d_ffn: Some(256),
        num_heads: Some(1),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        fhe_frequent_bootstrap: true,
        fhe_silu_log_msg_mod: Some(14),
        fhe_identity_log_msg_mod: Some(14),
        ..InferenceConfig::default()
    };

    let load_start = Instant::now();
    println!("[compare] Loading pipeline with FP16 precision and frequent bootstraps...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::tinyllama_1_1b(), config)?;
    let load_elapsed = load_start.elapsed();
    println!("[compare] Pipeline loaded in {:.2}s", load_elapsed.as_secs_f64());

    // Tokenize a simple prompt "2+2="
    let prompt = "2+2=";
    let prompt_tokens = pipeline.tokenize(prompt)?;
    println!("[compare] Prompt '{}' tokenized to {:?}", prompt, prompt_tokens);

    // Run inference on the last token (single-step)
    let last_token = *prompt_tokens.last().unwrap();
    let fhe_start = Instant::now();
    let result = pipeline.step_refreshed(last_token)?;
    let fhe_elapsed = fhe_start.elapsed();

    // Run exact step for comparison
    let exact_step = pipeline.plaintext_step(last_token);
    let exact_elapsed = fhe_start.elapsed(); // Roughly same time

    println!("[compare] FHE step completed in {:.2}s", fhe_elapsed.as_secs_f64());
    println!("[compare] Exact step completed in {:.2?}", exact_elapsed);
    println!("[compare] Predicted token ID (FHE): {}", result.token_id);
    println!("[compare] Predicted token text (FHE): {}", result.token_text);
    println!("[compare] Top-5 logits (FHE): {:?}", result.top_logits);
    println!(
        "[compare] Hidden state range (FHE): [{:?}, {:?}]",
        result.hidden_state.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        result.hidden_state.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    );

    println!("[compare] Exact token ID (exact): {}", exact_step.token_id);
    println!("[compare] Exact top-5 logits (exact): {:?}", exact_step.top_logits);
    // Note: PlaintextStepResult does not have token_text or hidden_state fields, but we have token_id and top_logits

    // Basic sanity check: token should not be zero and logits should be non-degenerate.
    if result.token_id == 0 {
        println!("[WARN] Predicted token ID is 0 (unlikely)");
    } else {
        println!("[OK] Token prediction non-zero");
    }

    let non_zero_logits: usize = result.top_logits.iter().filter(|&&(_, logit)| logit != 0).count();
    if non_zero_logits == 0 {
        println!("[WARN] All logits are zero - possible noise collapse");
    } else {
        println!("[OK] Non-zero logits present ({} entries)", non_zero_logits);
    }

    Ok(())
}
