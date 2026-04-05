use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use std::time::Instant;

const MODEL_PATH: &str = "/home/dev/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // TinyLlama 1.1B spec (will be truncated by config)
    let model_spec = ModelSpec::tinyllama_1_1b();

    // High‑precision configuration (Bits128 / FP16, 8192‑level LUTs, frequent bootstrap)
    let config = InferenceConfig {
        security: SecurityLevel::Bits128,
        precision: Precision::Fp16,
        num_layers: Some(1),     // keep single layer for tractability
        trunc_d_model: Some(64), // truncate to 64 model dims
        trunc_d_ffn: Some(128),  // truncate FFN to 128 hidden units
        num_heads: Some(2),      // 2 heads after truncation (d_head=32)
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        fhe_frequent_bootstrap: false,
        // Use 1024‑level LUTs to reduce quantisation error
        fhe_silu_log_msg_mod: Some(10),
        fhe_identity_log_msg_mod: Some(10),
        max_new_tokens: 1, // we only care about the first new token
        ..InferenceConfig::default()
    };

    // Load the pipeline
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, model_spec, config.clone())?;
    let load_elapsed = load_start.elapsed();
    println!("[demo] Pipeline loaded in {:.2}s", load_elapsed.as_secs_f64());

    // Prompt (English arithmetic)
    let prompt = "What is 2+2? Answer in one token.";
    println!("[demo] Prompt: {}", prompt);

    // Tokenize prompt (required for step_refreshed)
    let prompt_tokens = pipeline.tokenize(prompt)?;

    // FHE inference for one token (using the refreshed path)
    let fhe_step = pipeline.step_refreshed(*prompt_tokens.last().unwrap())?;
    let fhe_token_id = fhe_step.token_id as u32;
    let fhe_token_text = pipeline
        .decode_token(fhe_token_id)
        .unwrap_or_else(|_| format!("<tok:{}>", fhe_token_id));

    // Exact (plaintext) inference for the same starting token
    let exact_step = pipeline.plaintext_step(*prompt_tokens.last().unwrap());
    let exact_token_id = exact_step.token_id as u32;
    let exact_token_text = pipeline
        .decode_token(exact_token_id)
        .unwrap_or_else(|_| format!("<tok:{}>", exact_token_id));

    // L‑inf error of the hidden state (refreshed) vs. exact hidden state
    let (linf, l2, mae) = pipeline.compare_fhe_vs_plaintext_refreshed(&fhe_step.hidden_state, *prompt_tokens.last().unwrap());

    // Output
    println!("--- FHE step ---");
    println!("  Token ID: {}, Text: {}", fhe_token_id, fhe_token_text);
    println!("  Top‑5 logits: {:?}", fhe_step.top_logits);
    println!("  FHE forward time: {:.2}s", fhe_step.fhe_time.as_secs_f64());

    println!("--- Exact step ---");
    println!("  Token ID: {}, Text: {}", exact_token_id, exact_token_text);
    // We don't have plaintext total time in PlaintextStepResult, so omit
    println!("  Top‑5 logits: {:?}", exact_step.top_logits);

    println!("--- Comparison ---");
    println!("  L‑inf error of hidden states: {:.2}", linf);
    println!("  FHE token matches exact token? {}", fhe_token_id == exact_token_id);
    println!("  L2 error: {:.3}", l2);
    println!("  MAE: {:.3}", mae);

    // Full text (prompt + generated)
    let full_text = format!("{} {}", prompt, fhe_token_text);
    println!("Full text (prompt + generated): {}", full_text);

    Ok(())
}
