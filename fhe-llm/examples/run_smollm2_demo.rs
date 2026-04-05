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
        fhe_silu_log_msg_mod: Some(13),     // 8192‑level LUT for SiLU
        fhe_identity_log_msg_mod: Some(13), // 8192‑level LUT for identity refresh
        max_new_tokens: 1,                  // we only care about the first new token
        ..InferenceConfig::default()
    };

    // Load the pipeline
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, model_spec, config.clone())?;
    let load_elapsed = load_start.elapsed();
    println!("[demo] Pipeline loaded in {:.2}s", load_elapsed.as_secs_f64());

    // Prompt (English)
    let prompt = "The quick brown fox jumps over the lazy dog. Answer in one sentence.";
    println!("[demo] Prompt: {}", prompt);

    // Tokenize prompt
    let prompt_tokens = pipeline.tokenize(prompt)?;

    // FHE inference for one token (using the refreshed path)
    let fhe_step = pipeline.step_refreshed(*prompt_tokens.last().unwrap())?;
    let fhe_token_id = fhe_step.token_id;
    let fhe_token_text = fhe_step.token_text.clone();

    // Exact (plaintext) inference for the same starting token
    let exact_step = pipeline.plaintext_step(*prompt_tokens.last().unwrap());
    let exact_token_id = exact_step.token_id;
    let exact_token_text = exact_step.token_text.clone();

    // L‑inf error of the hidden state (refreshed) vs. exact hidden state
    let hidden_linf = pipeline.compare_fhe_vs_plaintext_refreshed(fhe_step.hidden_state, *prompt_tokens.last().unwrap());

    // Output
    println!("--- FHE step ---");
    println!("  Token ID: {}, Text: {}", fhe_token_id, fhe_token_text);
    println!("  Top‑5 logits: {:?}", fhe_step.top_logits);
    println!("  FHE forward time: {:.2}s", fhe_step.fhe_time.as_secs_f64());

    println!("--- Exact step ---");
    println!("  Token ID: {}, Text: {}", exact_token_id, exact_token_text);
    println!("  Top‑5 logits: {:?}", exact_step.top_logits);
    println!("  Plaintext time: {:.2}s", exact_step.total_time.as_secs_f64());

    println!("--- Comparison ---");
    println!("  L‑inf error of hidden states: {:.2}", hidden_linf);
    println!("  FHE token matches exact token? {}", fhe_token_id == exact_token_id);

    // Full text (prompt + generated)
    let full_text = format!("{} {}", prompt, fhe_token_text);
    println!("Full text (prompt + generated): {}", full_text);

    Ok(())
}
