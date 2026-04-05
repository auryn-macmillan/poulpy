//! Single-layer test with frequent bootstrapping.
//!
//! Tests whether aggressive bootstrapping after each major operation
//! can reduce hidden state noise to L-inf < 0.015 for correct token prediction.

use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "2+2=";
    let model_path = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
    let tokenizer_path =
        "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

    // Test with aggressive bootstrapping after each operation and higher precision
    let mut config = InferenceConfig::default();
    config.security = SecurityLevel::Bits80;
    config.precision = Precision::Int8;
    config.num_layers = Some(1);
    config.trunc_d_model = Some(64);
    config.trunc_d_ffn = Some(128);
    config.num_heads = Some(1);
    config.num_kv_heads = Some(1);
    config.apply_final_norm = true;
    config.max_new_tokens = 1;
    config.fhe_frequent_bootstrap = true;
    config.fhe_identity_log_msg_mod = Some(11); // Use 2048 levels instead of 128
    config.fhe_silu_log_msg_mod = Some(11); // Use 2048 levels instead of 128

    let model_spec = ModelSpec::smollm2_135m_instruct();

    println!("Loading model with aggressive bootstrapping config...");
    let pipeline = InferencePipeline::load(model_path, tokenizer_path, model_spec, config)?;

    println!("Tokenizing prompt: {:?}", prompt);
    let token_ids = pipeline.tokenize(prompt)?;
    println!("Token IDs: {:?}", token_ids);

    println!("\nGenerating single token with frequent bootstrapping...");
    let last_token = token_ids.last().copied().unwrap();
    let output = pipeline.step(last_token)?;

    println!("\n=== Results ===");
    println!("Prompt: {:?}", prompt);
    println!("FHE Token: {} (ID: {})", output.token_text, output.token_id);
    println!("Time: {:.2}s", output.total_time.as_secs_f64());
    println!("Top-5 logits:");
    for (i, (tid, logit)) in output.top_logits.iter().enumerate() {
        let text = pipeline.decode_token(*tid)?;
        println!("  {}: {} (logit={}) text={:?}", i + 1, tid, logit, text);
    }

    // Compare with exact path
    println!("\n=== Comparison ===");
    let exact_result = pipeline.plaintext_step(last_token);
    println!("Exact Token: {} (ID: {})", output.token_text, exact_result.token_id as u32);
    println!("Match: {}", output.token_id == exact_result.token_id as u32);

    Ok(())
}
