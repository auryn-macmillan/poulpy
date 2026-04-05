use fhe_llm::attention::SoftmaxStrategy;
/// Example to test encrypted LM head integration.
///
/// This example loads a small truncated model and runs a single token
/// through the FHE pipeline with encrypted LM head.
///
/// Usage:
///   cargo +nightly run --release --example encrypted_lm_head_test "Hello" 64 1
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let prompt = if args.len() > 1 { &args[1] } else { "Hello" };
    let trunc_d = if args.len() > 2 { args[2].parse::<u32>()? } else { 64 };
    let num_layers = if args.len() > 3 { args[3].parse::<u32>()? } else { 1 };

    println!("[test] Prompt: {}", prompt);
    println!("[test] Trunc d_model: {}, Layers: {}", trunc_d, num_layers);

    // Use TinyLlama model for testing
    let model_path = "/home/dev/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";
    let tokenizer_path = "/home/dev/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";

    if !std::path::Path::new(model_path).exists() {
        eprintln!("[SKIP] Model not found at {}", model_path);
        eprintln!("[SKIP] Set up model first or use toy dimensions");
        return Ok(());
    }

    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(num_layers as usize),
        trunc_d_model: Some(trunc_d as usize),
        trunc_d_ffn: Some((trunc_d * 4) as usize),
        num_heads: Some(1),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        ..InferenceConfig::default()
    };

    eprintln!("[test] Loading pipeline...");
    let pipeline = InferencePipeline::load(model_path, tokenizer_path, ModelSpec::tinyllama_1_1b(), config)?;
    eprintln!("[test] Pipeline loaded successfully");

    // Tokenize the prompt
    let tokens = pipeline.tokenize(prompt)?;
    eprintln!("[test] Tokenized {} tokens: {:?}", tokens.len(), tokens);

    // Run inference on the last token
    let last_token = *tokens.last().unwrap();
    eprintln!("[test] Running FHE inference on token {}...", last_token);

    let result = pipeline.step_refreshed(last_token)?;

    eprintln!("[test] === Results ===");
    eprintln!("[test] Predicted token ID: {}", result.token_id);
    eprintln!("[test] Predicted text: {}", result.token_text);
    eprintln!("[test] Top-5 logits: {:?}", result.top_logits);
    eprintln!("[test] Hidden state shape: {} dims", result.hidden_state.len());
    eprintln!(
        "[test] Hidden state range: [{:?}, {:?}]",
        result.hidden_state.iter().min(),
        result.hidden_state.iter().max()
    );
    eprintln!("[test] FHE time: {:.2}s", result.fhe_time.as_secs_f64());
    eprintln!("[test] Total time: {:.2}s", result.total_time.as_secs_f64());

    // Validate that logits are reasonable (not all zeros or NaN)
    let max_logit = result.top_logits.iter().map(|&(_, l)| l).max().unwrap();
    let min_logit = result.top_logits.iter().map(|&(_, l)| l).min().unwrap();

    if max_logit == 0 && min_logit == 0 {
        eprintln!("[WARN] All logits are zero - this may indicate an issue");
    } else {
        eprintln!("[OK] Logits are non-degenerate: [{}, {}]", min_logit, max_logit);
    }

    Ok(())
}
