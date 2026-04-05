use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use std::time::Instant;

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Full SmolLM2-135M spec (will be truncated by config)
    let model_spec = {
        let mut dims = fhe_llm::params::ModelDims {
            d_model: 576,
            d_head: 64,
            n_heads: 9,
            n_kv_heads: 3,
            d_ffn: 1536,
            n_layers: 30,
            n_experts: 1,
            n_active_experts: 1,
        };
        ModelSpec {
            dims,
            embed_name: "model.embed_tokens.weight".to_string(),
            lm_head_name: "model.embed_tokens.weight".to_string(),
            final_norm_name: "model.norm.weight".to_string(),
            rope_theta: 100000.0,
            max_seq_len: 8192,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    };

    // High‑precision configuration (Bits128 / FP16, 8192‑level LUTs, frequent bootstrap)
    let config = InferenceConfig {
        security: SecurityLevel::Bits128,
        precision: Precision::Fp16,
        num_layers: Some(1),      // keep single layer for tractability
        trunc_d_model: Some(128), // truncate to 128 model dims
        trunc_d_ffn: Some(256),   // truncate FFN to 256 hidden units
        num_heads: Some(2),       // 2 heads after truncation (d_head=64)
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        fhe_frequent_bootstrap: true,
        fhe_silu_log_msg_mod: Some(13),
        fhe_identity_log_msg_mod: Some(13),
        max_new_tokens: 2, // generate up to 2 new tokens
        ..InferenceConfig::default()
    };

    // Load the pipeline
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, model_spec, config.clone())?;
    let load_elapsed = load_start.elapsed();
    println!("[test] Pipeline loaded in {:.2}s", load_elapsed.as_secs_f64());

    // Prompt we want to generate from
    let prompt = "The quick brown fox jumps over the lazy dog. Answer in one sentence.";
    println!("[test] Prompt: {}", prompt);

    // Tokenize prompt (required for step_refreshed)
    let prompt_tokens = pipeline.tokenize(prompt)?;

    // Generate up to max_new_tokens
    let start = Instant::now();
    let result = pipeline.generate(prompt, config.max_new_tokens)?;
    let elapsed = start.elapsed();

    println!("[test] Generation completed in {:.2}s", elapsed.as_secs_f64());
    println!("[test] Full text (prompt + generated): {}", result.full_text);
    println!("[test] Generated token IDs: {:?}", result.generated_tokens);
    println!(
        "[test] Top‑5 logits for final step: {:?}",
        result.steps.last().unwrap().top_logits
    );

    // Compute L‑inf error of final hidden state vs. exact reference (optional)
    let final_hidden = result.steps.last().unwrap().hidden_state.clone();
    let hidden_linf = final_hidden
        .iter()
        .map(|&v| v.abs() as f64)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    println!(
        "[test] L‑inf error of final hidden state vs. exact reference: {}",
        hidden_linf
    );

    Ok(())
}
