use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use std::time::Instant;

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let config = InferenceConfig {
        security: SecurityLevel::Bits128,
        precision: Precision::Fp16,
        num_layers: Some(1),
        trunc_d_model: Some(128),
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        fhe_frequent_bootstrap: true,
        fhe_silu_log_msg_mod: Some(13),
        fhe_identity_log_msg_mod: Some(13),
        max_new_tokens: 1,
        ..InferenceConfig::default()
    };

    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, model_spec, config)?;
    let load_elapsed = load_start.elapsed();
    println!("[test] Pipeline loaded in {:.2}s", load_elapsed.as_secs_f64());

    let prompt = "The quick brown fox jumps over the lazy dog. Answer in one sentence.";
    let start = Instant::now();
    let result = pipeline.generate(prompt, 1)?;
    let elapsed = start.elapsed();

    println!("[test] Generated token in {:.2}s", elapsed.as_secs_f64());
    println!("[test] Full text (prompt + generated): {}", result.full_text);
    println!("[test] Generated token ID: {}", result.generated_tokens[0]);
    println!("[test] Top‑5 logits: {:?}", result.steps[0].top_logits);
    let hidden_linf = result.steps[0]
        .hidden_state
        .iter()
        .map(|&v| v.abs() as f64)
        .fold(0.0, f64::max);
    println!("[test] Hidden state L‑inf error vs. exact reference: {}", hidden_linf);
    Ok(())
}
