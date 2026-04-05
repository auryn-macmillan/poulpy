use std::process;
use poulpy_FHE-LLM::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use poulpy_FHE-LLM::params::{ModelDims, Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str = "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Int8,
        num_layers: Some(1),
        trunc_d_model: Some(128),
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: poulpy_FHE-LLM::attention::SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        fhe_frequent_bootstrap: false,
        fhe_extra_refresh: false,
        fhe_silu_log_msg_mod: None,
        fhe_identity_log_msg_mod: None,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
        ..InferenceConfig::default()
    };

    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), cfg)?;
    let prompt = "2+2=";
    let prompt_tokens = pipeline.tokenize(prompt)?;
    let result = pipeline.plaintext_step(prompt_tokens[0]);
    println!("Exact token: {}", result.token_id);
    println!("Expected token: 33");
    println!("Match: {}", result.token_id == 33);
    Ok(())
}
