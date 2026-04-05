use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str = "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() {
    let config = InferenceConfig {
        security: SecurityLevel::Bits100,  // Changed from Bits80 to Bits100
        precision: Precision::Int8,
        num_layers: Some(1),
        trunc_d_model: Some(128),
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)
        .expect("Failed to load model");

    let prompt_tokens = pipeline.tokenize("2+2=").expect("Failed to tokenize");
    let fhe_result = pipeline.step(prompt_tokens[0]).expect("Failed to run FHE inference");
    
    println!("FHE token: {} (ID: {})", pipeline.decode_token(fhe_result.token_id).unwrap_or_else(|_| "??".to_string()), fhe_result.token_id);
    println!("FHE logits: {:?}", fhe_result.top_logits);
}
