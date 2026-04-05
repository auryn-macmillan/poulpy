use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use std::process;
use std::time::Instant;

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str = "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Int8,
        num_layers: Some(30),
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

    eprintln!("=== 30-Layer FHE Inference with 100-bit Security ===");
    eprintln!("[main] Loading model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let prompt_tokens = pipeline.tokenize("2+2=")?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    eprintln!("[fhe] Running 30-layer FHE inference...");
    let fhe_start = Instant::now();
    let fhe_result = pipeline.step(prompt_tokens[0])?;
    let fhe_time = fhe_start.elapsed().as_secs_f64();

    eprintln!("\n=== Results ===");
    eprintln!("FHE time: {:.1}s", fhe_time);
    eprintln!("FHE token: {} (ID: {})", pipeline.decode_token(fhe_result.token_id).unwrap_or_else(|_| "??".to_string()), fhe_result.token_id);
    eprintln!("FHE logits: {:?}", &fhe_result.top_logits[..5]);
    
    Ok(())
}
