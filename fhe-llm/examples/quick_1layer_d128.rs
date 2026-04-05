// Quick 1-layer d=128 validation test with sequential LM head

use std::process;
use std::time::Instant;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let prompt = args.get(0).map(|s| s.as_str()).unwrap_or("2+2=");
    let n_layers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Int8,
        num_layers: Some(n_layers),
        trunc_d_model: Some(128), // KEY: Use d=128 for fast validation
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        fhe_silu_log_msg_mod: Some(11),
        fhe_identity_log_msg_mod: Some(11),
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("=== SmolLM2 Validation (d=128, layers={}, HOMOMORPHIC LM HEAD) ===", n_layers);
    eprintln!("Prompt: \"{}\"", prompt);
    eprintln!("Expected token: \"1\" (ID: 33)");
    eprintln!();

    eprintln!("[main] Loading model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    eprintln!("[fhe] Running FHE inference with HOMOMORPHIC LM HEAD...");

    let fhe_start = Instant::now();
    let fhe_result = pipeline.step(prompt_tokens[0])?;
    let fhe_time = fhe_start.elapsed();

    eprintln!("\n=== RESULTS ===");
    eprintln!("FHE time: {:.1}s", fhe_time.as_secs_f64());
    eprintln!(
        "FHE token: {} (ID: {})",
        pipeline.decode_token(fhe_result.token_id).unwrap_or("??".to_string()),
        fhe_result.token_id
    );
    eprintln!("FHE logits top-5: {:?}", &fhe_result.top_logits[..5]);

    // Check correctness
    if fhe_result.token_id == 33 {
        eprintln!("\n✅ SUCCESS: Token matches expected (\"1\")!");
    } else {
        eprintln!("\n❌ FAIL: Token does not match expected.");
        eprintln!("Expected: 33 (\"1\")");
        eprintln!(
            "Got: {} ({})",
            fhe_result.token_id,
            pipeline.decode_token(fhe_result.token_id).unwrap_or("??".to_string())
        );
    }

    Ok(())
}
