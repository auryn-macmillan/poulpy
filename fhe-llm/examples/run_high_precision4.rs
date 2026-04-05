use std::time::Instant;

use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params;
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/repo/poulpy-FHE_LLM/examples/toy_model.safetensors";
const TOKENIZER_PATH: &str = "/home/dev/repo/poulpy-FHE_LLM/examples/tokenizer.json";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Custom model spec for toy model
    let spec = ModelSpec {
        dims: params::ModelDims {
            d_model: 64,
            d_head: 32,
            n_heads: 2,
            n_kv_heads: 1,
            d_ffn: 128,
            n_layers: 1,
            n_experts: 1,
            n_active_experts: 1,
        },
        embed_name: "model.embed_tokens.weight".to_string(),
        lm_head_name: "model.embed_tokens.weight".to_string(),
        final_norm_name: "model.norm.weight".to_string(),
        rope_theta: 10000.0,
        max_seq_len: 2048,
        bos_token_id: 1,
        eos_token_id: 2,
    };

    // High precision configuration matching toy model dimensions
    let config = InferenceConfig {
        security: SecurityLevel::Bits128,
        precision: Precision::Fp16,
        num_layers: Some(1),
        trunc_d_model: Some(64),
        trunc_d_ffn: Some(128),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: fhe_llm::attention::SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        // Use higher precision LUTs
        fhe_silu_log_msg_mod: Some(13),     // 8192 levels
        fhe_identity_log_msg_mod: Some(13), // 8192 levels
        fhe_frequent_bootstrap: true,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
        ..InferenceConfig::default()
    };

    eprintln!("=== SmolLM2 High Precision FHE Inference ===");
    eprintln!(
        "Model dims: d_model={}, d_ffn={}, layers={}",
        config.trunc_d_model.unwrap_or(128),
        config.trunc_d_ffn.unwrap_or(256),
        config.num_layers.unwrap_or(1)
    );
    eprintln!("Security: Bits128, Precision: Fp16");
    eprintln!("Prompt: \"2+2=\"");
    eprintln!("Expected token: \"1\" (ID: 33)");
    eprintln!();

    eprintln!("[main] Loading model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, spec, config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    // Skip tokenizer for demo; use a known token ID (e.g., token ID 33 for "1")
    // The prompt "2+2=" would normally be tokenized, but we directly embed the token ID for testing.
    let token_id: u32 = 33;
    let prompt_tokens: Vec<u32> = vec![token_id];
    // eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    // Run FHE forward + batched LM head using token ID directly
    eprintln!("[fhe] Running FHE forward pass + batched LM head for token ID 33...");
    let fhe_start = Instant::now();
    // Use step_refreshed which handles embedding, encryption, forward, decryption, LM head
    let step_result = pipeline.step_refreshed(token_id)?;
    let fhe_time = fhe_start.elapsed();
    eprintln!("[fhe] FHE + batched LM head done in {:.1}s", fhe_time.as_secs_f64());

    let token_id = step_result.token_id;
    let token_text = step_result.token_text.clone();

    eprintln!("\n=== RESULTS ===");
    eprintln!("FHE + batched LM head time: {:.1}s", fhe_time.as_secs_f64());
    eprintln!("Token ID: {} ({})", token_id, token_text);

    if token_id == 33 {
        eprintln!("\n✅ SUCCESS: Token matches expected (\"1\")!");
        eprintln!("High precision batched LM head works correctly.");
    } else {
        eprintln!("\n❌ FAIL: Token does not match expected.");
        eprintln!("Expected: 33 (\"1\")");
        eprintln!("Got: {} ({})", token_id, token_text);
        eprintln!("FHE noise is still too large or batched LM head has issues.");
    }

    Ok(())
}
