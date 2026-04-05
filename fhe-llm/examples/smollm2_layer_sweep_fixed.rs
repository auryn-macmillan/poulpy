/// Layer-by-layer FHE vs plaintext shadow comparison with actual FHE encryption.
///
/// Runs the actual FHE forward pass with bootstrapping at each layer boundary,
/// decrypting hidden states for comparison with plaintext references.
///
/// Usage:
///   cargo run --release --example smollm2_layer_sweep_fixed [prompt] [d_model] [n_layers]
///
/// Defaults: prompt="2+2=", d_model=64, n_layers=4 (for quick testing)
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
    let mut args_iter = args.into_iter();
    let user_prompt = args_iter.next().unwrap_or_else(|| "2+2=".to_string());
    let d_model: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_layers: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let d_ffn = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(d_model * 2);

    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    eprintln!("=== SmolLM2 Layer-by-Layer FHE Audit ===");
    eprintln!("d_model={d_model}, d_ffn={d_ffn}, n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}");
    eprintln!("prompt: {:?}", user_prompt);
    eprintln!();

    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_ffn),
        num_heads: Some(n_heads),
        num_kv_heads: Some(n_kv_heads),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("[main] Loading model and generating FHE keys...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    let prompt_tokens = pipeline.tokenize(&user_prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    eprintln!("[fhe] Running FHE inference on first token...");
    let fhe_start = Instant::now();
    let fhe_result = pipeline.step(prompt_tokens[0])?;
    let fhe_time = fhe_start.elapsed().as_secs_f64();

    eprintln!("[fhe] FHE step completed in {:.1}s", fhe_time);
    eprintln!("[fhe] Hidden state shape: {}", fhe_result.hidden_state.len());
    if !fhe_result.hidden_state.is_empty() {
        let first_10: Vec<i64> = fhe_result.hidden_state.iter().take(10).map(|&x| x as i64).collect();
        eprintln!("[fhe] First 10 hidden values: {:?}", first_10);
    }

    eprintln!("\n[plaintext] Running plaintext comparison...");
    let plaintext_start = Instant::now();
    let plaintext_result = pipeline.plaintext_step(prompt_tokens[0]);
    let plaintext_time = plaintext_start.elapsed().as_secs_f64();

    eprintln!("[plaintext] Plaintext step completed in {:.1}s", plaintext_time);
    eprintln!("[plaintext] Hidden state shape: {}", plaintext_result.hidden_state.len());
    if !plaintext_result.hidden_state.is_empty() {
        let first_10: Vec<f64> = plaintext_result.hidden_state.iter().take(10).cloned().collect();
        eprintln!("[plaintext] First 10 hidden values: {:?}", first_10);
    }

    eprintln!("\n[comparison] Comparing FHE vs plaintext hidden states...");
    let comparison = pipeline.compare_fhe_vs_plaintext(&fhe_result.hidden_state, prompt_tokens[0]);
    eprintln!(
        "FHE vs exact (total error): Linf={:.4}, L2={:.4}, MAE={:.4}",
        comparison.fhe_vs_exact.0, comparison.fhe_vs_exact.1, comparison.fhe_vs_exact.2
    );
    eprintln!(
        "Polynomial vs exact (poly approximation): Linf={:.4}, L2={:.4}, MAE={:.4}",
        comparison.poly_vs_exact.0, comparison.poly_vs_exact.1, comparison.poly_vs_exact.2
    );
    eprintln!(
        "FHE vs poly (FHE noise only): Linf={:.4}, L2={:.4}, MAE={:.4}",
        comparison.fhe_vs_poly.0, comparison.fhe_vs_poly.1, comparison.fhe_vs_poly.2
    );

    let fhe_token_str = pipeline.decode_token(fhe_result.token_id)?;
    eprintln!(
        "\n[fhe] FHE predicted token: {} (ID: {}), Top-3 logits: {:?}",
        fhe_token_str,
        fhe_result.token_id,
        &fhe_result.top_logits[..fhe_result.top_logits.len().min(3)]
    );

    let plaintext_token_id = plaintext_result.token_id as u32;
    let plaintext_token_str = pipeline.decode_token(plaintext_token_id)?;
    eprintln!(
        "\n[plaintext] Plaintext predicted token: {} (ID: {}), Top-3 logits: {:?}",
        plaintext_token_str,
        plaintext_result.token_id,
        &plaintext_result.top_logits[..plaintext_result.top_logits.len().min(3)]
    );

    eprintln!("\n=== Result ===");
    if fhe_result.token_id == plaintext_token_id {
        eprintln!("✅ Tokens match! ({} = {})", fhe_token_str, plaintext_token_str);
    } else {
        eprintln!("❌ Tokens differ!");
        eprintln!(
            "   FHE:      {} ({}), logits: {:?}",
            fhe_token_str,
            fhe_result.token_id,
            &fhe_result.top_logits[..fhe_result.top_logits.len().min(3)]
        );
        eprintln!(
            "   Plaintext: {} ({}), logits: {:?}",
            plaintext_token_str,
            plaintext_token_id,
            &plaintext_result.top_logits[..plaintext_result.top_logits.len().min(3)]
        );
    }

    Ok(())
}
