/// End-to-end FHE inference test on truncated SmolLM2 with encrypted LM head.
///
/// Tests whether the full inference pipeline produces correct tokens on a
/// real model (truncated to smaller dimensions for speed).
///
/// Usage:
///   cargo run --release --example e2e_fhe_smollm2 [prompt] [d_model] [n_layers]
///
/// Defaults: prompt="2+2=", d_model=64, n_layers=4
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

    eprintln!("=== End-to-End FHE Inference on SmolLM2 ===");
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

    // Tokenize prompt
    let prompt_tokens = pipeline.tokenize(&user_prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    // Run plaintext inference (ground truth)
    eprintln!("[plaintext] Running plaintext inference...");
    let plaintext_start = Instant::now();
    let plaintext_result = pipeline.plaintext_step(prompt_tokens[0]);
    let plaintext_time = plaintext_start.elapsed().as_secs_f64();

    let plaintext_token_id = plaintext_result.token_id as u32;
    let plaintext_token_str = pipeline.decode_token(plaintext_token_id)?;
    let plaintext_logits = plaintext_result.logits.clone();

    eprintln!("[plaintext] Plaintext step completed in {:.1}s", plaintext_time);
    eprintln!(
        "[plaintext] Predicted token: {} (ID: {}), Top-3 logits: {:?}",
        plaintext_token_str,
        plaintext_token_id,
        &plaintext_logits[..3]
    );
    eprintln!(
        "[plaintext] Hidden state range: [{:.2}, {:.2}] (first 5 values: {:?})",
        plaintext_result
            .hidden_state
            .iter()
            .cloned()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)))
            .0,
        plaintext_result
            .hidden_state
            .iter()
            .cloned()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)))
            .1,
        &plaintext_result.hidden_state[..5.min(plaintext_result.hidden_state.len())]
    );

    // Run FHE inference
    eprintln!("\n[fhe] Running FHE inference...");
    let fhe_start = Instant::now();
    let fhe_result = pipeline.step(prompt_tokens[0])?;
    let fhe_time = fhe_start.elapsed().as_secs_f64();

    let fhe_token_str = pipeline.decode_token(fhe_result.token_id)?;
    let _fhe_logits: Vec<f64> = fhe_result.top_logits.iter().map(|(_id, logit)| *logit as f64).collect();

    eprintln!("[fhe] FHE step completed in {:.1}s", fhe_time);
    eprintln!(
        "[fhe] Predicted token: {} (ID: {}), Top-3 logits: {:?}",
        fhe_token_str,
        fhe_result.token_id,
        &fhe_result.top_logits[..fhe_result.top_logits.len().min(3)]
    );
    eprintln!(
        "[fhe] Hidden state range: [{}, {}] (first 5 values: {:?})",
        fhe_result.hidden_state.iter().min().unwrap(),
        fhe_result.hidden_state.iter().max().unwrap(),
        &fhe_result.hidden_state[..5.min(fhe_result.hidden_state.len())]
    );

    // Compare hidden states
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

    // Compare logits
    eprintln!("\n[comparison] Comparing FHE vs plaintext logits...");

    let plaintext_max_idx = plaintext_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let plaintext_max_logit = plaintext_logits[plaintext_max_idx];

    eprintln!(
        "Plaintext max logit: {:.2} (token {})",
        plaintext_max_logit, plaintext_token_str
    );

    // Decode FHE output
    let _fhe_max_idx = fhe_result.top_logits.first().map(|(_id, _)| *id as usize).unwrap_or(0);
    let fhe_max_logit = fhe_result.top_logits.first().map(|(_, l)| *l as f64).unwrap_or(0.0);

    eprintln!("FHE max logit: {:.2} (token {})", fhe_max_logit, fhe_token_str);

    if fhe_result.token_id == plaintext_token_id {
        eprintln!("\n✅ SUCCESS: Tokens match! ({} = {})", fhe_token_str, plaintext_token_str);
    } else {
        eprintln!("\n❌ FAILURE: Tokens differ!");
        eprintln!(
            "   FHE:      {} ({}), logit: {:.2}",
            fhe_token_str, fhe_result.token_id, fhe_max_logit
        );
        eprintln!(
            "   Plaintext: {} ({}), logit: {:.2}",
            plaintext_token_str, plaintext_token_id, plaintext_max_logit
        );

        // Show top-5 logits for both
        let mut fhe_sorted: Vec<(usize, f64)> = fhe_result.top_logits.iter().map(|(i, l)| (*i as usize, *l as f64)).collect();
        fhe_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("\nFHE top-5 logits:");
        for (i, logit) in fhe_sorted.iter().take(5) {
            let token_str = pipeline.decode_token(*i as u32).unwrap_or_else(|_| "??".to_string());
            eprintln!("  {}: {} ({:.2})", i, token_str, logit);
        }

        eprintln!("\nPlaintext top-5 logits:");
        let mut plain_sorted: Vec<(usize, f64)> = plaintext_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        plain_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, logit) in plain_sorted.iter().take(5) {
            let token_str = pipeline.decode_token(*i as u32).unwrap_or_else(|_| "??".to_string());
            eprintln!("  {}: {} ({:.2})", i, token_str, logit);
        }

        eprintln!("\nPlaintext top-5 logits:");
        let mut plain_sorted: Vec<(usize, f64)> = plaintext_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        plain_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, logit) in plain_sorted.iter().take(5) {
            let token_str = pipeline.decode_token(*i as u32).unwrap_or_else(|_| "??".to_string());
            eprintln!("  {}: {} ({:.2})", i, token_str, logit);
        }
    }

    Ok(())
}
