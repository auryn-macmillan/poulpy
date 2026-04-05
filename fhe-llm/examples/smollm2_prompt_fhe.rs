/// Prompt-conditioned SmolLM2 FHE comparison.
///
/// Runs three inference paths on the same prompt:
///   1. Exact dequantized plaintext (gold-standard reference)
///   2. Refreshed plaintext shadow (quantized, wrapping, exact nonlinearities)
///   3. FHE encrypted (real homomorphic encryption with bootstrap)
///
/// Usage:
///   cargo run --release --example smollm2_prompt_fhe [prompt] [d_model] [n_layers]
///
/// Defaults: prompt="What is 2+2?", d_model=64, n_layers=1
///
/// At d_model=576, n_layers=30, a 12-token prompt with O(L^2) attention will
/// take many hours. Start small to validate, then scale incrementally.
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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let raw_mode = args
        .iter()
        .position(|a| a == "--raw")
        .map(|i| {
            args.remove(i);
            true
        })
        .unwrap_or(false);
    let mut args_iter = args.into_iter();
    let user_prompt = args_iter.next().unwrap_or_else(|| "What is 2+2?".to_string());
    let d_model: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let d_ffn: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(d_model * 2);
    let n_layers: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    // SmolLM2-135M: d_head=64, GQA ratio 9:3 = 3:1
    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    eprintln!("=== SmolLM2 Prompt-Conditioned FHE Comparison ===");
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
        apply_final_norm: true,
        max_new_tokens: 1,
        ..InferenceConfig::default()
    };

    eprintln!("[main] Loading model and generating FHE keys...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!(
        "[main] Model loaded and keys generated in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    // Format the prompt (optionally using the chat template)
    let formatted = if raw_mode {
        user_prompt.clone()
    } else {
        pipeline.format_chat_prompt(Some("You are a helpful AI assistant."), &user_prompt)
    };
    let prompt_tokens = pipeline.tokenize(&formatted)?;
    eprintln!(
        "[main] {} prompt: {} chars, {} tokens",
        if raw_mode { "Raw" } else { "Formatted" },
        formatted.len(),
        prompt_tokens.len()
    );
    eprintln!("[main] Token IDs: {:?}", prompt_tokens);
    eprintln!();

    // --- Phase 1: Exact dequantized plaintext (gold standard) ---
    eprintln!("[phase 1] Exact plaintext prompt inference ({n_layers} layers)...");
    let exact_start = Instant::now();
    let exact_result = pipeline.plaintext_prompt_step_exact(&prompt_tokens)?;
    let exact_time = exact_start.elapsed();
    eprintln!(
        "[phase 1] Done in {:.3}s: token={} {:?}",
        exact_time.as_secs_f64(),
        exact_result.token_id,
        exact_result.token_text,
    );
    eprintln!("[phase 1] Top logits: {:?}", exact_result.top_logits);
    eprintln!();

    // --- Phase 2: Refreshed plaintext shadow ---
    eprintln!("[phase 2] Refreshed plaintext prompt shadow ({n_layers} layers)...");
    let refresh_start = Instant::now();
    let refreshed_result = pipeline.plaintext_prompt_step_refreshed(&prompt_tokens)?;
    let refresh_time = refresh_start.elapsed();
    eprintln!(
        "[phase 2] Done in {:.3}s: token={} {:?}",
        refresh_time.as_secs_f64(),
        refreshed_result.token_id,
        refreshed_result.token_text,
    );
    eprintln!("[phase 2] Top logits: {:?}", refreshed_result.top_logits);

    // Error vs exact
    let exact_hidden: Vec<f64> = exact_result.hidden_state.iter().map(|&v| v as f64).collect();
    let refreshed_hidden: Vec<f64> = refreshed_result.hidden_state.iter().map(|&v| v as f64).collect();
    let (ref_linf, ref_l2, ref_mae) = fhe_llm::plaintext_forward::error_metrics(&refreshed_hidden, &exact_hidden);
    eprintln!(
        "[phase 2] vs exact: L-inf={:.3} L2={:.3} MAE={:.3}",
        ref_linf, ref_l2, ref_mae
    );
    eprintln!();

    // --- Phase 3: FHE encrypted ---
    eprintln!(
        "[phase 3] FHE encrypted prompt inference ({n_layers} layers, {} tokens)...",
        prompt_tokens.len()
    );
    eprintln!("[phase 3] This may take a while...");
    let fhe_start = Instant::now();
    let fhe_result = pipeline.step_prompt_refreshed(&prompt_tokens)?;
    let fhe_time = fhe_start.elapsed();
    eprintln!(
        "[phase 3] Done in {:.1}s (FHE core: {:.1}s): token={} {:?}",
        fhe_time.as_secs_f64(),
        fhe_result.fhe_time.as_secs_f64(),
        fhe_result.token_id,
        fhe_result.token_text,
    );
    eprintln!("[phase 3] Top logits: {:?}", fhe_result.top_logits);

    // Error vs exact and vs refreshed
    let fhe_hidden: Vec<f64> = fhe_result.hidden_state.iter().map(|&v| v as f64).collect();
    let (fhe_linf, fhe_l2, fhe_mae) = fhe_llm::plaintext_forward::error_metrics(&fhe_hidden, &exact_hidden);
    let (fhe_vs_ref_linf, fhe_vs_ref_l2, fhe_vs_ref_mae) =
        fhe_llm::plaintext_forward::error_metrics(&fhe_hidden, &refreshed_hidden);
    eprintln!(
        "[phase 3] vs exact:     L-inf={:.3} L2={:.3} MAE={:.3}",
        fhe_linf, fhe_l2, fhe_mae,
    );
    eprintln!(
        "[phase 3] vs refreshed: L-inf={:.3} L2={:.3} MAE={:.3}",
        fhe_vs_ref_linf, fhe_vs_ref_l2, fhe_vs_ref_mae,
    );
    eprintln!();

    // --- Summary ---
    println!("=== RESULTS ===");
    println!();
    println!("Config: d_model={d_model} d_ffn={d_ffn} n_layers={n_layers} n_heads={n_heads} n_kv_heads={n_kv_heads}");
    println!("Prompt: {:?} ({} tokens)", user_prompt, prompt_tokens.len());
    println!();
    println!(
        "Exact:     token={} {:?} (time={:.3}s) top5={:?}",
        exact_result.token_id,
        exact_result.token_text,
        exact_time.as_secs_f64(),
        exact_result.top_logits,
    );
    println!(
        "Refreshed: token={} {:?} (time={:.3}s) top5={:?}",
        refreshed_result.token_id,
        refreshed_result.token_text,
        refresh_time.as_secs_f64(),
        refreshed_result.top_logits,
    );
    println!(
        "FHE:       token={} {:?} (fhe={:.1}s total={:.1}s) top5={:?}",
        fhe_result.token_id,
        fhe_result.token_text,
        fhe_result.fhe_time.as_secs_f64(),
        fhe_time.as_secs_f64(),
        fhe_result.top_logits,
    );
    println!();
    println!(
        "Refreshed vs Exact: L-inf={:.3} L2={:.3} MAE={:.3}",
        ref_linf, ref_l2, ref_mae
    );
    println!(
        "FHE vs Exact:       L-inf={:.3} L2={:.3} MAE={:.3}",
        fhe_linf, fhe_l2, fhe_mae
    );
    println!(
        "FHE vs Refreshed:   L-inf={:.3} L2={:.3} MAE={:.3}",
        fhe_vs_ref_linf, fhe_vs_ref_l2, fhe_vs_ref_mae
    );
    println!();
    println!(
        "Token match: exact_vs_refreshed={} exact_vs_fhe={} refreshed_vs_fhe={}",
        exact_result.token_id == refreshed_result.token_id,
        exact_result.token_id == fhe_result.token_id,
        refreshed_result.token_id == fhe_result.token_id,
    );
    println!();
    println!(
        "SUMMARY: d={d_model} L={n_layers} T={} exact={} refreshed={} fhe={} match_exact_fhe={} fhe_time={:.1}s linf_fhe_vs_exact={:.3}",
        prompt_tokens.len(),
        exact_result.token_id,
        refreshed_result.token_id,
        fhe_result.token_id,
        exact_result.token_id == fhe_result.token_id,
        fhe_result.fhe_time.as_secs_f64(),
        fhe_linf,
    );

    Ok(())
}
