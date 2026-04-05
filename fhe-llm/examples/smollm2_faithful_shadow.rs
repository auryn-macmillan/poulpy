/// FHE-faithful plaintext shadow with ablation toggles.
///
/// Runs ablation in the dequantized domain: starts from the working Refreshed
/// path and selectively enables FHE approximations to identify which causes
/// the most degradation.
///
/// Configs tested:
///   1. Baseline (all off, TP(8) quant) — should match Refreshed exactly
///   2. +relu_squared_attn only
///   3. +lut_rmsnorm only
///   4. +silu_lut only (7-bit)
///   5. +silu_lut 11-bit (reduced wrapping)
///   6. +quant7 only (7-bit vs 8-bit refresh precision)
///   7. lut_rms + silu_lut 11-bit (both key fixes)
///   8. Full FHE (all on, quant7, silu 7-bit)
///   9. Full FHE with silu 11-bit
///
/// Usage:
///   cargo run --release --example smollm2_faithful_shadow -- --raw "2+2=" 576 1536 30
///   cargo run --release --example smollm2_faithful_shadow -- --raw "2+2=" 576 1536 3
use std::process;
use std::time::Instant;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{DeqFaithfulShadowConfig, InferenceConfig, InferencePipeline, ModelSpec};
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
    let d_model: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(576);
    let d_ffn: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(1536);
    let n_layers: usize = args_iter.next().and_then(|s| s.parse().ok()).unwrap_or(30);

    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    eprintln!("=== SmolLM2 Dequantized Faithful Shadow Ablation ===");
    eprintln!("d_model={d_model}, d_ffn={d_ffn}, n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}");
    eprintln!("raw_mode={raw_mode}");

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

    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    let formatted = if raw_mode {
        user_prompt.clone()
    } else {
        pipeline.format_chat_prompt(Some("You are a helpful AI assistant."), &user_prompt)
    };
    let prompt_tokens = pipeline.tokenize(&formatted)?;
    eprintln!(
        "[main] prompt={:?} ({} tokens): {:?}",
        user_prompt,
        prompt_tokens.len(),
        prompt_tokens
    );

    // --- Reference: exact path ---
    let t = Instant::now();
    let exact = pipeline.plaintext_prompt_step_exact(&prompt_tokens)?;
    let exact_time = t.elapsed();
    let exact_h: Vec<f64> = exact.hidden_state.iter().map(|&v| v as f64).collect();

    println!("=== REFERENCE ===");
    println!(
        "Exact:     token={} {:?} ({:.3}s) top5={:?}",
        exact.token_id,
        exact.token_text,
        exact_time.as_secs_f64(),
        exact.top_logits,
    );

    // --- Reference: refreshed path ---
    let t = Instant::now();
    let refreshed = pipeline.plaintext_prompt_step_refreshed(&prompt_tokens)?;
    let refresh_time = t.elapsed();
    let refresh_h: Vec<f64> = refreshed.hidden_state.iter().map(|&v| v as f64).collect();
    let (linf, _, mae) = fhe_llm::plaintext_forward::error_metrics(&refresh_h, &exact_h);
    println!(
        "Refreshed: token={} {:?} ({:.3}s) L-inf={linf:.3} MAE={mae:.3} top5={:?}",
        refreshed.token_id,
        refreshed.token_text,
        refresh_time.as_secs_f64(),
        refreshed.top_logits,
    );

    // --- Dequantized-domain ablation ---
    let configs: Vec<(&str, DeqFaithfulShadowConfig)> = vec![
        ("Baseline (quant8)", DeqFaithfulShadowConfig::baseline()),
        (
            "+relu2_attn",
            DeqFaithfulShadowConfig {
                relu_squared_attn: true,
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "+lut_rmsnorm",
            DeqFaithfulShadowConfig {
                lut_rmsnorm: true,
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "+silu_lut (7b)",
            DeqFaithfulShadowConfig {
                silu_lut: true,
                silu_log_message_modulus: 7,
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "+silu_lut (11b)",
            DeqFaithfulShadowConfig {
                silu_lut: true,
                silu_log_message_modulus: 11,
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "+quant7",
            DeqFaithfulShadowConfig {
                quantize_refresh_bits: Some(7),
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "lut_rms+silu11b",
            DeqFaithfulShadowConfig {
                lut_rmsnorm: true,
                silu_lut: true,
                silu_log_message_modulus: 11,
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "lut_rms+relu2+quant7",
            DeqFaithfulShadowConfig {
                relu_squared_attn: true,
                lut_rmsnorm: true,
                silu_lut: false,
                quantize_refresh_bits: Some(7),
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "Full FHE (silu 7b)",
            DeqFaithfulShadowConfig {
                log_stats: true,
                ..DeqFaithfulShadowConfig::full_fhe()
            },
        ),
        (
            "Full FHE (silu 11b)",
            DeqFaithfulShadowConfig {
                relu_squared_attn: true,
                lut_rmsnorm: true,
                silu_lut: true,
                silu_rescaled: false,
                silu_log_message_modulus: 11,
                quantize_refresh_bits: Some(7),
                log_stats: true,
            },
        ),
        (
            "+silu_rsc (7b)",
            DeqFaithfulShadowConfig {
                silu_lut: true,
                silu_rescaled: true,
                silu_log_message_modulus: 7,
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "+silu_rsc (11b)",
            DeqFaithfulShadowConfig {
                silu_lut: true,
                silu_rescaled: true,
                silu_log_message_modulus: 11,
                ..DeqFaithfulShadowConfig::baseline()
            },
        ),
        (
            "Full FHE rsc (7b)",
            DeqFaithfulShadowConfig {
                log_stats: false,
                ..DeqFaithfulShadowConfig::full_fhe()
            },
        ),
        (
            "Full FHE rsc (11b)",
            DeqFaithfulShadowConfig {
                silu_log_message_modulus: 11,
                log_stats: false,
                ..DeqFaithfulShadowConfig::full_fhe()
            },
        ),
    ];

    println!("\n=== DEQ ABLATION ===");
    for (label, shadow_config) in &configs {
        let t = Instant::now();
        let result = pipeline.plaintext_prompt_step_faithful_shadow_deq(&prompt_tokens, shadow_config);
        let elapsed = t.elapsed();
        let result_h: Vec<f64> = result.hidden_state.iter().map(|&v| v as f64).collect();
        let (linf, _, mae) = fhe_llm::plaintext_forward::error_metrics(&result_h, &exact_h);
        let match_exact = result.token_id == exact.token_id;
        let match_refreshed = result.token_id == refreshed.token_id;
        println!(
            "{label:25} token={:5} {:20?} L-inf={linf:7.3} MAE={mae:7.3} ({:.3}s) match_exact={match_exact} match_refreshed={match_refreshed}  top5={:?}",
            result.token_id,
            result.token_text,
            elapsed.as_secs_f64(),
            result.top_logits,
        );
    }

    Ok(())
}
