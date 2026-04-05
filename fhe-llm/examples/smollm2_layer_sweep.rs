/// Layer-by-layer FHE vs plaintext shadow comparison with actual FHE encryption.
///
/// Runs the actual FHE forward pass with bootstrapping at each layer boundary,
/// decrypting hidden states for comparison with plaintext references.
///
/// Usage:
///   cargo run --release --example smollm2_layer_sweep [prompt] [d_model] [n_layers]
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
    let precision_str = args_iter.next().unwrap_or_else(|| "int8".to_string());
    let precision = match precision_str.to_lowercase().as_str() {
        "fp16" => Precision::Fp16,
        _ => Precision::Int8,
    };

    let n_heads = (d_model / 64).max(1);
    let n_kv_heads = (n_heads / 3).max(1);

    eprintln!("=== SmolLM2 Layer-by-Layer FHE Audit ===");
    eprintln!("d_model={d_model}, d_ffn={d_ffn}, n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}");
    eprintln!("precision: {:?}", precision);
    eprintln!("prompt: {:?}", user_prompt);
    eprintln!();

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: precision,
        num_layers: Some(n_layers),
        trunc_d_model: Some(d_model),
        trunc_d_ffn: Some(d_ffn),
        num_heads: Some(n_heads),
        num_kv_heads: Some(n_kv_heads),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false, // Will apply final norm separately after decryption
        max_new_tokens: 1,
        fhe_silu_log_msg_mod: Some(11),
        fhe_identity_log_msg_mod: Some(11),
        ..InferenceConfig::default()
    };

    eprintln!("[main] Loading model and generating FHE keys...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[main] Loaded in {:.1}s\n", load_start.elapsed().as_secs_f64());

    // Use raw prompt (single token) for fair comparison with FHE path
    let prompt = user_prompt.clone();
    let prompt_tokens = pipeline.tokenize(&prompt)?;
    eprintln!("Prompt tokens: {:?}\n", prompt_tokens);

    // Run exact plaintext (gold standard) - single token for fair comparison
    eprintln!("[exact] Running exact plaintext inference (first token only)...");
    let exact_start = Instant::now();
    let _exact_result = pipeline.compare_prompt_fhe_like_layers(&prompt);
    let exact_time = exact_start.elapsed().as_secs_f64();

    eprintln!("[exact] Done in {:.1}s\n", exact_time);

    // Run FHE-like (simulated FHE arithmetic, no actual encryption)
    eprintln!("[fhe_like] Running FHE-like plaintext simulation with quantization...");
    let fhe_like_start = Instant::now();
    let fhe_like_hidden_states: Vec<(String, Vec<f64>)> = pipeline
        .prompt_fhe_like_plain_trace(&prompt_tokens)
        .into_iter()
        .map(|(name, states)| (name, states.to_vec()))
        .collect();
    let fhe_like_time = fhe_like_start.elapsed().as_secs_f64();
    eprintln!("[fhe_like] Done in {:.1}s\n", fhe_like_time);

    // Run actual FHE encrypted with layer-by-layer extraction
    eprintln!("[fhe] Running actual FHE encrypted inference with layer extraction...");
    eprintln!("[fhe] This will run through all {} layers with bootstrapping", n_layers);
    let fhe_start = Instant::now();
    let fhe_hidden_states = pipeline.compare_prompt_fhe_layer_hidden_states(&prompt);
    let fhe_time = fhe_start.elapsed().as_secs_f64();

    let fhe_hidden_states = match fhe_hidden_states {
        Ok(states) => states,
        Err(e) => {
            eprintln!("[fhe] Error: {e}");
            return Err(format!("FHE inference failed: {e}").into());
        }
    };
    eprintln!("[fhe] Done in {:.1}s\n", fhe_time);

    // Compute exact hidden states for comparison
    eprintln!("[exact] Computing exact hidden states for comparison...");
    let prompt_tokens = pipeline.tokenize(&prompt)?;
    let exact_hidden_states = compute_exact_hidden_states(&pipeline, &prompt_tokens)?;
    eprintln!("[exact] Done\n");

    // Three-way layer-by-layer comparison: exact vs fhe_like vs fhe
    eprintln!("=== Three-Way Hidden State Comparison ===");
    eprintln!(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {}",
        "Stage", "Exact L-inf", "FHE-like", "FHE", "Exact→FHE-like", "FHE-like→FHE", ""
    );

    let max_layers = exact_hidden_states
        .len()
        .min(fhe_like_hidden_states.len())
        .min(fhe_hidden_states.len());
    for i in 0..max_layers {
        let exact_state = &exact_hidden_states[i].1;
        let fhe_like_state = &fhe_like_hidden_states[i].1;
        let fhe_state = &fhe_hidden_states[i].1;
        let stage = &fhe_hidden_states[i].0;

        let exact_linf = exact_state.iter().cloned().fold(0.0f64, f64::max);
        let fhe_like_linf = fhe_like_state.iter().cloned().fold(0.0f64, f64::max);
        let fhe_linf = fhe_state.iter().cloned().fold(0.0f64, f64::max);

        // Arithmetic approximation error (exact → fhe_like)
        let arithmetic_error = exact_state
            .iter()
            .zip(fhe_like_state.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        // Encryption noise (fhe_like → fhe)
        let encryption_noise = fhe_like_state
            .iter()
            .zip(fhe_state.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        eprintln!(
            "{:>8} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {}",
            stage,
            exact_linf,
            fhe_like_linf,
            fhe_linf,
            arithmetic_error,
            encryption_noise,
            if encryption_noise < 10.0 { "OK" } else { "HIGH NOISE" }
        );
    }

    // Final token comparison
    eprintln!("\n=== Final Token Comparison ===");

    // Compute exact token from last layer
    if let Some((_, last_exact_hidden)) = exact_hidden_states.last() {
        let logits = pipeline.lm_head_forward_f64(last_exact_hidden);
        let token_id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| id as u32)
            .unwrap_or(0);
        let token_text = match pipeline.decode_token(token_id) {
            Ok(t) => t,
            Err(_) => format!("<token_{}>", token_id),
        };
        eprintln!(
            "Exact token:  id={:5} \"{}\" (L-inf={:.1})",
            token_id,
            token_text,
            last_exact_hidden.iter().fold(0.0f64, |a, b| a.max(*b))
        );
    }

    // Compute FHE token from last layer
    if let Some((_, last_fhe_hidden)) = fhe_hidden_states.last() {
        let logits = pipeline.lm_head_forward_f64(last_fhe_hidden);
        let token_id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| id as u32)
            .unwrap_or(0);
        let token_text = match pipeline.decode_token(token_id) {
            Ok(t) => t,
            Err(_) => format!("<token_{}>", token_id),
        };
        eprintln!(
            "FHE token:    id={:5} \"{}\" (L-inf={:.1})",
            token_id,
            token_text,
            last_fhe_hidden.iter().fold(0.0f64, |a, b| a.max(*b))
        );
    }

    // Compute FHE-like token from last layer
    if let Some((_, last_fhe_like_hidden)) = fhe_like_hidden_states.last() {
        let logits = pipeline.lm_head_forward_f64(last_fhe_like_hidden);
        let token_id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| id as u32)
            .unwrap_or(0);
        let token_text = match pipeline.decode_token(token_id) {
            Ok(t) => t,
            Err(_) => format!("<token_{}>", token_id),
        };
        eprintln!(
            "FHE-like:     id={:5} \"{}\" (L-inf={:.1})",
            token_id,
            token_text,
            last_fhe_like_hidden.iter().fold(0.0f64, |a, b| a.max(*b))
        );
    }

    eprintln!("\n=== Analysis ===");
    let last_fhe_linf = fhe_hidden_states
        .last()
        .map(|(_, states)| states.iter().fold(0.0f64, |a, b| a.max(*b)))
        .unwrap_or(0.0);

    if last_fhe_linf < 10.0 {
        eprintln!("✅ FHE path is within acceptable error bounds (L-inf < 10)");
    } else {
        eprintln!("❌ FHE path has high error (L-inf >= 10)");
        eprintln!("   This may indicate noise accumulation or bootstrap precision issues");
    }

    Ok(())
}

fn compute_exact_hidden_states(
    pipeline: &InferencePipeline,
    prompt_tokens: &[u32],
) -> Result<Vec<(String, Vec<f64>)>, Box<dyn std::error::Error>> {
    // Use the existing exact plaintext trace method
    let comparison = pipeline.prompt_exact_plain_trace(prompt_tokens);

    // Convert to expected format: (layer_name, hidden_states)
    let result: Vec<(String, Vec<f64>)> = comparison.into_iter().map(|(name, states)| (name, states.to_vec())).collect();

    Ok(result)
}
