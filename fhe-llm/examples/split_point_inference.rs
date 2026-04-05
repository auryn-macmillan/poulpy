//! FHE inference with split-point optimization: server runs transformer only,
//! client computes LM head (unembed + softmax + argmax).
//!
//! This demonstrates the architecture where the server returns encrypted hidden
//! state and the client does: decrypt → RMSNorm → unembed → softmax → argmax.

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args.get(1).map(|s| s.as_str()).unwrap_or("2+2=").to_string();
    let trunc_d_model: usize = args.get(2).map(|s| s.as_str()).unwrap_or("128").parse().unwrap();
    let num_layers: usize = args.get(3).map(|s| s.as_str()).unwrap_or("30").parse().unwrap();
    let trunc_d_ffn = 256usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;

    println!("=== Split-Point FHE Inference (Server → Client LM Head) ===");
    println!("Model: SmolLM2-135M-Instruct");
    println!("Prompt: {:?}", prompt);
    println!("Layers: {}, d_model: {}, d_ffn: {}", num_layers, trunc_d_model, trunc_d_ffn);
    println!("Heads: {} Q, {} KV (GQA)", num_heads, num_kv_heads);
    println!("Security: {:?}, Precision: {:?}", SecurityLevel::Bits100, Precision::Fp16);
    println!();

    // Load model
    let model_path = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
    let tokenizer_path =
        "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

    println!("Loading model and tokenizer...");
    let pipeline = InferencePipeline::load(
        model_path,
        tokenizer_path,
        ModelSpec::smollm2_135m_instruct(),
        InferenceConfig {
            security: SecurityLevel::Bits100,
            precision: Precision::Fp16,
            num_layers: Some(num_layers),
            trunc_d_model: Some(trunc_d_model),
            trunc_d_ffn: Some(trunc_d_ffn),
            num_heads: Some(num_heads),
            num_kv_heads: Some(num_kv_heads),
            softmax_strategy: SoftmaxStrategy::ReluSquared,
            apply_final_norm: true,
            max_new_tokens: 1,
            fhe_log_message_modulus: Some(13), // 8192 levels for FP16 precision
            ..InferenceConfig::default()
        },
    )?;

    let effective_d_model = pipeline.effective_dims().d_model;
    let vocab_size = pipeline.model_spec().vocab_size;
    println!(
        "Effective dimensions: d_model={}, vocab_size={}",
        effective_d_model, vocab_size
    );
    println!();

    // Tokenize (use single token for simplicity)
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Tokenizer");
    let encoding = tokenizer.encode(prompt.clone(), true).expect("Encode");
    let token_ids = encoding.get_ids().to_vec();
    let tokens = tokenizer.decode(&token_ids, true).expect("Decode");
    println!("Prompt tokens ({}): {:?}", token_ids.len(), tokens);
    // Use first token for simplicity
    let first_token = token_ids[0];
    println!();

    // Get embeddings from pipeline
    println!("Accessing embeddings from pipeline...");
    let embeddings = pipeline.embedding();
    println!("Embeddings shape: {} × {}", embeddings.vocab_size, embeddings.d_model);
    println!();

    // Embed prompt (use first token only)
    println!("Embedding first token ({})...", first_token);
    let full_embedding = embeddings.lookup(first_token as usize).to_vec();
    // Truncate to effective d_model
    let input_embeddings: Vec<i64> = full_embedding[..effective_d_model].to_vec();
    println!(
        "Input shape: {} dims (truncated from {})",
        input_embeddings.len(),
        full_embedding.len()
    );
    println!("First 4 dims: {:?}", &input_embeddings[..4.min(input_embeddings.len())]);
    println!();

    // Run FHE forward pass (server side)
    println!("Running FHE transformer layers...");
    let hidden_encrypted = pipeline.fhe_forward_pass_to_hidden(&input_embeddings);
    println!("Hidden state encrypted: {} cts", hidden_encrypted.len());
    println!();

    // Decrypt hidden state (client side)
    println!("Decrypting hidden state...");
    let hidden_cleartext = pipeline.decrypt_vec(&hidden_encrypted)?;
    println!("Hidden state decrypted: {} values", hidden_cleartext.len());
    let hidden_min: i8 = hidden_cleartext.iter().cloned().min().unwrap();
    let hidden_max: i8 = hidden_cleartext.iter().cloned().max().unwrap();
    let hidden_l2 = (hidden_cleartext.iter().map(|&v| v as f64 * v as f64).sum::<f64>()).sqrt();
    println!(
        "Hidden stats: min={}, max={}, L2_norm={:.2}",
        hidden_min, hidden_max, hidden_l2
    );

    // Signal preservation analysis
    println!();
    println!("=== Per-Layer Signal Preservation ===");
    println!("Layer 30 (final): L2_norm = {:.2}", hidden_l2);
    println!("Expected L2_norm: ~1442 (from 1-layer baseline)");
    println!("Signal preservation: {:.1}% (vs 100% baseline)", (hidden_l2 / 1442.0) * 100.0);
    println!();

    // Apply final RMSNorm (client side)
    println!("Applying final RMSNorm...");
    let final_norm_gamma = pipeline.final_norm_gamma();
    let gamma_f64: Vec<f64> = final_norm_gamma.as_ref().unwrap().iter().map(|&v| v as f64).collect();
    let hidden_normed = apply_rms_norm_cleartext(&hidden_cleartext, &gamma_f64);
    let normed_min: f64 = hidden_normed.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
    let normed_max: f64 = hidden_normed.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    println!("RMSNorm output: min={:.3}, max={:.3}", normed_min, normed_max);
    println!();

    // LM head (cleartext matmul using lm_head_cleartext weights)
    println!("Computing LM head (cleartext)...");
    let lm_head = pipeline.lm_head_cleartext().as_ref().unwrap();
    let lm_head_weights: Vec<f64> = lm_head.weights.iter().flat_map(|row| row.iter().map(|&v| v as f64)).collect();
    let logits = lm_head_matmul(&lm_head_weights, &hidden_normed, lm_head.vocab_size, lm_head.d_model);
    println!("Logits shape: {} tokens", logits.len());
    let logits_min: f64 = logits.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
    let logits_max: f64 = logits.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let logits_mean: f64 = logits.iter().cloned().sum::<f64>() / logits.len() as f64;
    println!(
        "Logits stats: min={:.2}, max={:.2}, mean={:.2}",
        logits_min, logits_max, logits_mean
    );
    println!();

    // Top-10 logits
    let mut indexed: Vec<(usize, f64)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 10 logits:");
    for (i, (idx, logit)) in indexed.iter().take(10).enumerate() {
        let token_text = tokenizer.decode(&[*idx as u32], true).unwrap_or_else(|_| "??".to_string());
        println!("  {}: token={} ({}), logit={:.2}", i + 1, idx, token_text, logit);
    }
    println!();

    // Argmax
    let best_token = logits
        .iter()
        .cloned()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;
    let best_token_text = tokenizer.decode(&[best_token as u32], true)?;
    println!("Predicted token: {} (ID: {})", best_token_text, best_token);
    println!();

    println!("=== Complete ===");
    println!("Split-point architecture:");
    println!("  Server: FHE transformer layers (encrypted)");
    println!("  Client: Decrypt → RMSNorm → unembed → softmax → argmax (cleartext)");
    println!();
    println!("Benefits:");
    println!("  - Eliminates encrypted LM head computation (49k × d_model matmul)");
    println!("  - Reduces data transfer from 49k logits to d_model hidden state");
    println!("  - Client-side softmax/argmax is trivial (O(vocab) vs O(d_model × vocab))");

    Ok(())
}

/// Client-side RMSNorm (matches FHE version semantics)
fn apply_rms_norm_cleartext(x: &[i8], weight: &[f64]) -> Vec<f64> {
    let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mean_sq = x_f64.iter().map(|&v| v * v).sum::<f64>() / x_f64.len() as f64;
    let rms = (mean_sq + 1e-6).sqrt();
    let inv_rms = 1.0 / rms;
    x_f64.iter().map(|&v| v * inv_rms * weight[0]).collect()
}

/// Cleartext matrix-vector multiplication for LM head
fn lm_head_matmul(weights: &[f64], hidden: &[f64], vocab_size: usize, d_model: usize) -> Vec<f64> {
    let mut logits = vec![0.0; vocab_size];
    for i in 0..vocab_size {
        let base = i * d_model;
        for j in 0..d_model {
            logits[i] += weights[base + j] * hidden[j];
        }
    }
    logits
}
