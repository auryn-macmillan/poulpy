//! Full FHE inference with homomorphic LM head.
//!
//! Runs all 30 transformer layers + LM head under FHE (encrypted),
//! eliminating the cleartext noise amplification bottleneck.
//!
//! Usage:
//!   cargo +nightly run --release --example fhe_full_lm_head "2+2=" 128 30

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use poulpy_hal::api::ModuleNew;
use poulpy_hal::layouts::Module;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
type BE = poulpy_cpu_ref::FFT64Ref;
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
type BE = poulpy_cpu_avx::FFT64Avx;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = args.get(1).map(|s| s.as_str()).unwrap_or("2+2=").to_string();
    let trunc_d_model: usize = args.get(2).map(|s| s.as_str()).unwrap_or("128").parse().unwrap();
    let num_layers: usize = args.get(3).map(|s| s.as_str()).unwrap_or("30").parse().unwrap();

    println!("=== Full FHE Inference (Homomorphic LM Head) ===");
    println!("Model: SmolLM2-135M-Instruct");
    println!("Prompt: {:?}", prompt);
    println!("Layers: {}, d_model: {}", num_layers, trunc_d_model);
    println!("Security: {:?}, Precision: {:?}", SecurityLevel::Bits100, Precision::Fp16);
    println!();

    let model_path = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
    let tokenizer_path = "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

    println!("Loading model and tokenizer...");
    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Fp16,
        num_layers: Some(num_layers),
        trunc_d_model: Some(trunc_d_model),
        trunc_d_ffn: Some(256),
        num_heads: Some(2),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 1,
        fhe_log_message_modulus: Some(13),
        ..InferenceConfig::default()
    };

    let pipeline = InferencePipeline::load(
        model_path,
        tokenizer_path,
        ModelSpec::smollm2_135m_instruct(),
        config,
    )?;

    let effective_d_model = pipeline.effective_dims().d_model;
    let vocab_size = pipeline.model_spec().vocab_size;
    println!("Effective dimensions: d_model={}, vocab_size={}", effective_d_model, vocab_size);
    println!();

    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).expect("Tokenizer");
    let encoding = tokenizer.encode(prompt.clone(), true).expect("Encode");
    let token_ids = encoding.get_ids().to_vec();
    let tokens = tokenizer.decode(&token_ids, true).expect("Decode");
    println!("Prompt tokens ({}): {:?}", token_ids.len(), tokens);
    let first_token = token_ids[0];
    println!();

    println!("Embedding first token ({})...", first_token);
    let embeddings = pipeline.embedding();
    let full_embedding = embeddings.lookup(first_token as usize).to_vec();
    let input_embeddings: Vec<i64> = full_embedding[..effective_d_model].to_vec();
    println!("Input shape: {} dims", input_embeddings.len());
    println!();

    println!("Running {} FHE transformer layers...", num_layers);
    let hidden_encrypted = pipeline.fhe_forward_pass_to_hidden(&input_embeddings);
    println!("Hidden state encrypted: {} cts", hidden_encrypted.len());
    println!();

    println!("Encrypting LM head weights...");
    let lm_head_cleartext = pipeline.lm_head_cleartext().as_ref().unwrap();
    let module = Module::new_marker(1);
    
    let start = std::time::Instant::now();
    let encrypted_lm_head = fhe_llm::model_loader::EncryptedLMHead::from_cleartext::<BE>(
        &module,
        pipeline.params(),
        &pipeline.eval_key,
        lm_head_cleartext,
        lm_head_cleartext.quant_info.scale,
        Precision::Fp16,
    );
    let encrypt_time = start.elapsed();
    println!("LM head encrypted: {} × {} weights", encrypted_lm_head.vocab_size, encrypted_lm_head.d_model);
    println!("Encryption time: {:.2}s", encrypt_time.as_secs_f64());
    println!();

    println!("Computing {} logits under FHE...", vocab_size);
    let start = std::time::Instant::now();
    let logits_encrypted = encrypted_lm_head.forward_with_batch::<BE>(
        &module,
        &pipeline.eval_key,
        &hidden_encrypted,
        1024,
    );
    let fhe_lm_head_time = start.elapsed();
    println!("FHE LM head time: {:.2}s", fhe_lm_head_time.as_secs_f64());
    println!();

    println!("Decrypting logits...");
    let logits_cleartext: Vec<f64> = logits_encrypted
        .iter()
        .map(|ct| {
            let decoded = fhe_llm::encrypt::_FHE_LLM_decrypt::<BE>(
                &module,
                &pipeline.key,
                ct,
                poulpy_core::layouts::GLWEInfos::new(),
            );
            decoded[0] as f64 / (1 << 13) as f64
        })
        .collect();

    let logits_min: f64 = logits_cleartext.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
    let logits_max: f64 = logits_cleartext.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    println!("Logits: min={:.2}, max={:.2}", logits_min, logits_max);
    println!();

    let mut indexed: Vec<(usize, f64)> = logits_cleartext.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 10 logits:");
    for (i, (idx, logit)) in indexed.iter().take(10).enumerate() {
        let token_text = tokenizer.decode(&[*idx as u32], true).unwrap_or_else(|_| "??".to_string());
        println!("  {}: token={} ({}), logit={:.2}", i + 1, idx, token_text, logit);
    }
    println!();

    let best_token = logits_cleartext
        .iter()
        .cloned()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;
    let best_token_text = tokenizer.decode(&[best_token as u32], true)?;
    println!("Predicted token: {} (ID: {})", best_token_text, best_token);
    println!("Expected token: \"4\" (ID: 271)");
    println!("Match: {}", best_token == 271);
    println!();

    println!("=== Complete ===");
    println!("Full FHE architecture:");
    println!("  Server: FHE transformer + FHE LM head (all encrypted)");
    println!("  Client: Decrypt logits → softmax → argmax (cleartext)");
    println!();
    println!("Benefits:");
    println!("  - Eliminates cleartext LM head noise amplification");
    println!("  - All computation under FHE, noise stays bounded");
    println!("  - Potentially correct token prediction");
    println!();
    println!("Costs:");
    println!("  - ~30× slower than split-point (49k FHE dot products)");
    println!("  - ~400MB memory for encrypted LM head weights");
    println!("  - ~2 hours runtime at d_model=128");

    Ok(())
}
    println!();

    // Argmax
    let best_token = logits_cleartext
        .iter()
        .cloned()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;
    let best_token_text = tokenizer.decode(&[best_token as u32], true)?;
    println!("Predicted token: {} (ID: {})", best_token_text, best_token);
    println!("Expected token: \"4\" (ID: 271)");
    println!("Match: {}", best_token == 271);
    println!();

    println!("=== Complete ===");
    println!("Full FHE architecture:");
    println!("  Server: FHE transformer + FHE LM head (all encrypted)");
    println!("  Client: Decrypt logits → softmax → argmax (cleartext)");
    println!();
    println!("Benefits:");
    println!("  - Eliminates cleartext LM head noise amplification");
    println!("  - All computation under FHE, noise stays bounded");
    println!("  - Potentially correct token prediction");
    println!();
    println!("Costs:");
    println!("  - ~30× slower than split-point (49k FHE dot products)");
    println!("  - ~400MB memory for encrypted LM head weights");
    println!("  - ~2 hours runtime at d_model=128");

    Ok(())
}
