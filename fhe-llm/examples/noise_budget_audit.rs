//! Systematic noise budget audit for FHE_LLM FHE inference

use fhe_llm::{
    inference::{InferenceConfig, InferencePipeline, ModelSpec},
    noise::NoiseTracker,
    params::{ModelDims, Precision, SecurityLevel},
};
use tokenizers::Tokenizer;

fn main() {
    let prompt = std::env::args().nth(1).unwrap_or_else(|| "2+2=".to_string());

    println!("========================================");
    println!("Noise Budget Audit");
    println!("========================================");
    println!("Prompt: {:?}", prompt);

    // Load tokenizer
    let tokenizer_path =
        "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    // Tokenize
    let encoding = tokenizer.encode(prompt.trim(), true).expect("Failed to tokenize");
    let tokens: Vec<u32> = encoding.get_ids().to_vec();
    let t = tokens.len();
    println!("\nTokens: {} ({:?})", t, prompt);

    // Load model with FP16 precision
    let model_path = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";

    let dims = ModelDims {
        d_model: 576,
        d_head: 64,
        n_heads: 9,
        n_kv_heads: 3,
        d_ffn: 1536,
        n_layers: 1,
        n_experts: 1,
        n_active_experts: 1,
    };

    let spec = ModelSpec {
        dims,
        embed_name: "model.embed_tokens.weight".to_string(),
        lm_head_name: "model.embed_tokens.weight".to_string(),
        final_norm_name: "model.norm.weight".to_string(),
        rope_theta: 10000.0,
        max_seq_len: 2048,
        bos_token_id: 1,
        eos_token_id: 2,
    };

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Fp16,
        num_layers: Some(1),
        trunc_d_model: None,
        trunc_d_ffn: None,
        num_heads: None,
        num_kv_heads: None,
        softmax_strategy: fhe_llm::attention::SoftmaxStrategy::PolynomialDeg4,
        apply_final_norm: true,
        max_new_tokens: 1,
        key_seed: [0u8; 32],
        eval_seed_a: [1u8; 32],
        eval_seed_b: [2u8; 32],
        fhe_silu_log_msg_mod: Some(10),
        fhe_identity_log_msg_mod: Some(10),
        fhe_frequent_bootstrap: false,
    };

    // Extract fields before moving into load()
    let d_model = spec.dims.d_model;
    let d_ffn = spec.dims.d_ffn;
    let d_head = spec.dims.d_head;
    let n_heads = spec.dims.n_heads;
    let scale_bits = match config.precision {
        Precision::Int8 => 8,
        Precision::Fp16 => 14,
    };

    // Print config fields before moving it into load()
    println!("\nModel loaded successfully");
    println!("Precision: {:?}", config.precision);
    println!("Security: {:?}", config.security);

    let pipeline = InferencePipeline::load(model_path, tokenizer_path, spec, config).expect("Failed to load model");

    // Initialize noise tracker
    let mut noise = NoiseTracker::fresh();

    println!("\n=== Noise Budget Analysis ===");
    println!(
        "d_model: {}, d_ffn: {}, n_heads: {}, d_head: {}",
        d_model, d_ffn, n_heads, d_head
    );
    println!("Scale bits: {}", scale_bits);
    println!("Bootstrap levels: {}", 1u64 << (scale_bits - 1));
    println!(
        "Bootstrap quantization error: ±{:.6}",
        1.0 / (1u64 << (scale_bits - 1)) as f64
    );

    // Simulate noise growth through transformer sublayers
    println!("\n--- Sublayer Noise Accumulation ---");

    // 1. Initial encryption (negligible, already set by fresh())
    println!(
        "After init encryption: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 2. Pre-attn RMSNorm (negligible polynomial noise)
    noise.rms_norm(3, 1.0);
    println!(
        "After pre-attn RMSNorm: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 3. QKV projection
    let qkv_weight_l2 = (d_model as f64 * 6.4 * 6.4).sqrt();
    noise.qkv_projection(qkv_weight_l2);
    println!(
        "After QKV projection: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 4. Attention scores
    let scores_noise = noise.clone();
    noise.attention_scores(&scores_noise, d_model);
    println!(
        "After attention scores: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 5. Bootstrap #1
    let bootstrap_error = 1.0 / (1u64 << (scale_bits - 1)) as f64;
    noise.bootstrap_quantization(scale_bits, scale_bits);
    println!(
        "After bootstrap #1: std_dev={:.6}, L-inf≈{:.6} (quantization: ±{:.6})",
        noise.std_dev(),
        noise.linf_bound(),
        bootstrap_error
    );

    // 6. RMSNorm
    noise.rms_norm(3, 1.0);
    println!(
        "After RMSNorm: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 7. FFN gate/up projection
    let ffn_weight_l2 = (d_ffn as f64 * 6.4 * 6.4).sqrt();
    noise.ffn_gate_up(ffn_weight_l2);
    println!(
        "After FFN gate/up: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 8. SiLU activation
    let noise_before_silu = noise.clone();
    noise.silu_activation(&noise_before_silu, d_model);
    println!(
        "After SiLU LUT: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 9. Down projection
    noise.ffn_down(ffn_weight_l2);
    println!(
        "After down projection: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    // 10. Bootstrap #2
    noise.bootstrap_quantization(scale_bits, scale_bits);
    println!(
        "After bootstrap #2: std_dev={:.6}, L-inf≈{:.6} (quantization: ±{:.6})",
        noise.std_dev(),
        noise.linf_bound(),
        bootstrap_error
    );

    // 11. Final RMSNorm
    noise.rms_norm(3, 1.0);
    println!(
        "After final RMSNorm: std_dev={:.6}, L-inf≈{:.6}",
        noise.std_dev(),
        noise.linf_bound()
    );

    println!("\n=== Summary ===");
    println!("Final hidden state noise: L-inf ≈ {:.6}", noise.linf_bound());
    println!("Required for correct token: L-inf < 0.015");
    println!("Gap: {:.1}× too large", 0.015 / noise.linf_bound());

    // Compare with INT8 precision
    println!("\n--- Comparison: INT8 vs FP16 ---");
    let int8_scale = 8u32;
    let fp16_scale = 14u32;
    println!(
        "INT8: {} bits ({} levels), error ±{:.6}",
        int8_scale,
        1u64 << int8_scale,
        1.0 / (1u64 << (int8_scale - 1)) as f64
    );
    println!(
        "FP16: {} bits ({} levels), error ±{:.6}",
        fp16_scale,
        1u64 << fp16_scale,
        1.0 / (1u64 << (fp16_scale - 1)) as f64
    );
    println!("FP16 precision gain: {}×", (1u64 << fp16_scale) / (1u64 << int8_scale));

    println!("\n--- Recommendation ---");
    println!("Current bootstrap frequency: once per layer");
    println!("Problem: FHE computation noise (QKV, attention, FFN) dominates");
    println!("Solution: Add more bootstrap points (after QKV, attention, FFN)");
    println!("Target: Reduce per-operation noise to < 0.015 L-inf");
}
