/// Compare cleartext vs encrypted LM head on toy model
/// This validates the encrypted LM head implementation matches cleartext
use fhe_llm::{
    model_loader::{EncryptedLMHead, LMHead, QuantInfo},
    params::{FHE_LLMParams, Precision, SecurityLevel},
    plaintext_forward::lm_head_forward_i8,
};

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
type BE = poulpy_cpu_ref::FFT64Ref;
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
type BE = poulpy_cpu_avx::FFT64Avx;

use fhe_llm::encoding::{decode_int8, encode_int8};
use fhe_llm::encrypt::{_FHE_LLM_decrypt, _FHE_LLM_encrypt, FHE_LLMEvalKey, FHE_LLMKey};
use poulpy_core::layouts::GLWE;
use poulpy_hal::api::ModuleNew;
use poulpy_hal::layouts::Module;

fn main() {
    println!("=== LM Head Correctness Comparison ===\n");

    // Tiny model: vocab=10, d_model=4
    let vocab_size = 10;
    let d_model = 4;

    // Generate deterministic weights (seed=42)
    let weights: Vec<Vec<i64>> = vec![
        vec![1, 2, 3, 4],     // vocab[0]
        vec![5, 6, 7, 8],     // vocab[1]
        vec![9, 10, 11, 12],  // vocab[2]
        vec![13, 14, 15, 16], // vocab[3]
        vec![17, 18, 19, 20], // vocab[4]
        vec![21, 22, 23, 24], // vocab[5]
        vec![25, 26, 27, 28], // vocab[6]
        vec![29, 30, 31, 32], // vocab[7]
        vec![33, 34, 35, 36], // vocab[8]
        vec![37, 38, 39, 40], // vocab[9]
    ];

    // Create cleartext LM head (expects [vocab_size][d_model])
    let cleartext_lm_head = LMHead {
        vocab_size,
        d_model,
        weights: weights.clone(),
        quant_info: QuantInfo {
            scale: 1.0,
            abs_max: 40.0,
        },
    };

    println!("Cleartext LM head:");
    println!("  vocab_size: {}", vocab_size);
    println!("  d_model: {}", d_model);

    // Generate test hidden state (INT8)
    let hidden: Vec<i8> = vec![1, -2, 3, -1];

    println!("\nTest hidden state (i8): {:?}", hidden);

    // Cleartext LM head computation
    let cleartext_logits = lm_head_forward_i8(&hidden, &cleartext_lm_head.weights);
    println!("\nCleartext logits (i64): {:?}", cleartext_logits);
    println!(
        "Cleartext argmax: {}",
        cleartext_logits.iter().enumerate().max_by(|a, b| a.1.cmp(b.1)).unwrap().0
    );

    // Transpose weights for encrypted LM head (expects [d_model][vocab_size])
    let weights_transposed: Vec<Vec<i64>> = (0..d_model)
        .map(|i| (0..vocab_size).map(|j| weights[j][i]).collect())
        .collect();

    // Create encrypted LM head (expects [d_model][vocab_size])
    let encrypted_lm_head = EncryptedLMHead {
        weights: weights_transposed.clone(),
        vocab_size,
        d_model,
        scale: 1.0,
    };

    println!("\nEncrypted LM head:");
    println!("  vocab_size: {}", encrypted_lm_head.vocab_size);
    println!("  d_model: {}", encrypted_lm_head.d_model);

    println!("\n=== Testing Encrypted LM Head ===");

    println!("\nExpected encrypted LM head behavior:");
    println!("  For each vocab j: logits[j] = Σ_i weights[i][j] * hidden[i]");
    println!("  (using _FHE_LLM_dot_product_scaled with res_offset=5)");

    // Manually verify the computation matches cleartext
    println!("\nManual verification of weight layout:");
    for j in 0..vocab_size {
        let expected: i64 = (0..d_model).map(|i| weights_transposed[i][j] * (hidden[i] as i64)).sum();
        println!("  logits[{}]: Σ_i w[i][{}] * h[i] = {}", j, j, expected);
        assert_eq!(expected, cleartext_logits[j], "Mismatch at logits[{}]", j);
    }

    // Now run the actual encrypted computation
    println!("\n=== Running Encrypted LM Head Forward ===");

    let params = FHE_LLMParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = FHE_LLMKey::generate(&module, &params, [42u8; 32]);
    let eval_key = FHE_LLMEvalKey::generate(&module, &key, &params, [43u8; 32], [44u8; 32]);

    // Encrypt hidden state
    let hidden_cts: Vec<GLWE<Vec<u8>>> = hidden
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let pt = encode_int8(&module, &params, &[v]);
            let mut seed_a = [0u8; 32];
            let mut seed_e = [0u8; 32];
            seed_a[0] = (i as u8).wrapping_add(100);
            seed_e[0] = (i as u8).wrapping_add(200);
            _FHE_LLM_encrypt(&module, &key, &pt, seed_a, seed_e)
        })
        .collect();

    println!("Encrypted {} hidden state components", hidden_cts.len());

    // Create encrypted LM head
    let encrypted_lm_head =
        EncryptedLMHead::from_cleartext(&module, &params, &eval_key, &cleartext_lm_head, 1.0, Precision::Int8);

    // Compute encrypted logits
    let logits_cts = encrypted_lm_head.forward(&module, &eval_key, &hidden_cts);

    println!("Computed {} encrypted logits", logits_cts.len());

    // Decrypt logits
    let encrypted_logits: Vec<i8> = logits_cts
        .iter()
        .map(|ct| {
            let pt = _FHE_LLM_decrypt(&module, &key, ct, &params);
            let vals = decode_int8(&module, &params, &pt, 1);
            vals[0]
        })
        .collect();

    let encrypted_logits_i64: Vec<i64> = encrypted_logits.iter().map(|&v| v as i64).collect();

    println!("\nCleartext logits: {:?}", cleartext_logits);
    println!("Encrypted logits: {:?}", encrypted_logits_i64);

    // Compute L-inf error
    let l_inf_error: i64 = cleartext_logits
        .iter()
        .zip(encrypted_logits_i64.iter())
        .map(|(a, b)| (a - b).abs())
        .max()
        .unwrap_or(0);

    println!("\nL-inf error: {}", l_inf_error);

    // Verify argmax matches
    let cleartext_argmax = cleartext_logits.iter().enumerate().max_by(|a, b| a.1.cmp(b.1)).unwrap().0;
    let encrypted_argmax = encrypted_logits_i64.iter().enumerate().max_by(|a, b| a.1.cmp(b.1)).unwrap().0;

    println!("Cleartext argmax: {}", cleartext_argmax);
    println!("Encrypted argmax: {}", encrypted_argmax);

    assert_eq!(cleartext_argmax, encrypted_argmax, "Argmax mismatch");
    assert!(l_inf_error < 2, "L-inf error {} >= 2.0", l_inf_error);

    println!("\n=== Test Complete ===");
    println!("✅ Cleartext LM head computation verified");
    println!("✅ Weight transposition verified");
    println!("✅ Encrypted LM head forward verified");
    println!("✅ L-inf error: {} (< 2.0)", l_inf_error);
    println!("✅ Argmax match: {}", cleartext_argmax == encrypted_argmax);
}
