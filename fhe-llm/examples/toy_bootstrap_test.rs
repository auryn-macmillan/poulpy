//! Toy bootstrap test with known values.

use fhe_llm::bootstrapping::_FHE_LLM_bootstrap;
use fhe_llm::encrypt::_FHE_LLM_decrypt;
use fhe_llm::params::{FHE_LLMParams, Precision, SecurityLevel};
use poulpy_core::layouts::TorusPrecision;
use poulpy_core::plaintext::Plaintext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Toy Bootstrap Test ===\n");

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    let module = BE::default();
    let params = FHE_LLMParams::new(SecurityLevel::Bits80, Precision::Int8);

    println!(
        "Params: N={}, scale_bits={}, log_message_modulus={}",
        params.n(),
        params.scale_bits,
        params.scale_bits - 1
    );

    let pipeline = fhe_llm::inference::InferencePipeline::toy(&params);
    let key = &pipeline.key;
    let eval_key = &pipeline.eval_key;

    println!("Keys generated\n");

    let test_values = vec![0.0f64, 10.0, 50.0, 100.0, -10.0, -50.0, -100.0];
    let seed_a = [0u8; 32];
    let seed_e = [1u8; 32];

    println!("=== Test 1: Baseline (encrypt → decrypt) ===");
    for &v in &test_values {
        let pt = Plaintext::from_f64(v);
        let ct = fhe_llm::encrypt::_FHE_LLM_encrypt(&module, key, &pt, seed_a, seed_e);
        let ct_decrypted = _FHE_LLM_decrypt(&module, key, &ct, &params);
        let decoded = ct_decrypted.decode_coeff_i64(TorusPrecision(params.scale_bits), 0);
        let decoded_f64 = decoded as f64;
        let error = (v - decoded_f64).abs();
        println!("  Original: {:8.2} → Decoded: {:8.2} → Error: {:.6}", v, decoded_f64, error);
    }
    println!();

    println!("=== Test 2: Bootstrap (128 levels, default) ===");
    for &v in &test_values {
        let pt = Plaintext::from_f64(v);
        let ct = fhe_llm::encrypt::_FHE_LLM_encrypt(&module, key, &pt, seed_a, seed_e);
        let mut tracker = fhe_llm::noise::NoiseTracker::fresh();
        let ct_bootstrap = _FHE_LLM_bootstrap(&module, &ct, &mut tracker, &params, eval_key);
        let ct_decrypted = _FHE_LLM_decrypt(&module, key, &ct_bootstrap, &params);
        let decoded = ct_decrypted.decode_coeff_i64(TorusPrecision(params.scale_bits), 0);
        let decoded_f64 = decoded as f64;
        let error = (v - decoded_f64).abs();
        println!("  Original: {:8.2} → Decoded: {:8.2} → Error: {:.6}", v, decoded_f64, error);
    }
    println!();

    println!();

    println!("=== Test 2: Bootstrap (128 levels) ===");
    let log_message_modulus: usize = params.scale_bits as usize - 1;
    let identity_lut = NonlinearLUT::identity_message_lut(log_message_modulus);
    for &v in &test_values {
        let pt = Plaintext::from_f64(v);
        let ct = fhe_llm::encrypt::_FHE_LLM_encrypt(&module, key, &pt, seed_a, seed_e);
        let mut tracker = NoiseTracker::fresh();
        let ct_bootstrap = _FHE_LLM_bootstrap_with_lut(&module, &ct, &mut tracker, &bsk_prepared, &identity_lut.entries);
        let ct_decrypted = _FHE_LLM_decrypt(&module, key, &ct_bootstrap, &params);
        let decoded = ct_decrypted.decode_coeff_i64(TorusPrecision(params.scale_bits), 0);
        let decoded_f64 = decoded as f64;
        let error = (v - decoded_f64).abs();
        println!("  Original: {:8.2} → Decoded: {:8.2} → Error: {:.6}", v, decoded_f64, error);
    }
    println!();

    println!("=== Test 3: Bootstrap (1024 levels) ===");
    let high_precision_lut = NonlinearLUT::identity_message_lut(10);
    for &v in &test_values {
        let pt = Plaintext::from_f64(v);
        let ct = fhe_llm::encrypt::_FHE_LLM_encrypt(&module, key, &pt, seed_a, seed_e);
        let mut tracker = NoiseTracker::fresh();
        let ct_bootstrap = _FHE_LLM_bootstrap_with_lut(&module, &ct, &mut tracker, &bsk_prepared, &high_precision_lut.entries);
        let ct_decrypted = _FHE_LLM_decrypt(&module, key, &ct_bootstrap, &params);
        let decoded = ct_decrypted.decode_coeff_i64(TorusPrecision(params.scale_bits), 0);
        let decoded_f64 = decoded as f64;
        let error = (v - decoded_f64).abs();
        println!("  Original: {:8.2} → Decoded: {:8.2} → Error: {:.6}", v, decoded_f64, error);
    }
    println!();

    println!("=== Test 4: Bootstrap (2048 levels) ===");
    let very_high_precision_lut = NonlinearLUT::identity_message_lut(11);
    for &v in &test_values {
        let pt = Plaintext::from_f64(v);
        let ct = fhe_llm::encrypt::_FHE_LLM_encrypt(&module, key, &pt, seed_a, seed_e);
        let mut tracker = NoiseTracker::fresh();
        let ct_bootstrap =
            _FHE_LLM_bootstrap_with_lut(&module, &ct, &mut tracker, &bsk_prepared, &very_high_precision_lut.entries);
        let ct_decrypted = _FHE_LLM_decrypt(&module, key, &ct_bootstrap, &params);
        let decoded = ct_decrypted.decode_coeff_i64(TorusPrecision(params.scale_bits), 0);
        let decoded_f64 = decoded as f64;
        let error = (v - decoded_f64).abs();
        println!("  Original: {:8.2} → Decoded: {:8.2} → Error: {:.6}", v, decoded_f64, error);
    }
    println!();
    Ok(())
}
