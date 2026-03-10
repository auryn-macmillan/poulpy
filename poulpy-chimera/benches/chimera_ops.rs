//! Benchmark harness for CHIMERA transformer operations.
//!
//! Measures latency and noise budget consumption across representative
//! transformer operations: encoding, encryption, addition, plaintext
//! multiplication, activation approximation, and planning.
//!
//! ## Benchmark groups
//!
//! - **Plaintext-domain**: encoding, decoding, polynomial eval, LUT eval,
//!   planning, LayerNorm, noise tracking.
//! - **FHE-domain**: encrypt/decrypt, add/sub, mul_const, ct*ct mul,
//!   polynomial activation, matmul (single-ct), standard FFN.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use poulpy_chimera::params::*;
use poulpy_chimera::encoding::*;
use poulpy_chimera::encrypt::*;
use poulpy_chimera::arithmetic::*;
use poulpy_chimera::activations::*;
use poulpy_chimera::layernorm::*;
use poulpy_chimera::transformer::*;
use poulpy_chimera::lut::*;
use poulpy_chimera::noise::*;

use poulpy_hal::{api::ModuleNew, layouts::Module};

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
type BE = poulpy_cpu_ref::FFT64Ref;
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
type BE = poulpy_cpu_avx::FFT64Avx;

fn bench_encoding(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let values: Vec<i8> = (0..params.slots).map(|i| (i % 256) as i8).collect();

    c.bench_function("chimera/encode_int8", |b| {
        b.iter(|| encode_int8(&module, &params, black_box(&values)))
    });

    let fp_values: Vec<f32> = (0..params.slots).map(|i| (i as f32) / 1000.0).collect();
    let fp_params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Fp16);

    c.bench_function("chimera/encode_fp16", |b| {
        b.iter(|| encode_fp16(&module, &fp_params, black_box(&fp_values)))
    });
}

fn bench_encrypt_decrypt(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
    let values: Vec<i8> = vec![1; 64];
    let pt = encode_int8(&module, &params, &values);

    c.bench_function("chimera/encrypt", |b| {
        b.iter(|| chimera_encrypt(&module, &key, black_box(&pt), [1u8; 32], [2u8; 32]))
    });

    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    c.bench_function("chimera/decrypt", |b| {
        b.iter(|| chimera_decrypt(&module, &key, black_box(&ct), &params))
    });
}

fn bench_arithmetic(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

    let pt_a = encode_int8(&module, &params, &[1i8; 64]);
    let pt_b = encode_int8(&module, &params, &[2i8; 64]);
    let ct_a = chimera_encrypt(&module, &key, &pt_a, [1u8; 32], [2u8; 32]);
    let ct_b = chimera_encrypt(&module, &key, &pt_b, [3u8; 32], [4u8; 32]);

    c.bench_function("chimera/add", |b| {
        b.iter(|| chimera_add(&module, black_box(&ct_a), black_box(&ct_b)))
    });

    c.bench_function("chimera/sub", |b| {
        b.iter(|| chimera_sub(&module, black_box(&ct_a), black_box(&ct_b)))
    });
}

fn bench_activations(c: &mut Criterion) {
    c.bench_function("chimera/gelu_poly_eval", |b| {
        let approx = gelu_poly_approx();
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..1000 {
                sum += approx.eval(black_box(i as f64 / 100.0 - 5.0));
            }
            sum
        })
    });

    c.bench_function("chimera/gelu_lut_eval", |b| {
        let lut = NonlinearLUT::gelu(8);
        b.iter(|| {
            let mut sum = 0.0;
            for i in 0..1000 {
                sum += lut.eval(black_box(i as f64 / 100.0 - 5.0));
            }
            sum
        })
    });
}

fn bench_planning(c: &mut Criterion) {
    c.bench_function("chimera/plan_dense_7b", |b| {
        let dims = ModelDims::dense_7b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = default_block_config(dims.clone(), params.clone());
        b.iter(|| plan_forward_pass(black_box(&config), black_box(dims.n_layers), black_box(&params)))
    });

    c.bench_function("chimera/plan_moe_40b", |b| {
        let dims = ModelDims::moe_40b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = default_block_config(dims.clone(), params.clone());
        b.iter(|| plan_forward_pass(black_box(&config), black_box(dims.n_layers), black_box(&params)))
    });
}

fn bench_layernorm(c: &mut Criterion) {
    c.bench_function("chimera/rmsnorm_plaintext_4096", |b| {
        let config = LayerNormConfig::rms_norm(4096);
        let values: Vec<f64> = (0..4096).map(|i| (i as f64) / 1000.0).collect();
        b.iter(|| layernorm_plaintext(black_box(&values), black_box(&config)))
    });
}

fn bench_noise_tracking(c: &mut Criterion) {
    c.bench_function("chimera/noise_estimate_layer", |b| {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        b.iter(|| estimate_transformer_layer_noise(black_box(&params), black_box(4096)))
    });
}

// ---------------------------------------------------------------------------
// FHE-domain benchmarks
// ---------------------------------------------------------------------------

/// Benchmarks ciphertext-plaintext multiplication (chimera_mul_const).
fn bench_mul_const(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

    let pt = encode_int8(&module, &params, &[3i8; 64]);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);
    let weight = vec![7i64; 1]; // single-coefficient polynomial

    c.bench_function("chimera/mul_const_scalar", |b| {
        b.iter(|| chimera_mul_const(&module, black_box(&ct), black_box(&weight)))
    });
}

/// Benchmarks ciphertext × ciphertext multiplication (tensor product + relinearization).
fn bench_ct_ct_mul(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
    let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [50u8; 32], [60u8; 32]);

    let pt = encode_int8(&module, &params, &[3i8; 64]);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    c.bench_function("chimera/ct_ct_mul", |b| {
        b.iter(|| chimera_ct_mul(&module, &eval_key, black_box(&ct), black_box(&ct)))
    });
}

/// Benchmarks polynomial activation evaluation on ciphertexts.
fn bench_poly_activation_fhe(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
    let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [50u8; 32], [60u8; 32]);

    let pt = encode_int8(&module, &params, &[3i8; 64]);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    // Squared ReLU (degree-2, 1 ct*ct mul)
    let sqrelu = squared_relu_approx();
    c.bench_function("chimera/activation_sqrelu_fhe", |b| {
        b.iter(|| apply_poly_activation(&module, &eval_key, black_box(&ct), &sqrelu))
    });

    // GELU (effective degree 2 since c3=0, same cost as SqReLU)
    let gelu = gelu_poly_approx();
    c.bench_function("chimera/activation_gelu_fhe", |b| {
        b.iter(|| apply_poly_activation(&module, &eval_key, black_box(&ct), &gelu))
    });
}

/// Benchmarks single-ciphertext matrix-vector product (ct-pt matmul).
fn bench_matmul_fhe(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

    let pt = encode_int8(&module, &params, &[3i8; 64]);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    // Small matmul: 4 output rows, each a single-coefficient scalar
    let weight_rows: Vec<Vec<i64>> = (0..4).map(|i| vec![(i + 1) as i64]).collect();

    c.bench_function("chimera/matmul_single_ct_4rows", |b| {
        b.iter(|| chimera_matmul_single_ct(&module, black_box(&ct), black_box(&weight_rows)))
    });
}

/// Benchmarks a minimal standard FFN pipeline (up-project → activation → down-project).
fn bench_ffn_standard_fhe(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
    let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [50u8; 32], [60u8; 32]);

    let pt = encode_int8(&module, &params, &[2i8; 1]);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    // Minimal FFN: d_model=1, d_ffn=2 (2 hidden units)
    let weights = FFNWeights {
        w1: vec![vec![1], vec![2]],    // 2 hidden units
        w2: vec![vec![1, 1]],          // 1 output, summing both hidden
        w3: None,
    };

    c.bench_function("chimera/ffn_standard_d1h2", |b| {
        b.iter(|| {
            chimera_ffn_standard(
                &module,
                &eval_key,
                black_box(&ct),
                black_box(&weights),
                &ActivationChoice::SquaredReLU,
            )
        })
    });
}

// ---------------------------------------------------------------------------
// Realistic-dimension benchmarks (d_model=128)
// ---------------------------------------------------------------------------

/// Benchmarks matmul at d_model=128 (128 output rows, each with 128 coefficients).
fn bench_matmul_d128(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

    let vals: Vec<i8> = (0..128).map(|i| ((i % 7) as i8) - 3).collect();
    let pt = encode_int8(&module, &params, &vals);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    // 128 output rows, each with 128 polynomial coefficients
    let weight_rows: Vec<Vec<i64>> = (0..128)
        .map(|r| (0..128).map(|c| ((r * 128 + c) % 5) as i64 - 2).collect())
        .collect();

    c.bench_function("chimera/matmul_d128", |b| {
        b.iter(|| chimera_matmul_single_ct(&module, black_box(&ct), black_box(&weight_rows)))
    });
}

/// Benchmarks mul_const with a 128-coefficient polynomial weight.
fn bench_mul_const_d128(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

    let vals: Vec<i8> = (0..128).map(|i| ((i % 5) as i8) - 2).collect();
    let pt = encode_int8(&module, &params, &vals);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    let weight: Vec<i64> = (0..128).map(|i| (i % 5) as i64 - 2).collect();

    c.bench_function("chimera/mul_const_128coeff", |b| {
        b.iter(|| chimera_mul_const(&module, black_box(&ct), black_box(&weight)))
    });
}

/// Benchmarks ct*ct multiplication at realistic dimension.
fn bench_ct_ct_mul_d128(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
    let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [50u8; 32], [60u8; 32]);

    let vals: Vec<i8> = (0..128).map(|i| ((i % 7) as i8) - 3).collect();
    let pt = encode_int8(&module, &params, &vals);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    c.bench_function("chimera/ct_ct_mul_d128", |b| {
        b.iter(|| chimera_ct_mul(&module, &eval_key, black_box(&ct), black_box(&ct)))
    });
}

/// Benchmarks a standard FFN pipeline at modest dimension (d_model=4, d_ffn=8).
/// Full d_model=128 FFN would take minutes per iteration; we measure a smaller
/// but representative configuration to derive per-op scaling factors.
fn bench_ffn_d4h8(c: &mut Criterion) {
    let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
    let module: Module<BE> = Module::new(params.n());
    let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
    let eval_key = ChimeraEvalKey::generate(&module, &key, &params, [50u8; 32], [60u8; 32]);

    let vals: Vec<i8> = vec![1, 2, 3, 4];
    let pt = encode_int8(&module, &params, &vals);
    let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

    // d_model=4, d_ffn=8: w1 has 8 rows (up-project), w2 has 4 rows (down-project)
    let w1: Vec<Vec<i64>> = (0..8).map(|_| vec![1i64]).collect();
    let w2: Vec<Vec<i64>> = (0..4)
        .map(|_| vec![1i64; 8])
        .collect();

    let weights = FFNWeights { w1, w2, w3: None };

    c.bench_function("chimera/ffn_standard_d4h8", |b| {
        b.iter(|| {
            chimera_ffn_standard(
                &module,
                &eval_key,
                black_box(&ct),
                black_box(&weights),
                &ActivationChoice::SquaredReLU,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_encoding,
    bench_encrypt_decrypt,
    bench_arithmetic,
    bench_activations,
    bench_planning,
    bench_layernorm,
    bench_noise_tracking,
    bench_mul_const,
    bench_ct_ct_mul,
    bench_poly_activation_fhe,
    bench_matmul_fhe,
    bench_ffn_standard_fhe,
    bench_matmul_d128,
    bench_mul_const_d128,
    bench_ct_ct_mul_d128,
    bench_ffn_d4h8,
);
criterion_main!(benches);
