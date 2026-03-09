//! Benchmark harness for CHIMERA transformer operations.
//!
//! Measures latency and noise budget consumption across representative
//! transformer operations: encoding, encryption, addition, plaintext
//! multiplication, activation approximation, and planning.

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

criterion_group!(
    benches,
    bench_encoding,
    bench_encrypt_decrypt,
    bench_arithmetic,
    bench_activations,
    bench_planning,
    bench_layernorm,
    bench_noise_tracking,
);
criterion_main!(benches);
