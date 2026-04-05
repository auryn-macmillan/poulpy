//! Plaintext encoding for INT8 and FP16 values.
//!
//! Provides encoding of scalar vectors into polynomial ring elements suitable
//! for GLWE encryption, with SIMD-style slot packing aligned to transformer
//! model dimensions (attention heads, embedding width, expert blocks).

use poulpy_core::layouts::GLWEPlaintext;
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module},
};

use crate::params::FHE_LLMParams;

/// Encodes a vector of `i8` values into a GLWE plaintext at the full
/// encoding scale (`TorusPrecision(encoding_scale)` = `2 * in_base2k`).
///
/// This encoding is compatible with both linear operations (`glwe_mul_const`)
/// and nonlinear operations (`glwe_tensor_apply` with `res_offset = encoding_scale`).
/// The tensor product preserves the torus position when both inputs are encoded
/// at `TorusPrecision(encoding_scale)`, making ct×ct multiplication exact at
/// INT8 precision (zero noise from the encoding).
///
/// The first `values.len()` polynomial coefficients are set; remaining
/// coefficients are zero.
///
/// # Panics
///
/// Panics if `values.len() > params.slots`.
pub fn encode_int8<BE: Backend>(module: &Module<BE>, params: &FHE_LLMParams, values: &[i8]) -> GLWEPlaintext<Vec<u8>>
where
    Module<BE>: ModuleN,
{
    assert!(
        values.len() <= params.slots,
        "encode_int8: values.len()={} > slots={}",
        values.len(),
        params.slots
    );

    let layout = poulpy_core::layouts::GLWEPlaintextLayout {
        n: params.degree,
        base2k: poulpy_core::layouts::Base2K(params.in_base2k() as u32),
        k: params.k_pt,
    };

    let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&layout);

    // Encode at TorusPrecision(encoding_scale) where encoding_scale = 2 * in_base2k.
    // This places values across two limbs of the base2k representation, spanning
    // the full encoding precision. This is critical for compatibility with the
    // tensor product (ct×ct multiplication): with res_offset = encoding_scale,
    // the product of two values at TorusPrecision(encoding_scale) lands at the
    // same torus position, preserving the encoding through nonlinear operations.
    //
    // Previously, values were placed only in limb 0 (at in_base2k position),
    // which caused the tensor product to produce zero output — the product of
    // two values at position 2^{-13} is at 2^{-26}, and res_offset=26 shifts
    // it to torus position 0 (complete signal loss).
    let n = module.n() as usize;
    let scale = params.encoding_scale(); // 2 * in_base2k = 26

    // Build the i64 data array (N elements, most zeroed)
    let mut data = vec![0i64; n];
    for (i, &v) in values.iter().enumerate() {
        data[i] = v as i64;
    }

    pt.encode_vec_i64(&data, poulpy_core::layouts::TorusPrecision(scale as u32));

    pt
}

/// Encodes a vector of `f32` values (representing FP16-range) into a GLWE plaintext.
///
/// Values are quantised to fixed-point with `params.scale_bits` fractional bits,
/// then encoded as torus elements. The encoding range is approximately
/// [-2^(15 - scale_bits), 2^(15 - scale_bits)].
///
/// # Panics
///
/// Panics if `values.len() > params.slots`.
pub fn encode_fp16<BE: Backend>(module: &Module<BE>, params: &FHE_LLMParams, values: &[f32]) -> GLWEPlaintext<Vec<u8>>
where
    Module<BE>: ModuleN,
{
    assert!(
        values.len() <= params.slots,
        "encode_fp16: values.len()={} > slots={}",
        values.len(),
        params.slots
    );

    let layout = poulpy_core::layouts::GLWEPlaintextLayout {
        n: params.degree,
        base2k: poulpy_core::layouts::Base2K(params.in_base2k() as u32),
        k: params.k_pt,
    };

    let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&layout);

    let n = module.n() as usize;
    // For FP16, quantise to fixed-point within the limb's range.
    // The limb stores values in [-2^(in_base2k-1), 2^(in_base2k-1)).
    //
    // Use f64 for the scale computation to match decode_fp16 and avoid
    // asymmetric rounding errors for large values.
    let scale = (1u64 << (params.scale_bits - 1)) as f64;

    let raw: &mut [u8] = pt.data.data.as_mut();
    let coeffs: &mut [i64] = bytemuck::cast_slice_mut(&mut raw[..n * 8]);

    for (i, &v) in values.iter().enumerate() {
        let quantised = (v as f64 * scale).round() as i64;
        coeffs[i] = quantised;
    }

    pt
}

/// Decodes a GLWE plaintext back to `i8` values.
///
/// Extracts `count` coefficient values, reversing the encoding applied by
/// [`encode_int8`]. Decodes at `TorusPrecision(encoding_scale)` to match
/// the encoding position, then truncates to `i8`.
///
/// # Panics
///
/// Panics if `count > params.slots`.
pub fn decode_int8<BE: Backend>(module: &Module<BE>, params: &FHE_LLMParams, pt: &GLWEPlaintext<Vec<u8>>, count: usize) -> Vec<i8>
where
    Module<BE>: ModuleN,
{
    assert!(count <= params.slots);

    let n = module.n() as usize;
    let scale = params.encoding_scale(); // 2 * in_base2k = 26

    let mut decoded = vec![0i64; n];
    pt.decode_vec_i64(&mut decoded, poulpy_core::layouts::TorusPrecision(scale as u32));

    decoded[..count].iter().map(|&v| v as i8).collect()
}

/// Decodes a GLWE plaintext back to `f32` values.
///
/// Extracts `count` coefficient values, reversing the encoding applied by
/// [`encode_fp16`].
///
/// # Panics
///
/// Panics if `count > params.slots`.
pub fn decode_fp16<BE: Backend>(
    module: &Module<BE>,
    params: &FHE_LLMParams,
    pt: &GLWEPlaintext<Vec<u8>>,
    count: usize,
) -> Vec<f32>
where
    Module<BE>: ModuleN,
{
    assert!(count <= params.slots);

    let n = module.n() as usize;
    let scale = (1u64 << (params.scale_bits - 1)) as f64;

    let raw: &[u8] = pt.data.data.as_ref();
    let coeffs: &[i64] = bytemuck::cast_slice(&raw[..n * 8]);

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let v = (coeffs[i] as f64) / scale;
        result.push(v as f32);
    }
    result
}

/// Packing mode describing how values map to ciphertext slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackingMode {
    /// Pack d_head values per ciphertext (for attention).
    HeadAligned {
        /// Attention head dimension (d_head), e.g. 128.
        d_head: usize,
    },
    /// Pack d_model values per ciphertext (for FFN).
    EmbeddingAligned {
        /// Model embedding dimension (d_model), e.g. 4096.
        d_model: usize,
    },
    /// Pack d_expert values per ciphertext (for MoE expert FFN).
    ExpertAligned {
        /// Expert FFN dimension, e.g. d_ffn / n_experts.
        d_expert: usize,
    },
    /// Pack arbitrary count values.
    Custom(usize),
}

impl PackingMode {
    /// Returns the number of active slots for this packing mode.
    ///
    /// The result is clamped to `params.slots` (the ring degree N) since
    /// a single ciphertext cannot hold more values than the polynomial degree.
    pub fn active_slots(&self, params: &FHE_LLMParams) -> usize {
        let raw = match self {
            PackingMode::HeadAligned { d_head } => *d_head,
            PackingMode::EmbeddingAligned { d_model } => *d_model,
            PackingMode::ExpertAligned { d_expert } => *d_expert,
            PackingMode::Custom(n) => *n,
        };
        raw.min(params.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{FHE_LLMParams, Precision, SecurityLevel};
    use poulpy_hal::api::ModuleNew;

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    #[test]
    fn test_int8_roundtrip() {
        let params = FHE_LLMParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());

        let values: Vec<i8> = vec![0, 1, -1, 42, -42, 127, -128, 0];
        let pt = encode_int8(&module, &params, &values);
        let decoded = decode_int8(&module, &params, &pt, values.len());

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_fp16_roundtrip() {
        let params = FHE_LLMParams::new(SecurityLevel::Bits80, Precision::Fp16);
        let module: Module<BE> = Module::new(params.n());

        let values: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.001];
        let pt = encode_fp16(&module, &params, &values);
        let decoded = decode_fp16(&module, &params, &pt, values.len());

        for (v, d) in values.iter().zip(decoded.iter()) {
            let err = (v - d).abs();
            // Allow quantisation error up to 2^(-scale_bits)
            assert!(err < 0.01, "fp16 roundtrip error too large: {v} -> {d}, err={err}");
        }
    }

    #[test]
    fn test_packing_mode() {
        let params = FHE_LLMParams::new(SecurityLevel::Bits80, Precision::Int8);
        assert_eq!(PackingMode::Custom(64).active_slots(&params), 64);
        assert_eq!(PackingMode::HeadAligned { d_head: 128 }.active_slots(&params), 128,);
        assert_eq!(PackingMode::EmbeddingAligned { d_model: 4096 }.active_slots(&params), 4096,);
        // ExpertAligned clamped to slots when dimension exceeds N
        assert_eq!(
            PackingMode::ExpertAligned { d_expert: 99999 }.active_slots(&params),
            params.slots,
        );
    }
}
