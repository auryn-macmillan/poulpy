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

use crate::params::ChimeraParams;

/// Encodes a vector of `i8` values into a GLWE plaintext.
///
/// Each value is scaled to the upper bits of the torus by left-shifting
/// by `(k_pt - 8)` bits (i.e. multiplied by Δ = 2^(k_pt * base2k - 8)).
/// The first `values.len()` polynomial coefficients are set; remaining
/// coefficients are zero.
///
/// # Panics
///
/// Panics if `values.len() > params.slots`.
pub fn encode_int8<BE: Backend>(
    module: &Module<BE>,
    params: &ChimeraParams,
    values: &[i8],
) -> GLWEPlaintext<Vec<u8>>
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
        base2k: params.base2k,
        k: params.k_pt,
    };

    let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&layout);

    // Write scaled values into the first limb of each coefficient.
    // In Poulpy's bivariate representation, each limb stores coefficients
    // in the range [-2^(base2k-1), 2^(base2k-1)). For INT8 values (8-bit),
    // we shift by (base2k - scale_bits) to place them in the upper bits
    // of the limb's digit range.
    let n = module.n() as usize;
    let shift = params.base2k.0 as usize - params.scale_bits as usize;

    let raw: &mut [u8] = pt.data.data.as_mut();
    let coeffs: &mut [i64] = bytemuck::cast_slice_mut(&mut raw[..n * 8]);

    for (i, &v) in values.iter().enumerate() {
        coeffs[i] = (v as i64) << shift;
    }

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
pub fn encode_fp16<BE: Backend>(
    module: &Module<BE>,
    params: &ChimeraParams,
    values: &[f32],
) -> GLWEPlaintext<Vec<u8>>
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
        base2k: params.base2k,
        k: params.k_pt,
    };

    let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&layout);

    let n = module.n() as usize;
    // For FP16, quantise to fixed-point within the limb's range.
    // The limb stores values in [-2^(base2k-1), 2^(base2k-1)).
    // scale_bits = base2k for FP16, so the quantised value maps
    // directly into the limb's range (shift = 0).
    let scale = (1u64 << (params.scale_bits - 1)) as f32;

    let raw: &mut [u8] = pt.data.data.as_mut();
    let coeffs: &mut [i64] = bytemuck::cast_slice_mut(&mut raw[..n * 8]);

    for (i, &v) in values.iter().enumerate() {
        let quantised = (v * scale).round() as i64;
        coeffs[i] = quantised;
    }

    pt
}

/// Decodes a GLWE plaintext back to `i8` values.
///
/// Extracts `count` coefficient values, reversing the encoding applied by
/// [`encode_int8`].
///
/// # Panics
///
/// Panics if `count > params.slots`.
pub fn decode_int8<BE: Backend>(
    module: &Module<BE>,
    params: &ChimeraParams,
    pt: &GLWEPlaintext<Vec<u8>>,
    count: usize,
) -> Vec<i8>
where
    Module<BE>: ModuleN,
{
    assert!(count <= params.slots);

    let n = module.n() as usize;
    let shift = params.base2k.0 as usize - params.scale_bits as usize;

    let raw: &[u8] = pt.data.data.as_ref();
    let coeffs: &[i64] = bytemuck::cast_slice(&raw[..n * 8]);

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        // Arithmetic right-shift to recover the signed value.
        let v = coeffs[i] >> shift;
        result.push(v as i8);
    }
    result
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
    params: &ChimeraParams,
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
    HeadAligned,
    /// Pack d_model values per ciphertext (for FFN).
    EmbeddingAligned,
    /// Pack d_expert values per ciphertext (for MoE expert FFN).
    ExpertAligned,
    /// Pack arbitrary count values.
    Custom(usize),
}

impl PackingMode {
    /// Returns the number of active slots for this packing mode.
    pub fn active_slots(&self, params: &ChimeraParams) -> usize {
        match self {
            PackingMode::Custom(n) => (*n).min(params.slots),
            // For the other modes, we use the full slot count.
            // In practice, the caller should determine the actual count
            // from ModelDims and pass it via Custom.
            _ => params.slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{ChimeraParams, Precision, SecurityLevel};
    use poulpy_hal::api::ModuleNew;

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    #[test]
    fn test_int8_roundtrip() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());

        let values: Vec<i8> = vec![0, 1, -1, 42, -42, 127, -128, 0];
        let pt = encode_int8(&module, &params, &values);
        let decoded = decode_int8(&module, &params, &pt, values.len());

        assert_eq!(values, decoded);
    }

    #[test]
    fn test_fp16_roundtrip() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Fp16);
        let module: Module<BE> = Module::new(params.n());

        let values: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.001];
        let pt = encode_fp16(&module, &params, &values);
        let decoded = decode_fp16(&module, &params, &pt, values.len());

        for (v, d) in values.iter().zip(decoded.iter()) {
            let err = (v - d).abs();
            // Allow quantisation error up to 2^(-scale_bits)
            assert!(
                err < 0.01,
                "fp16 roundtrip error too large: {v} -> {d}, err={err}"
            );
        }
    }

    #[test]
    fn test_packing_mode() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        assert_eq!(PackingMode::Custom(64).active_slots(&params), 64);
        assert_eq!(
            PackingMode::HeadAligned.active_slots(&params),
            params.slots
        );
    }
}
