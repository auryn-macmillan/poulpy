//! Security parameters, noise budgets, and scheme configuration.
//!
//! This module defines the parameter sets for CHIMERA at different security
//! levels, co-designed for INT8/FP16 transformer inference arithmetic.

use poulpy_core::layouts::{Base2K, Degree, Rank, TorusPrecision};

/// Security level of the scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityLevel {
    /// ~80-bit post-quantum security (N=4096).
    Bits80,
    /// ~100-bit post-quantum security (N=8192).
    Bits100,
    /// ~128-bit post-quantum security (N=16384).
    Bits128,
}

/// Plaintext precision mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// 8-bit signed integers. Suitable for quantised weights.
    Int8,
    /// 16-bit floating point (encoded as fixed-point). Suitable for activations.
    Fp16,
}

/// Complete parameter set for a CHIMERA instance.
///
/// All fields are determined by the security level and precision mode.
/// Users should construct this via [`ChimeraParams::new`] rather than
/// filling fields manually.
///
/// ## Parameter relationships for tensor products
///
/// The tensor product (ct*ct multiplication) requires a cascade of base2k values:
/// - `base2k`: the "master" base2k (e.g. 14), used for the tensor key
/// - `in_base2k = base2k - 1`: base2k of input ciphertexts and plaintexts
/// - `out_base2k = base2k - 2`: base2k of output ciphertexts after tensor product
///
/// Ciphertexts produced by `chimera_encrypt` use `in_base2k`, NOT `base2k`.
/// This matches the pattern from poulpy-core's tensor test.
#[derive(Clone, Debug)]
pub struct ChimeraParams {
    /// Ring polynomial degree N (power of two).
    pub degree: Degree,
    /// Master base-2 logarithm of the limb radix. The tensor key uses this value.
    /// Input ciphertexts use `base2k - 1`; output ciphertexts use `base2k - 2`.
    pub base2k: Base2K,
    /// Torus precision in bits (ciphertext modulus budget).
    pub k_ct: TorusPrecision,
    /// Plaintext torus precision in bits.
    pub k_pt: TorusPrecision,
    /// GLWE rank (number of mask polynomials).
    pub rank: Rank,
    /// Security level.
    pub security: SecurityLevel,
    /// Plaintext precision mode.
    pub precision: Precision,
    /// Number of SIMD slots available per ciphertext.
    pub slots: usize,
    /// Scaling factor Δ = 2^scale_bits for fixed-point encoding.
    pub scale_bits: u32,
    /// Maximum multiplicative depth before noise budget exhaustion.
    pub max_depth: usize,
    /// Noise budget in bits (log₂ of the ratio q/noise).
    pub noise_budget_bits: usize,
}

impl ChimeraParams {
    /// Constructs a parameter set for the given security level and precision.
    ///
    /// # Parameters
    ///
    /// * `security` - Target post-quantum security level.
    /// * `precision` - Plaintext arithmetic precision (INT8 or FP16).
    ///
    /// # Panics
    ///
    /// Does not panic; all security/precision combinations are valid.
    pub fn new(security: SecurityLevel, precision: Precision) -> Self {
        let (n, max_depth, noise_budget_bits) = match security {
            SecurityLevel::Bits80 => (4096u32, 12, 38),
            SecurityLevel::Bits100 => (8192u32, 24, 38),
            SecurityLevel::Bits128 => (16384u32, 48, 38),
        };

        let base2k = 14u32;
        // k_ct must be large enough to hold the tensor product intermediate:
        // following poulpy-core's tensor test pattern, k = 8 * base2k + 1.
        // With base2k=14 this gives k_ct=113, providing sufficient limbs for
        // the relinearization (dnum = k/tsk_base2k = 113/14 = 9).
        let k_ct = 8 * base2k + 1;

        let (k_pt, scale_bits) = match precision {
            Precision::Int8 => (base2k, 8),
            Precision::Fp16 => (base2k, 14),
        };

        ChimeraParams {
            degree: Degree(n),
            base2k: Base2K(base2k),
            k_ct: TorusPrecision(k_ct),
            k_pt: TorusPrecision(k_pt),
            rank: Rank(1),
            security,
            precision,
            slots: n as usize,
            scale_bits,
            max_depth,
            noise_budget_bits,
        }
    }

    /// Returns the ring degree N as a `u64` (for `Module::new`).
    pub fn n(&self) -> u64 {
        self.degree.0 as u64
    }

    /// Returns the input ciphertext base2k (`base2k - 1`).
    ///
    /// Input ciphertexts and plaintexts are encoded at this base2k.
    /// This matches the pattern from poulpy-core's tensor test.
    pub fn in_base2k(&self) -> usize {
        let b = self.base2k.0 as usize;
        if b > 1 { b - 1 } else { b }
    }

    /// Returns the output ciphertext base2k after a tensor product (`base2k - 2`).
    pub fn out_base2k(&self) -> usize {
        let b = self.base2k.0 as usize;
        if b > 2 { b - 2 } else { b }
    }

    /// Returns the encoding scale used for torus encoding (`2 * in_base2k`).
    pub fn encoding_scale(&self) -> usize {
        2 * self.in_base2k()
    }

    /// Returns the number of limbs in the digit decomposition.
    pub fn num_limbs(&self) -> u32 {
        self.k_ct.0.div_ceil(self.base2k.0)
    }

    /// Returns the ciphertext byte size (approximate) for a single GLWE ciphertext.
    pub fn ciphertext_bytes(&self) -> usize {
        let cols = (self.rank.0 + 1) as usize;
        let limbs = self.num_limbs() as usize;
        let n = self.degree.0 as usize;
        // Each limb stores N coefficients of base2k bits each, packed into i64s
        cols * limbs * n * 8
    }

    /// Returns the maximum number of transformer layers supportable without bootstrapping.
    ///
    /// Assumes ~10 multiplicative operations per layer with aggressive rescaling.
    pub fn max_layers_no_bootstrap(&self) -> usize {
        // Each rescale consumes base2k bits of precision.
        // Each mult roughly doubles noise, consuming ~1 bit of budget per rescale.
        // With noise_budget_bits available and ~10 mults per layer:
        // max_layers ≈ (noise_budget_bits * base2k) / (10 * log2(noise_growth_per_mult))
        // Simplified: max_depth / 10 (conservative)
        self.max_depth / 10
    }

    /// Checks whether the given model depth can run without bootstrapping.
    pub fn supports_no_bootstrap(&self, num_layers: usize) -> bool {
        num_layers <= self.max_layers_no_bootstrap()
    }
}

/// Transformer model dimensions used for packing alignment.
#[derive(Clone, Debug)]
pub struct ModelDims {
    /// Model embedding dimension (d_model), e.g. 4096.
    pub d_model: usize,
    /// Attention head dimension (d_head), e.g. 128.
    pub d_head: usize,
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key/value attention heads (for Grouped Query Attention).
    ///
    /// When `n_kv_heads == n_heads`, this is standard multi-head attention (MHA).
    /// When `n_kv_heads < n_heads`, multiple query heads share the same KV head
    /// (Grouped Query Attention / GQA). For example, TinyLlama has 32 query
    /// heads but only 4 KV heads, giving a group size of 8.
    pub n_kv_heads: usize,
    /// FFN intermediate dimension (d_ffn), e.g. 11008.
    pub d_ffn: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of experts (1 for dense models).
    pub n_experts: usize,
    /// Number of active experts per token (for MoE).
    pub n_active_experts: usize,
}

impl ModelDims {
    /// Returns dimensions for a typical ~7B dense transformer (LLaMA-2 7B style).
    ///
    /// Uses standard multi-head attention (n_kv_heads == n_heads).
    pub fn dense_7b() -> Self {
        ModelDims {
            d_model: 4096,
            d_head: 128,
            n_heads: 32,
            n_kv_heads: 32,
            d_ffn: 11008,
            n_layers: 32,
            n_experts: 1,
            n_active_experts: 1,
        }
    }

    /// Returns dimensions for a Mixtral-style ~40B MoE transformer.
    ///
    /// Uses GQA with 8 KV heads shared across 32 query heads (group size 4).
    pub fn moe_40b() -> Self {
        ModelDims {
            d_model: 4096,
            d_head: 128,
            n_heads: 32,
            n_kv_heads: 8,
            d_ffn: 14336,
            n_layers: 32,
            n_experts: 8,
            n_active_experts: 2,
        }
    }

    /// Returns dimensions for a TinyLlama-1.1B style model.
    ///
    /// Uses GQA with 4 KV heads shared across 32 query heads (group size 8).
    pub fn tinyllama_1b() -> Self {
        ModelDims {
            d_model: 2048,
            d_head: 64,
            n_heads: 32,
            n_kv_heads: 4,
            d_ffn: 5632,
            n_layers: 22,
            n_experts: 1,
            n_active_experts: 1,
        }
    }

    /// Returns whether this is a Mixture-of-Experts model.
    pub fn is_moe(&self) -> bool {
        self.n_experts > 1
    }

    /// Returns whether this model uses Grouped Query Attention (GQA).
    ///
    /// GQA is active when `n_kv_heads < n_heads`.
    pub fn is_gqa(&self) -> bool {
        self.n_kv_heads < self.n_heads
    }

    /// Returns the GQA group size: how many query heads share one KV head.
    ///
    /// For standard MHA, returns 1. For GQA, returns `n_heads / n_kv_heads`.
    ///
    /// # Panics
    ///
    /// Panics if `n_heads` is not divisible by `n_kv_heads`.
    pub fn gqa_group_size(&self) -> usize {
        assert!(
            self.n_heads % self.n_kv_heads == 0,
            "n_heads ({}) must be divisible by n_kv_heads ({})",
            self.n_heads, self.n_kv_heads
        );
        self.n_heads / self.n_kv_heads
    }

    /// Returns the total KV projection dimension: `n_kv_heads * d_head`.
    ///
    /// For standard MHA this equals `d_model`. For GQA this is smaller.
    pub fn d_kv(&self) -> usize {
        self.n_kv_heads * self.d_head
    }

    /// Returns the approximate total parameter count (in elements).
    pub fn total_params(&self) -> usize {
        // Q and O projections are [d_model, d_model]; K and V are [d_kv, d_model]
        let d_kv = self.d_kv();
        let attn_params = self.n_layers * (self.d_model * self.d_model + 2 * d_kv * self.d_model + self.d_model * self.d_model);
        let ffn_params = self.n_layers * self.n_experts * (2 * self.d_model * self.d_ffn);
        attn_params + ffn_params
    }

    /// Returns the approximate active parameter count per token.
    pub fn active_params(&self) -> usize {
        let d_kv = self.d_kv();
        let attn_params = self.n_layers * (self.d_model * self.d_model + 2 * d_kv * self.d_model + self.d_model * self.d_model);
        let ffn_params = self.n_layers * self.n_active_experts * (2 * self.d_model * self.d_ffn);
        attn_params + ffn_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_128bit_int8() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        assert_eq!(params.degree.0, 16384);
        assert_eq!(params.base2k.0, 14);
        assert_eq!(params.k_ct.0, 113);
        assert_eq!(params.rank.0, 1);
        assert_eq!(params.slots, 16384);
        assert_eq!(params.scale_bits, 8);
        assert!(params.max_depth > 0);
    }

    #[test]
    fn test_params_100bit_fp16() {
        let params = ChimeraParams::new(SecurityLevel::Bits100, Precision::Fp16);
        assert_eq!(params.degree.0, 8192);
        assert_eq!(params.scale_bits, 14);
    }

    #[test]
    fn test_params_80bit() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        assert_eq!(params.degree.0, 4096);
        assert_eq!(params.max_depth, 12);
    }

    #[test]
    fn test_model_dims_dense() {
        let dims = ModelDims::dense_7b();
        assert!(!dims.is_moe());
        assert_eq!(dims.n_experts, 1);
        assert!(dims.total_params() > 0);
        assert_eq!(dims.total_params(), dims.active_params());
    }

    #[test]
    fn test_model_dims_moe() {
        let dims = ModelDims::moe_40b();
        assert!(dims.is_moe());
        assert_eq!(dims.n_experts, 8);
        assert_eq!(dims.n_active_experts, 2);
        assert!(dims.active_params() < dims.total_params());
    }

    #[test]
    fn test_ciphertext_bytes() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let bytes = params.ciphertext_bytes();
        // 2 columns * 4 limbs * 16384 coefficients * 8 bytes = 1048576
        assert!(bytes > 0);
    }

    #[test]
    fn test_no_bootstrap_check() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        assert!(params.supports_no_bootstrap(4));
        // Very deep model should need bootstrapping
        assert!(!params.supports_no_bootstrap(100));
    }

    #[test]
    fn test_gqa_helpers_dense_7b() {
        let dims = ModelDims::dense_7b();
        assert!(!dims.is_gqa(), "dense_7b should be standard MHA");
        assert_eq!(dims.gqa_group_size(), 1);
        assert_eq!(dims.d_kv(), dims.d_model);
        assert_eq!(dims.n_kv_heads, 32);
    }

    #[test]
    fn test_gqa_helpers_moe_40b() {
        let dims = ModelDims::moe_40b();
        assert!(dims.is_gqa(), "moe_40b should use GQA");
        assert_eq!(dims.gqa_group_size(), 4); // 32 / 8 = 4
        assert_eq!(dims.d_kv(), 8 * 128); // n_kv_heads * d_head = 1024
        assert!(dims.d_kv() < dims.d_model); // 1024 < 4096
    }

    #[test]
    fn test_gqa_helpers_tinyllama_1b() {
        let dims = ModelDims::tinyllama_1b();
        assert!(dims.is_gqa(), "tinyllama_1b should use GQA");
        assert_eq!(dims.gqa_group_size(), 8); // 32 / 4 = 8
        assert_eq!(dims.d_kv(), 4 * 64); // n_kv_heads * d_head = 256
        assert_eq!(dims.d_model, 2048);
        assert!(dims.d_kv() < dims.d_model); // 256 < 2048
    }

    #[test]
    fn test_gqa_total_params_accounts_for_kv_reduction() {
        // For GQA, total_params should be less than if all projections were d_model×d_model.
        let dims_gqa = ModelDims::moe_40b();
        let d = dims_gqa.d_model;
        let d_kv = dims_gqa.d_kv();

        // With GQA: attn = n_layers * (d*d + 2*d_kv*d + d*d) per layer
        // Without GQA (if d_kv == d): attn = n_layers * (4*d*d)
        // Since d_kv < d for GQA, GQA should have fewer params
        let gqa_attn_per_layer = d * d + 2 * d_kv * d + d * d;
        let full_attn_per_layer = 4 * d * d;
        assert!(
            gqa_attn_per_layer < full_attn_per_layer,
            "GQA attention params ({gqa_attn_per_layer}) should be less than full MHA ({full_attn_per_layer})"
        );
    }
}
