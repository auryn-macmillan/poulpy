//! # poulpy-chimera
//!
//! **CHIMERA**: Ciphertext Homomorphic Inference with Minimised Encryption for Robust AI
//!
//! An inference-optimised FHE scheme for transformer neural networks, built on top
//! of Poulpy's hardware abstraction layer and cryptographic core.
//!
//! ## Design
//!
//! CHIMERA is an RLWE-based approximate-arithmetic FHE scheme purpose-built for
//! quantised transformer inference. It differs from general-purpose CKKS in three
//! key ways:
//!
//! 1. **Native low-precision fields.** Torus precision and noise parameters are
//!    co-designed for INT8 weights and FP16 activations (14-27 bit coefficients
//!    vs. CKKS's 50-60 bits).
//!
//! 2. **Transformer-aligned SIMD packing.** Ciphertext slot counts are matched to
//!    attention head dimensions and embedding widths, maximising utilisation of
//!    packed operations across a forward pass.
//!
//! 3. **Co-designed nonlinearities.** Polynomial activation approximations and
//!    LUT-based evaluation are first-class operations with pre-computed
//!    coefficients for GELU, softmax, and LayerNorm.
//!
//! ## Module Overview
//!
//! | Module         | Responsibility                                          |
//! |----------------|---------------------------------------------------------|
//! | [`params`]     | Security levels, noise budgets, and parameter selection  |
//! | [`encoding`]   | INT8/FP16 plaintext encoding with SIMD slot packing      |
//! | [`encrypt`]    | RLWE encryption/decryption via `poulpy-core`             |
//! | [`arithmetic`] | Homomorphic add, plaintext multiply, rescale, rotate     |
//! | [`activations`]| Polynomial GELU/softmax approximations                   |
//! | [`lut`]        | LUT-based nonlinearity evaluation via blind rotation     |
//! | [`layernorm`]  | Approximate LayerNorm / RMSNorm under FHE                |
//! | [`attention`]  | Transformer attention mechanism                          |
//! | [`transformer`]| Full transformer block and forward pass                  |
//! | [`moe`]        | Mixture-of-Experts routing under FHE                     |
//! | [`noise`]      | Noise tracking and budget estimation                     |
//! | [`verification`]| MAC-based user-side computation verification             |
//!
//! ## Backend Selection
//!
//! Like all Poulpy crates, CHIMERA is generic over the compute backend.
//! Use `FFT64Ref` for portable execution or `FFT64Avx` for AVX2-accelerated
//! computation on x86-64.
//!
//! ## Scratch-Space Allocation
//!
//! Core backend operations use scratch-space arenas, but higher-level research
//! helpers and tests may still allocate owned buffers.

pub mod activations;
pub mod arithmetic;
pub mod attention;
pub mod bootstrapping;
pub mod encoding;
pub mod encrypt;
pub mod layernorm;
pub mod lut;
pub mod model_loader;
pub mod moe;
pub mod noise;
pub mod params;
pub mod transformer;
pub mod verification;

#[cfg(test)]
mod tests;
