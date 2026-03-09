//! Integration tests for the CHIMERA scheme.

#[cfg(test)]
mod integration {
    use crate::activations::*;
    use crate::attention::*;
    use crate::encoding::*;
    use crate::encrypt::*;
    use crate::layernorm::*;
    use crate::lut::*;
    use crate::moe::*;
    use crate::noise::*;
    use crate::params::*;
    use crate::transformer::*;

    #[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
    type BE = poulpy_cpu_ref::FFT64Ref;
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    type BE = poulpy_cpu_avx::FFT64Avx;

    use poulpy_hal::{api::ModuleNew, layouts::Module};

    // ---- Parameter tests ----

    #[test]
    fn test_all_security_levels() {
        for security in [SecurityLevel::Bits80, SecurityLevel::Bits100, SecurityLevel::Bits128] {
            for precision in [Precision::Int8, Precision::Fp16] {
                let params = ChimeraParams::new(security, precision);
                assert!(params.slots > 0);
                assert!(params.max_depth > 0);
                assert!(params.noise_budget_bits > 0);
                assert!(params.ciphertext_bytes() > 0);
            }
        }
    }

    // ---- Encoding roundtrip tests ----

    #[test]
    fn test_int8_encoding_full_range() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());

        // Test full i8 range
        let values: Vec<i8> = (-128..=127).map(|x| x as i8).collect();
        let pt = encode_int8(&module, &params, &values);
        let decoded = decode_int8(&module, &params, &pt, values.len());

        for (i, (&v, &d)) in values.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(v, d, "int8 mismatch at index {i}: {v} != {d}");
        }
    }

    // ---- Encrypt/decrypt tests ----

    #[test]
    fn test_encrypt_decrypt_zeros() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [0u8; 32]);

        let values: Vec<i8> = vec![0; 64];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);
        let pt_dec = chimera_decrypt(&module, &key, &ct, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, values.len());

        for (i, &d) in decoded.iter().enumerate() {
            assert!(
                d.abs() <= 1,
                "zero encryption error at {i}: got {d}"
            );
        }
    }

    // ---- Activation approximation tests ----

    #[test]
    fn test_gelu_vs_lut() {
        let poly = gelu_poly_approx();
        let lut = NonlinearLUT::gelu(8);

        // Compare polynomial and LUT approximations
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let poly_val = poly.eval(x);
            let lut_val = lut.eval(x);

            // Both should be reasonable approximations of GELU
            // but may differ from each other
            let gelu_exact = 0.5 * x * (1.0 + ((2.0_f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());

            assert!(
                (poly_val - gelu_exact).abs() < 0.2,
                "Poly GELU({x}) = {poly_val}, exact = {gelu_exact}"
            );
            assert!(
                (lut_val - gelu_exact).abs() < 0.2,
                "LUT GELU({x}) = {lut_val}, exact = {gelu_exact}"
            );
        }
    }

    // ---- LayerNorm tests ----

    #[test]
    fn test_rmsnorm_unit_input() {
        let config = LayerNormConfig::rms_norm(4);
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let result = layernorm_plaintext(&values, &config);

        // All equal inputs → all equal outputs ≈ 1.0 (RMS = 1.0)
        for &r in &result {
            assert!((r - 1.0).abs() < 0.01, "RMSNorm unit input: {r}");
        }
    }

    // ---- Attention tests ----

    #[test]
    fn test_softmax_properties() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];

        let poly_result = poly_softmax_plaintext(&scores);
        let relu_result = relu_squared_softmax_plaintext(&scores);

        // Both should sum to 1
        let poly_sum: f64 = poly_result.iter().sum();
        let relu_sum: f64 = relu_result.iter().sum();
        assert!((poly_sum - 1.0).abs() < 1e-6);
        assert!((relu_sum - 1.0).abs() < 1e-6);

        // Both should be monotonically increasing
        for i in 1..scores.len() {
            assert!(poly_result[i] > poly_result[i - 1]);
            assert!(relu_result[i] > relu_result[i - 1]);
        }

        // All values should be positive
        for &v in poly_result.iter().chain(relu_result.iter()) {
            assert!(v > 0.0);
        }
    }

    // ---- Transformer planning tests ----

    #[test]
    fn test_full_model_plan() {
        let dims = ModelDims::dense_7b();
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let config = default_block_config(dims.clone(), params.clone());
        let plan = plan_forward_pass(&config, dims.n_layers, &params);

        // Verify plan is reasonable
        assert_eq!(plan.num_layers, 32);
        assert!(plan.total_depth > 0);
        assert!(plan.total_ct_pt_muls > 0);
        assert!(plan.total_ct_ct_muls > 0);

        // Print cost summary for documentation
        let _ = format!(
            "Dense 7B model @ CHIMERA-128:\n\
             - Total depth: {}\n\
             - Bootstrapping needed: {}\n\
             - Bootstraps: {}\n\
             - CT-PT muls: {}\n\
             - CT-CT muls: {}",
            plan.total_depth,
            plan.needs_bootstrapping,
            plan.num_bootstraps,
            plan.total_ct_pt_muls,
            plan.total_ct_ct_muls,
        );
    }

    // ---- MoE tests ----

    #[test]
    fn test_moe_cost_comparison() {
        let dims_dense = ModelDims::dense_7b();
        let dims_moe = ModelDims::moe_40b();

        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);

        let config_dense = default_block_config(dims_dense.clone(), params.clone());
        let config_moe = default_block_config(dims_moe.clone(), params.clone());

        let _plan_dense = plan_transformer_block(&config_dense);
        let _plan_moe = plan_transformer_block(&config_moe);

        // MoE block has more FFN multiplications (larger d_ffn)
        // but in practice only n_active experts are evaluated
        let moe_config = MoEConfig::from_dims(&dims_moe, RoutingStrategy::CleartextRouting);
        let moe_plan = plan_moe_routing(&moe_config);

        // Active cost should be fraction of total
        assert!(moe_plan.sparsity_ratio < 1.0);
        assert_eq!(moe_plan.expert_evals, 2);
    }

    // ---- Noise tracking tests ----

    #[test]
    fn test_noise_budget_progression() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let mut tracker = NoiseTracker::fresh();

        let initial_budget = tracker.budget_bits(&params);
        assert!(initial_budget > 0.0);

        // After plaintext multiplication, budget should decrease
        tracker.mul_const(10.0);
        let after_mul = tracker.budget_bits(&params);
        assert!(after_mul < initial_budget);

        // After rescaling, budget should partially recover
        tracker.rescale(params.base2k.0);
        let after_rescale = tracker.budget_bits(&params);
        // Rescaling reduces the noise variance
        assert!(after_rescale > after_mul || tracker.variance < 1e10);
    }

    // ---- LUT cost estimation tests ----

    #[test]
    fn test_lut_vs_poly_cost() {
        let params = ChimeraParams::new(SecurityLevel::Bits128, Precision::Int8);
        let lut = NonlinearLUT::gelu(8);

        let lut_cost = lut_eval_total_cost(&params, &lut, 4096);
        // Polynomial evaluation cost is ~degree multiplications per slot
        let poly_cost = 3 * 4096; // degree-3 GELU, 4096 slots

        // LUT should be much more expensive than polynomial
        assert!(
            lut_cost > poly_cost * 100,
            "LUT cost ({lut_cost}) should be >> poly cost ({poly_cost})"
        );
    }

    // ---- Ciphertext-ciphertext multiplication tests ----

    #[test]
    fn test_chimera_ct_mul_basic() {
        // Test ct*ct multiplication by directly calling glwe_tensor_apply
        // and glwe_tensor_relinearize, following the exact pattern from
        // poulpy-core's tensor test. This validates our parameter setup.
        use poulpy_core::{
            GLWESub, GLWETensoring,
            layouts::{
                Base2K, Degree, Dsize, GLWE, GLWELayout, GLWEPlaintext,
                GLWESecret, GLWETensor, GLWETensorKey,
                GLWETensorKeyLayout, LWEInfos, Rank, TorusPrecision,
                prepared::{
                    GLWESecretPrepared,
                    GLWETensorKeyPrepared,
                },
            },
        };
        use poulpy_hal::{
            api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize},
            layouts::{ScratchOwned, VecZnx},
            source::Source,
        };

        // Parameters following the tensor test pattern exactly:
        let base2k: usize = 14;
        let in_base2k: usize = base2k - 1;
        let out_base2k: usize = base2k - 2;
        let tsk_base2k: usize = base2k;
        let k: usize = 8 * base2k + 1; // 113
        let rank: usize = 1;
        let n: u64 = 4096;

        let module: Module<BE> = Module::new(n);
        let n_usize = module.n();

        let glwe_in = GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(in_base2k as u32),
            k: TorusPrecision(k as u32),
            rank: Rank(rank as u32),
        };

        let glwe_out = GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(out_base2k as u32),
            k: TorusPrecision(k as u32),
            rank: Rank(rank as u32),
        };

        let tsk_layout = GLWETensorKeyLayout {
            n: Degree(n as u32),
            base2k: Base2K(tsk_base2k as u32),
            k: TorusPrecision((k + tsk_base2k) as u32),
            rank: Rank(rank as u32),
            dnum: poulpy_core::layouts::Dnum(k.div_ceil(tsk_base2k) as u32),
            dsize: Dsize(1),
        };

        // Generate secret key
        let mut source_xs = Source::new([10u8; 32]);
        let mut source_xe = Source::new([20u8; 32]);
        let mut source_xa = Source::new([30u8; 32]);

        let mut sk = GLWESecret::<Vec<u8>>::alloc(Degree(n as u32), Rank(rank as u32));
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft = GLWESecretPrepared::<Vec<u8>, BE>::alloc_from_infos(&module, &sk);
        sk_dft.prepare(&module, &sk);

        // Generate and prepare tensor key
        let mut tsk = GLWETensorKey::<Vec<u8>>::alloc_from_infos(&tsk_layout);

        let scale: usize = 2 * in_base2k;

        // Allocate all ciphertexts/plaintexts
        let mut a = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        let mut b = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        let mut res_tensor = GLWETensor::<Vec<u8>>::alloc_from_infos(&glwe_out);
        let mut res_relin = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_out);
        let mut pt_in = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_in);
        let mut pt_have = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
        let mut pt_want = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
        let mut pt_tmp = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);

        // Compute max scratch needed
        let scratch_bytes = GLWETensorKey::encrypt_sk_tmp_bytes(&module, &tsk_layout)
            .max(GLWE::encrypt_sk_tmp_bytes(&module, &glwe_in))
            .max(GLWE::decrypt_tmp_bytes(&module, &glwe_out))
            .max(module.glwe_tensor_apply_tmp_bytes(&res_tensor, scale, &a, &b))
            .max(module.glwe_tensor_relinearize_tmp_bytes(&res_relin, &res_tensor, &tsk_layout));
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

        tsk.encrypt_sk(&module, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

        let mut tsk_prep = GLWETensorKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(&module, &tsk_layout);
        let prep_bytes = tsk_prep.prepare_tmp_bytes(&module, &tsk_layout);
        let mut prep_scratch: ScratchOwned<BE> = ScratchOwned::alloc(prep_bytes);
        tsk_prep.prepare(&module, &tsk, prep_scratch.borrow());

        // Encode a single small value (3) at coefficient 0
        let mut data = vec![0i64; n_usize];
        data[0] = 3;
        pt_in.encode_vec_i64(&data, TorusPrecision(scale as u32));

        // Compute expected product plaintext using bivariate convolution
        // For [3, 0, 0, ...] * [3, 0, 0, ...] = [9, 0, 0, ...] in negacyclic ring
        let mut expected_data = vec![0i64; n_usize];
        expected_data[0] = 9; // 3 * 3

        // Encrypt
        a.encrypt_sk(&module, &pt_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());
        b.encrypt_sk(&module, &pt_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());

        // Direct tensor product + relinearize (following tensor test)
        module.glwe_tensor_apply(&mut res_tensor, scale, &a, &b, scratch.borrow());
        module.glwe_tensor_relinearize(&mut res_relin, &res_tensor, &tsk_prep, tsk_prep.size(), scratch.borrow());

        // Decrypt
        res_relin.decrypt(&module, &mut pt_have, &sk_dft, scratch.borrow());

        // Compute expected output plaintext using normalize (same as tensor test)
        // The bivariate convolution of pt_in with itself at base in_base2k gives
        // a result in in_base2k representation. We need to normalize it to out_base2k.
        use poulpy_hal::test_suite::convolution::bivariate_convolution_naive;
        let mut pt_want_base2k_in = VecZnx::alloc(n_usize, 1, pt_in.size());
        bivariate_convolution_naive(
            &module,
            in_base2k,
            2, // truncation
            &mut pt_want_base2k_in,
            0,
            pt_in.data(),
            0,
            pt_in.data(),
            0,
            scratch.borrow(),
        );

        // Normalize to out_base2k (res_offset = 0 for the first iteration like the tensor test)
        module.vec_znx_normalize(
            pt_want.data_mut(),
            out_base2k,
            0, // res_offset_lo = 0 (first iteration)
            0, // col
            &pt_want_base2k_in,
            in_base2k,
            0, // col
            scratch.borrow(),
        );

        // Compare pt_have vs pt_want via subtraction + noise check
        module.glwe_sub(&mut pt_tmp, &pt_have, &pt_want);

        // Decode both to check values
        let mut decoded_have = vec![0i64; n_usize];
        let mut decoded_want = vec![0i64; n_usize];
        pt_have.decode_vec_i64(&mut decoded_have, TorusPrecision(scale as u32));
        pt_want.decode_vec_i64(&mut decoded_want, TorusPrecision(scale as u32));

        // The product 3*3 = 9 should appear in coefficient 0
        let diff = (decoded_have[0] - decoded_want[0]).abs();
        assert!(
            diff <= 2,
            "direct tensor product error at coeff 0: have={}, want={}, diff={}",
            decoded_have[0], decoded_want[0], diff
        );

        // Now test via chimera_ct_mul wrapper
        use crate::activations::chimera_ct_mul;

        // Re-encrypt fresh ciphertexts
        let mut ct_a2 = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        ct_a2.encrypt_sk(&module, &pt_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());
        let mut ct_b2 = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        ct_b2.encrypt_sk(&module, &pt_in, &sk_dft, &mut source_xa, &mut source_xe, scratch.borrow());

        let eval_key = ChimeraEvalKey {
            tensor_key_prepared: tsk_prep,
            tensor_key_layout: tsk_layout,
            output_layout: glwe_out.clone(),
            tensor_layout: glwe_out.clone(),
            res_offset: scale,
            auto_keys: std::collections::HashMap::new(),
            auto_key_layout: poulpy_core::layouts::GLWEAutomorphismKeyLayout {
                n: Degree(n as u32),
                base2k: Base2K(in_base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(rank as u32),
                dsize: Dsize(1),
                dnum: poulpy_core::layouts::Dnum(1),
            },
        };

        let ct_prod = chimera_ct_mul(&module, &eval_key, &ct_a2, &ct_b2);

        // Decrypt the result
        let mut pt_dec2 = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
        ct_prod.decrypt(&module, &mut pt_dec2, &sk_dft, scratch.borrow());

        let mut decoded2 = vec![0i64; n_usize];
        pt_dec2.decode_vec_i64(&mut decoded2, TorusPrecision(scale as u32));

        let diff2 = (decoded2[0] - decoded_want[0]).abs();
        assert!(
            diff2 <= 2,
            "chimera_ct_mul error at coeff 0: have={}, want={}, diff={}",
            decoded2[0], decoded_want[0], diff2
        );
    }

    #[test]
    fn test_chimera_eval_key_generation() {
        // Test that ChimeraEvalKey can be generated and has sensible parameters.
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let eval_key = ChimeraEvalKey::generate(
            &module,
            &key,
            &params,
            [50u8; 32],
            [60u8; 32],
        );

        // Output base2k should be params.base2k - 2
        assert_eq!(
            eval_key.output_layout.base2k.0 as usize,
            params.base2k.0 as usize - 2,
        );

        // res_offset should be 2 * base2k
        assert_eq!(eval_key.res_offset, 2 * params.base2k.0 as usize);

        // Automorphism keys should be generated for trace (log2(N) keys)
        assert!(
            !eval_key.auto_keys.is_empty(),
            "auto_keys should not be empty"
        );
    }

    #[test]
    fn test_apply_poly_activation_squared_relu() {
        // Test apply_poly_activation with squared ReLU (x²), the simplest
        // non-trivial polynomial (degree 2, one multiplication).
        //
        // Uses the same low-level parameter setup as test_chimera_ct_mul_basic.
        use crate::activations::{apply_poly_activation, squared_relu_approx};
        use poulpy_core::layouts::{
            Base2K, Degree, Dsize, GLWE, GLWELayout, GLWEPlaintext,
            GLWESecret, GLWETensorKey, GLWETensorKeyLayout, Rank, TorusPrecision,
            prepared::{GLWESecretPrepared, GLWETensorKeyPrepared},
        };
        use poulpy_hal::{
            api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
            layouts::ScratchOwned,
            source::Source,
        };

        let base2k: usize = 14;
        let in_base2k: usize = base2k - 1;
        let out_base2k: usize = base2k - 2;
        let tsk_base2k: usize = base2k;
        let k: usize = 8 * base2k + 1;
        let rank: usize = 1;
        let n: u64 = 4096;

        let module: Module<BE> = Module::new(n);
        let n_usize = module.n();

        let glwe_in = GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(in_base2k as u32),
            k: TorusPrecision(k as u32),
            rank: Rank(rank as u32),
        };

        let glwe_out = GLWELayout {
            n: Degree(n as u32),
            base2k: Base2K(out_base2k as u32),
            k: TorusPrecision(k as u32),
            rank: Rank(rank as u32),
        };

        let tsk_layout = GLWETensorKeyLayout {
            n: Degree(n as u32),
            base2k: Base2K(tsk_base2k as u32),
            k: TorusPrecision((k + tsk_base2k) as u32),
            rank: Rank(rank as u32),
            dnum: poulpy_core::layouts::Dnum(k.div_ceil(tsk_base2k) as u32),
            dsize: Dsize(1),
        };

        let mut source_xs = Source::new([11u8; 32]);
        let mut source_xe = Source::new([22u8; 32]);
        let mut source_xa = Source::new([33u8; 32]);

        let mut sk = GLWESecret::<Vec<u8>>::alloc(Degree(n as u32), Rank(rank as u32));
        sk.fill_ternary_prob(0.5, &mut source_xs);

        let mut sk_dft = GLWESecretPrepared::<Vec<u8>, BE>::alloc_from_infos(&module, &sk);
        sk_dft.prepare(&module, &sk);

        let mut tsk = GLWETensorKey::<Vec<u8>>::alloc_from_infos(&tsk_layout);
        let encrypt_bytes = GLWETensorKey::encrypt_sk_tmp_bytes(&module, &tsk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(encrypt_bytes);
        tsk.encrypt_sk(&module, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

        let mut tsk_prep = GLWETensorKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(&module, &tsk_layout);
        let prep_bytes = tsk_prep.prepare_tmp_bytes(&module, &tsk_layout);
        let mut prep_scratch: ScratchOwned<BE> = ScratchOwned::alloc(prep_bytes);
        tsk_prep.prepare(&module, &tsk, prep_scratch.borrow());

        let eval_key = ChimeraEvalKey {
            tensor_key_prepared: tsk_prep,
            tensor_key_layout: tsk_layout,
            output_layout: glwe_out.clone(),
            tensor_layout: glwe_out.clone(),
            res_offset: 2 * in_base2k,
            auto_keys: std::collections::HashMap::new(),
            auto_key_layout: poulpy_core::layouts::GLWEAutomorphismKeyLayout {
                n: Degree(n as u32),
                base2k: Base2K(in_base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(rank as u32),
                dsize: Dsize(1),
                dnum: poulpy_core::layouts::Dnum(1),
            },
        };

        // Encode a single value (3) in coefficient 0.
        // For SqReLU (x²), the result should be 9.
        let scale = 2 * in_base2k;
        let mut data = vec![0i64; n_usize];
        data[0] = 3;
        let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_in);
        pt.encode_vec_i64(&data, TorusPrecision(scale as u32));

        let encrypt_ct_bytes = GLWE::encrypt_sk_tmp_bytes(&module, &glwe_in);
        let mut ct_scratch: ScratchOwned<BE> = ScratchOwned::alloc(encrypt_ct_bytes);

        let mut ct = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        ct.encrypt_sk(&module, &pt, &sk_dft, &mut source_xa, &mut source_xe, ct_scratch.borrow());

        // Apply squared ReLU: p(x) = x²
        let approx = squared_relu_approx();
        let ct_result = apply_poly_activation(&module, &eval_key, &ct, &approx);

        // Decrypt
        let decrypt_bytes = GLWE::decrypt_tmp_bytes(&module, &glwe_out);
        let mut dec_scratch: ScratchOwned<BE> = ScratchOwned::alloc(decrypt_bytes);
        let mut pt_dec = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
        ct_result.decrypt(&module, &mut pt_dec, &sk_dft, dec_scratch.borrow());

        // Decode using decode_vec_i64 (same approach as the passing ct_mul test)
        let mut decoded = vec![0i64; n_usize];
        pt_dec.decode_vec_i64(&mut decoded, TorusPrecision(scale as u32));

        // For SqReLU with coeffs [0.0, 0.0, 1.0]:
        // c₂ = 1.0 → mul_const coefficient = 1 (identity).
        // The tensor product computes ct*ct = encrypt(3*3) = encrypt(9).
        // mul_const by 1 leaves it unchanged.
        // decode_vec_i64 should recover 9 (within noise tolerance).
        let diff = (decoded[0] - 9).abs();
        assert!(
            diff <= 2,
            "squared ReLU(3) should be ~9, got decoded[0]={}, diff={}",
            decoded[0], diff
        );
    }
}
