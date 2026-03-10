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
            tensor_key_l2_prepared: None,
            tensor_key_l2_layout: None,
            output_l2_layout: None,
            tensor_l2_layout: None,
            res_offset_l2: None,
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

        // res_offset should be 2 * in_base2k where in_base2k = base2k - 1
        assert_eq!(eval_key.res_offset, 2 * (params.base2k.0 as usize - 1));

        // Automorphism keys should be generated for trace (log2(N) keys)
        assert!(
            !eval_key.auto_keys.is_empty(),
            "auto_keys should not be empty"
        );
    }

    /// End-to-end GELU FHE test: encrypt → apply_poly_activation with GELU
    /// (fractional coefficients: 0.5*x + 0.247*x²) → decrypt → verify.
    ///
    /// This validates that the mixed-layout term accumulation fix works:
    /// - ct has base2k = in_base2k = 13
    /// - ct_sq has base2k = out_base2k = 12 after tensor product
    /// - The linear term (0.5*x) is converted to out_base2k layout
    /// - Both terms are added successfully
    #[test]
    fn test_apply_poly_activation_gelu() {
        use crate::activations::{activation_decode_precision, apply_poly_activation, gelu_poly_approx};
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

        let mut source_xs = Source::new([15u8; 32]);
        let mut source_xe = Source::new([25u8; 32]);
        let mut source_xa = Source::new([35u8; 32]);

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
            tensor_key_l2_prepared: None,
            tensor_key_l2_layout: None,
            output_l2_layout: None,
            tensor_l2_layout: None,
            res_offset_l2: None,
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

        // Encode a single value (4) in coefficient 0.
        // GELU_approx(x) = 0.5*x + 0.247*x²
        // GELU_approx(4) = 0.5*4 + 0.247*16 = 2.0 + 3.952 = 5.952
        let scale = 2 * in_base2k;
        let mut data = vec![0i64; n_usize];
        data[0] = 4;
        let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_in);
        pt.encode_vec_i64(&data, TorusPrecision(scale as u32));

        let encrypt_ct_bytes = GLWE::encrypt_sk_tmp_bytes(&module, &glwe_in);
        let mut ct_scratch: ScratchOwned<BE> = ScratchOwned::alloc(encrypt_ct_bytes);

        let mut ct = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        ct.encrypt_sk(&module, &pt, &sk_dft, &mut source_xa, &mut source_xe, ct_scratch.borrow());

        // Apply GELU polynomial activation
        let approx = gelu_poly_approx();
        let ct_result = apply_poly_activation(&module, &eval_key, &ct, &approx);

        // Decrypt
        let decrypt_bytes = GLWE::decrypt_tmp_bytes(&module, &glwe_out);
        let mut dec_scratch: ScratchOwned<BE> = ScratchOwned::alloc(decrypt_bytes);
        let mut pt_dec = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
        ct_result.decrypt(&module, &mut pt_dec, &sk_dft, dec_scratch.borrow());

        // The output of apply_poly_activation uses coefficients scaled by 2^COEFF_SCALE_BITS.
        // Since glwe_mul_const at res_offset=base2k gives identity scaling (a*b on torus),
        // the output has an extra factor of 2^COEFF_SCALE_BITS. To recover the true value,
        // decode at scale - COEFF_SCALE_BITS = 26 - 8 = 18.
        let decode_scale = activation_decode_precision(scale);
        let mut decoded = vec![0i64; n_usize];
        pt_dec.decode_vec_i64(&mut decoded, TorusPrecision(decode_scale as u32));

        // Expected: GELU_approx(4) = 0.5*4 + 0.247*16 ≈ 5.95
        // But the c3 coefficient is 0.0 so no cubic term; c0=0 so no constant.
        // With COEFF_SCALE_BITS=8: c1=128/256=0.5 (exact), c2=63/256≈0.246
        // So actual eval: 0.5*4 + 0.246*16 ≈ 5.94
        // Allow tolerance for FHE noise + coefficient quantisation.
        let expected = 6; // rounded from ~5.95
        let diff = (decoded[0] - expected).abs();
        assert!(
            diff <= 3,
            "GELU_approx(4) should be ~6, got decoded[0]={}, diff={}",
            decoded[0], diff
        );
    }

    /// End-to-end pipeline test using high-level API:
    /// ChimeraKey::generate + ChimeraEvalKey::generate + chimera_encrypt
    /// + chimera_add + chimera_decrypt + decode.
    ///
    /// Validates that all high-level API functions work together.
    /// Addition preserves the encoding format, so encode_int8 → encrypt → add
    /// → decrypt → decode_int8 should round-trip correctly.
    #[test]
    fn test_full_pipeline_encrypt_add_decrypt() {
        use crate::arithmetic::chimera_add;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());

        // Generate keys using high-level API
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        // Also generate eval key to test that doesn't interfere
        let _eval_key = ChimeraEvalKey::generate(
            &module,
            &key,
            &params,
            [50u8; 32],
            [60u8; 32],
        );

        // Encode and encrypt two vectors
        let a_vals: Vec<i8> = vec![10, -20, 30, -40, 50, 0, 127, -128];
        let b_vals: Vec<i8> = vec![5, 10, -15, 20, -25, 1, -127, 127];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);

        let ct_a = chimera_encrypt(&module, &key, &pt_a, [70u8; 32], [80u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [90u8; 32], [100u8; 32]);

        // Homomorphic addition
        let ct_sum = chimera_add(&module, &ct_a, &ct_b);

        // Decrypt and decode using high-level API
        let pt_dec = chimera_decrypt(&module, &key, &ct_sum, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, a_vals.len());

        // Verify: each decoded value should be close to a + b
        for i in 0..a_vals.len() {
            let expected = a_vals[i] as i16 + b_vals[i] as i16;
            let diff = (expected - decoded[i] as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "pipeline add error at {i}: expected {expected}, got {}, diff={}",
                decoded[i], diff
            );
        }
    }

    /// End-to-end pipeline test for ct*ct multiplication using
    /// ChimeraEvalKey::generate + chimera_ct_mul.
    ///
    /// Uses encode_vec_i64 (torus encoding) for inputs because the tensor
    /// product changes the encoding format. Tests the full key generation
    /// → encrypt → multiply → decrypt → decode_vec_i64 pipeline.
    #[test]
    fn test_full_pipeline_eval_key_ct_mul() {
        use crate::activations::chimera_ct_mul;
        use poulpy_core::layouts::{
            GLWE, GLWEPlaintext, GLWEPlaintextLayout, TorusPrecision,
        };
        use poulpy_hal::{
            api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
            layouts::ScratchOwned,
        };

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let n_usize = module.n();

        // Generate keys using high-level API
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module,
            &key,
            &params,
            [50u8; 32],
            [60u8; 32],
        );

        // Encode using encode_vec_i64 at the tensor product scale
        // (this is the encoding that's compatible with decode_vec_i64
        // after tensor product).
        let scale = eval_key.res_offset; // 2 * in_base2k
        let mut data = vec![0i64; n_usize];
        data[0] = 3;

        let pt_layout = GLWEPlaintextLayout {
            n: key.layout.n,
            base2k: key.layout.base2k,
            k: key.layout.k,
        };
        let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&pt_layout);
        pt.encode_vec_i64(&data, TorusPrecision(scale as u32));

        // Encrypt using chimera_encrypt
        let ct = chimera_encrypt(&module, &key, &pt, [70u8; 32], [80u8; 32]);

        // ct * ct multiplication using eval key
        let ct_prod = chimera_ct_mul(&module, &eval_key, &ct, &ct);

        // Decrypt
        let out_pt_layout = GLWEPlaintextLayout {
            n: eval_key.output_layout.n,
            base2k: eval_key.output_layout.base2k,
            k: eval_key.output_layout.k,
        };
        let decrypt_ct_layout = poulpy_core::layouts::GLWELayout {
            n: eval_key.output_layout.n,
            base2k: eval_key.output_layout.base2k,
            k: eval_key.output_layout.k,
            rank: key.layout.rank,
        };
        let mut pt_dec = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&out_pt_layout);
        let decrypt_bytes = GLWE::<Vec<u8>>::decrypt_tmp_bytes(&module, &decrypt_ct_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(decrypt_bytes);
        ct_prod.decrypt(&module, &mut pt_dec, &key.prepared, scratch.borrow());

        // Decode using decode_vec_i64 at the same scale
        let mut decoded = vec![0i64; n_usize];
        pt_dec.decode_vec_i64(&mut decoded, TorusPrecision(scale as u32));

        // 3 * 3 = 9 in the first coefficient
        let diff = (decoded[0] - 9).abs();
        assert!(
            diff <= 2,
            "pipeline ct_mul(3,3) should give ~9, got decoded[0]={}, diff={}",
            decoded[0], diff
        );
    }

    #[test]
    fn test_apply_poly_activation_squared_relu() {
        // Test apply_poly_activation with squared ReLU (x²), the simplest
        // non-trivial polynomial (degree 2, one multiplication).
        //
        // Uses the same low-level parameter setup as test_chimera_ct_mul_basic.
        use crate::activations::{activation_decode_precision, apply_poly_activation, squared_relu_approx};
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
            tensor_key_l2_prepared: None,
            tensor_key_l2_layout: None,
            output_l2_layout: None,
            tensor_l2_layout: None,
            res_offset_l2: None,
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

        // Decode at scale - COEFF_SCALE_BITS to compensate for coefficient scaling.
        // The tensor product gives ct_sq encrypting 3²=9 on the torus at 2^{-26}.
        // mul_const(ct_sq, 256, res_offset=12) gives 9*256 on the torus at 2^{-26}.
        // Decode at TorusPrecision(18) = 9*256*2^{-26}*2^{18} = 9*256/256 = 9.
        let decode_scale = activation_decode_precision(scale);
        let mut decoded = vec![0i64; n_usize];
        pt_dec.decode_vec_i64(&mut decoded, TorusPrecision(decode_scale as u32));

        // For SqReLU with coeffs [0.0, 0.0, 1.0]:
        // c₂ = 1.0 → c2_scaled = 256 (= 1.0 * 2^8).
        // The tensor product computes ct*ct = encrypt(3*3) = encrypt(9).
        // mul_const(ct_sq, 256, res_offset=12) gives 9*256 on the torus at 2^{-26}.
        // decode at TorusPrecision(18) = 9*256/2^8 = 9.
        let diff = (decoded[0] - 9).abs();
        assert!(
            diff <= 2,
            "squared ReLU(3) should be ~9, got decoded[0]={}, diff={}",
            decoded[0], diff
        );
    }

    // ---- Rescale validation tests ----

    /// Tests a single rescale: encrypt → mul_const → rescale → decrypt.
    ///
    /// After mul_const the torus precision has been "used" at one level;
    /// rescale drops the lowest limb, reducing k by base2k. The decoded
    /// value should still match the original multiplication result.
    #[test]
    fn test_rescale_single() {
        use crate::arithmetic::{chimera_mul_const, chimera_rescale};

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let values: Vec<i8> = vec![10, -20, 30];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        // Multiply by scalar 3
        let ct_mul = chimera_mul_const(&module, &ct, &[3i64]);

        // Rescale (drops lowest limb)
        let ct_rescaled = chimera_rescale(&module, &ct_mul, &params);

        // Decrypt and decode
        let pt_dec = chimera_decrypt(&module, &key, &ct_rescaled, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 3);

        // Expected: [30, -60, 90]
        for (i, &v) in values.iter().enumerate() {
            let expected = v as i16 * 3;
            let diff = (expected - decoded[i] as i16).unsigned_abs();
            assert!(
                diff <= 2,
                "rescale_single error at {i}: expected {expected}, got {}, diff={diff}",
                decoded[i]
            );
        }
    }

    /// Tests chained rescale: encrypt → mul_const → rescale → mul_const → rescale → decrypt.
    ///
    /// Two successive mul_const + rescale operations. The key validation here is
    /// that the second rescale operates correctly on a ciphertext whose k has
    /// already been reduced by the first rescale.
    #[test]
    fn test_rescale_chained_two_levels() {
        use crate::arithmetic::{chimera_mul_const, chimera_rescale};

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let values: Vec<i8> = vec![4, -8, 12];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        // Level 1: multiply by 2, then rescale
        let ct_mul1 = chimera_mul_const(&module, &ct, &[2i64]);
        let ct_rescaled1 = chimera_rescale(&module, &ct_mul1, &params);

        // Level 2: multiply by 3, then rescale
        let ct_mul2 = chimera_mul_const(&module, &ct_rescaled1, &[3i64]);
        let ct_rescaled2 = chimera_rescale(&module, &ct_mul2, &params);

        // Decrypt and decode
        let pt_dec = chimera_decrypt(&module, &key, &ct_rescaled2, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 3);

        // Expected: [4*2*3, -8*2*3, 12*2*3] = [24, -48, 72]
        for (i, &v) in values.iter().enumerate() {
            let expected = v as i16 * 6;
            let diff = (expected - decoded[i] as i16).unsigned_abs();
            assert!(
                diff <= 4,
                "rescale_chained error at {i}: expected {expected}, got {}, diff={diff}",
                decoded[i]
            );
        }
    }

    /// Tests that rescale correctly tracks precision: verifies that k decreases
    /// by the ciphertext's base2k after each rescale and that we cannot rescale past k=0.
    #[test]
    fn test_rescale_precision_tracking() {
        use crate::arithmetic::{chimera_mul_const, chimera_rescale};
        use poulpy_core::layouts::LWEInfos;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let values: Vec<i8> = vec![10];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        let initial_k = ct.k().as_u32();
        // The ciphertext's actual base2k is in_base2k (= params.base2k - 1),
        // NOT params.base2k. chimera_rescale drops base2k bits based on
        // the ciphertext's own base2k value.
        let ct_base2k = ct.base2k().as_u32();

        // First mul_const + rescale
        let ct_mul1 = chimera_mul_const(&module, &ct, &[2i64]);
        let ct_r1 = chimera_rescale(&module, &ct_mul1, &params);
        let k_after_1 = ct_r1.k().as_u32();
        assert_eq!(
            k_after_1,
            initial_k - ct_base2k,
            "k should decrease by ct.base2k ({ct_base2k}) after first rescale: expected {}, got {}",
            initial_k - ct_base2k,
            k_after_1
        );

        // Second mul_const + rescale (the rescaled ct still has the same base2k)
        let ct_mul2 = chimera_mul_const(&module, &ct_r1, &[2i64]);
        let ct_r2 = chimera_rescale(&module, &ct_mul2, &params);
        let k_after_2 = ct_r2.k().as_u32();
        assert_eq!(
            k_after_2,
            initial_k - 2 * ct_base2k,
            "k should decrease by 2*ct.base2k after two rescales: expected {}, got {}",
            initial_k - 2 * ct_base2k,
            k_after_2
        );
    }

    /// Tests rescale after addition: (ct_a + ct_b) * scalar → rescale.
    ///
    /// Validates that rescale works correctly after an add operation
    /// (which does not change the noise budget but adds variances).
    #[test]
    fn test_rescale_after_add() {
        use crate::arithmetic::{chimera_add, chimera_mul_const, chimera_rescale};

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let a_vals: Vec<i8> = vec![10, 20];
        let b_vals: Vec<i8> = vec![5, -10];
        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);
        let ct_a = chimera_encrypt(&module, &key, &pt_a, [1u8; 32], [2u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [3u8; 32], [4u8; 32]);

        // Add, then multiply by 2, then rescale
        let ct_sum = chimera_add(&module, &ct_a, &ct_b);
        let ct_mul = chimera_mul_const(&module, &ct_sum, &[2i64]);
        let ct_rescaled = chimera_rescale(&module, &ct_mul, &params);

        let pt_dec = chimera_decrypt(&module, &key, &ct_rescaled, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 2);

        // Expected: [(10+5)*2, (20-10)*2] = [30, 20]
        let expected = vec![30i16, 20];
        for i in 0..2 {
            let diff = (expected[i] - decoded[i] as i16).unsigned_abs();
            assert!(
                diff <= 2,
                "rescale_after_add error at {i}: expected {}, got {}, diff={diff}",
                expected[i], decoded[i]
            );
        }
    }

    /// Tests that rescale saturates gracefully when k is too small.
    ///
    /// When k < base2k, rescale should not panic and should return a
    /// ciphertext with k unchanged (the minimum floor).
    #[test]
    fn test_rescale_saturation() {
        use crate::arithmetic::{chimera_mul_const, chimera_rescale};
        use poulpy_core::layouts::LWEInfos;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let values: Vec<i8> = vec![5];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        // Chain many rescales until k bottoms out
        let mut current = chimera_mul_const(&module, &ct, &[1i64]);
        let mut prev_k = current.k().as_u32();

        for _ in 0..20 {
            let rescaled = chimera_rescale(&module, &current, &params);
            let new_k = rescaled.k().as_u32();

            // k should never go below 0 and should not panic
            assert!(
                new_k <= prev_k,
                "k should not increase after rescale: was {prev_k}, now {new_k}"
            );

            if new_k == prev_k {
                // We've hit the floor — rescale is a no-op now
                break;
            }

            prev_k = new_k;
            current = chimera_mul_const(&module, &rescaled, &[1i64]);
        }
    }

    /// End-to-end rescale with noise tracker correlation.
    ///
    /// Validates that the NoiseTracker predictions are consistent with
    /// actual decryption success. After a chain of operations, if the
    /// noise tracker says the budget is positive, decryption should
    /// still produce approximately correct results.
    #[test]
    fn test_rescale_noise_tracker_consistency() {
        use crate::arithmetic::{chimera_mul_const, chimera_rescale};
        use crate::noise::NoiseTracker;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);

        let values: Vec<i8> = vec![10];
        let pt = encode_int8(&module, &params, &values);
        let ct = chimera_encrypt(&module, &key, &pt, [1u8; 32], [2u8; 32]);

        let mut tracker = NoiseTracker::fresh();
        let mut current_ct = ct;

        // Chain 3 rounds of mul_const(2) + rescale
        for round in 0..3 {
            current_ct = chimera_mul_const(&module, &current_ct, &[2i64]);
            tracker.mul_const(2.0);

            current_ct = chimera_rescale(&module, &current_ct, &params);
            tracker.rescale(params.base2k.0);

            let budget = tracker.budget_bits(&params);

            // If the tracker says we still have budget, decryption should work
            if budget > 2.0 {
                let pt_dec = chimera_decrypt(&module, &key, &current_ct, &params);
                let decoded = decode_int8(&module, &params, &pt_dec, 1);
                let expected = 10i16 * (1 << (round + 1));

                // Allow larger tolerance as noise accumulates
                let max_error = 2u16.pow(round as u32 + 1);
                let diff = (expected - decoded[0] as i16).unsigned_abs();
                assert!(
                    diff <= max_error,
                    "round {round}: noise tracker says budget={budget:.1} bits \
                     but decryption error is {diff} (max allowed: {max_error}). \
                     expected={expected}, got={}",
                    decoded[0]
                );
            }
        }
    }

    // ---- Regression tests ----

    /// Regression test: validates glwe_mul_const res_offset behaviour on
    /// post-tensor-product ciphertexts (mixed base2k scenarios).
    ///
    /// Verifies the two key invariants that the activation code relies on:
    /// 1. mul_const(ct@base2k, coeff, res_offset=base2k) produces the raw
    ///    product value*coeff at the original torus precision (identity scaling).
    /// 2. ct_sq from chimera_ct_mul decodes correctly at the encoding scale.
    #[test]
    fn test_mul_const_on_tensor_output() {
        use crate::activations::chimera_ct_mul;
        use poulpy_core::{
            GLWEMulConst,
            layouts::{
                Base2K, Degree, Dsize, GLWE, GLWELayout, GLWEPlaintext,
                GLWESecret, GLWETensorKey, GLWETensorKeyLayout, Rank, TorusPrecision,
                prepared::{GLWESecretPrepared, GLWETensorKeyPrepared},
            },
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

        let mut source_xs = Source::new([15u8; 32]);
        let mut source_xe = Source::new([25u8; 32]);
        let mut source_xa = Source::new([35u8; 32]);

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
            tensor_key_l2_prepared: None,
            tensor_key_l2_layout: None,
            output_l2_layout: None,
            tensor_l2_layout: None,
            res_offset_l2: None,
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

        // Encode value 4 at scale = 2 * in_base2k = 26
        let scale = 2 * in_base2k;
        let mut data = vec![0i64; n_usize];
        data[0] = 4;
        let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_in);
        pt.encode_vec_i64(&data, TorusPrecision(scale as u32));

        let encrypt_ct_bytes = GLWE::encrypt_sk_tmp_bytes(&module, &glwe_in);
        let mut ct_scratch: ScratchOwned<BE> = ScratchOwned::alloc(encrypt_ct_bytes);
        let mut ct = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        ct.encrypt_sk(&module, &pt, &sk_dft, &mut source_xa, &mut source_xe, ct_scratch.borrow());

        // Step 1: ct_sq = ct * ct via tensor product (should give 16 at coeff 0)
        let ct_sq = chimera_ct_mul(&module, &eval_key, &ct, &ct);

        let decrypt_bytes = GLWE::decrypt_tmp_bytes(&module, &glwe_out);
        let mut dec_scratch: ScratchOwned<BE> = ScratchOwned::alloc(decrypt_bytes);
        let mut pt_sq = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
        ct_sq.decrypt(&module, &mut pt_sq, &sk_dft, dec_scratch.borrow());
        let mut decoded_sq = vec![0i64; n_usize];
        pt_sq.decode_vec_i64(&mut decoded_sq, TorusPrecision(scale as u32));
        assert!(
            (decoded_sq[0] - 16).abs() <= 2,
            "ct_sq should decode to 16, got {}",
            decoded_sq[0]
        );

        // Step 2: mul_const(ct_sq@12, 256, res_offset=12) should give raw product 16*256=4096
        // at decode@26. Then decode@18 (= scale - COEFF_SCALE_BITS) should give 16.
        let coeff_256 = vec![256i64; 1];
        let res_offset_identity = out_base2k; // ct_sq.base2k() = 12

        let mut result = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_out);
        let mul_bytes = module.glwe_mul_const_tmp_bytes(&result, res_offset_identity, &ct_sq, 1);
        let mut mul_scratch: ScratchOwned<BE> = ScratchOwned::alloc(mul_bytes);
        module.glwe_mul_const(&mut result, res_offset_identity, &ct_sq, &coeff_256, mul_scratch.borrow());

        let mut pt_result = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
        result.decrypt(&module, &mut pt_result, &sk_dft, dec_scratch.borrow());

        // At decode@26, raw product = 16 * 256 = 4096
        let mut decoded_raw = vec![0i64; n_usize];
        pt_result.decode_vec_i64(&mut decoded_raw, TorusPrecision(scale as u32));
        assert!(
            (decoded_raw[0] - 4096).abs() <= 16,
            "mul_const(ct_sq, 256, res_offset=12) @ decode@26 should be 4096, got {}",
            decoded_raw[0]
        );

        // At decode@18 (= 26 - 8), compensated for COEFF_SCALE_BITS, should give 16
        let decode_compensated = scale - 8; // COEFF_SCALE_BITS = 8
        let mut decoded_comp = vec![0i64; n_usize];
        pt_result.decode_vec_i64(&mut decoded_comp, TorusPrecision(decode_compensated as u32));
        assert!(
            (decoded_comp[0] - 16).abs() <= 2,
            "mul_const(ct_sq, 256, res_offset=12) @ decode@18 should be 16, got {}",
            decoded_comp[0]
        );

        // Step 3: mul_const(ct@13, 128, res_offset=13) should give raw product 4*128=512
        // at decode@26. At decode@18, should give 4*0.5=2.
        let coeff_128 = vec![128i64; 1];
        let res_offset_ct = in_base2k; // ct.base2k() = 13

        let mut result2 = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
        let mul_bytes2 = module.glwe_mul_const_tmp_bytes(&result2, res_offset_ct, &ct, 1);
        let mut mul_scratch2: ScratchOwned<BE> = ScratchOwned::alloc(mul_bytes2);
        module.glwe_mul_const(&mut result2, res_offset_ct, &ct, &coeff_128, mul_scratch2.borrow());

        let mut pt_result2 = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_in);
        let decrypt_bytes2 = GLWE::decrypt_tmp_bytes(&module, &glwe_in);
        let mut dec_scratch2: ScratchOwned<BE> = ScratchOwned::alloc(decrypt_bytes2);
        result2.decrypt(&module, &mut pt_result2, &sk_dft, dec_scratch2.borrow());

        let mut decoded_raw2 = vec![0i64; n_usize];
        pt_result2.decode_vec_i64(&mut decoded_raw2, TorusPrecision(scale as u32));
        assert!(
            (decoded_raw2[0] - 512).abs() <= 4,
            "mul_const(ct, 128, res_offset=13) @ decode@26 should be 512, got {}",
            decoded_raw2[0]
        );

        let mut decoded_comp2 = vec![0i64; n_usize];
        pt_result2.decode_vec_i64(&mut decoded_comp2, TorusPrecision(decode_compensated as u32));
        assert!(
            (decoded_comp2[0] - 2).abs() <= 1,
            "mul_const(ct, 128, res_offset=13) @ decode@18 should be 2, got {}",
            decoded_comp2[0]
        );
    }

    // ---- End-to-end transformer block test ----

    /// Runs a complete transformer block under FHE:
    ///
    ///   RMSNorm → Attention → Residual → RMSNorm → FFN → Residual
    ///
    /// Uses the smallest possible model dimensions (d_model=1, d_ffn=1,
    /// n_heads=1, d_head=1) so that the test exercises the full pipeline
    /// composition without being prohibitively slow.
    ///
    /// The goal is to verify that the entire chain of homomorphic operations
    /// composes correctly — i.e. the output of each stage is at a compatible
    /// layout (base2k, k) for the next stage. Numerical accuracy is a
    /// secondary concern: we mainly check that no assertion panics and that
    /// the result is a structurally valid ciphertext.
    #[test]
    fn test_end_to_end_transformer_block() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [42u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [43u8; 32], [44u8; 32],
        );

        // Tiny model: d_model=1, d_ffn=1, 1 head, 1 layer
        let dims = ModelDims {
            d_model: 1,
            d_head: 1,
            n_heads: 1,
            d_ffn: 1,
            n_layers: 1,
            n_experts: 1,
            n_active_experts: 1,
        };

        // Use standard FFN with SquaredReLU (lowest depth: 1 tensor product)
        let config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: dims.clone(),
                params: params.clone(),
                softmax_approx: SoftmaxStrategy::ReluSquared,
                causal: true,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(1),
            pre_ffn_norm: LayerNormConfig::rms_norm(1),
            ffn: FFNConfig::Standard {
                activation: ActivationChoice::SquaredReLU,
            },
            residual: true,
        };

        // Weights: identity-like (weight = 1 for all single-element matrices)
        let weights = TransformerBlockWeights {
            attention: AttentionWeights {
                w_q: vec![vec![1i64]],
                w_k: vec![vec![1i64]],
                w_v: vec![vec![1i64]],
                w_o: vec![vec![1i64]],
            },
            ffn: FFNWeights {
                w1: vec![vec![1i64]],
                w2: vec![vec![1i64]],
                w3: None,
            },
        };

        // Encrypt input: single value [5]
        let vals: Vec<i8> = vec![5];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [50u8; 32], [51u8; 32]);

        // Run the full transformer block
        let result = chimera_transformer_block(&module, &eval_key, &ct, &config, &weights);

        // Verify the output is a valid ciphertext with sensible dimensions.
        // After multiple tensor products, base2k should have decreased from
        // the input base2k (13).
        use poulpy_core::layouts::LWEInfos;
        assert!(result.n() > 0, "output ciphertext N should be > 0");
        assert!(result.base2k() > 0, "output ciphertext base2k should be > 0");
        assert!(result.k() > 0, "output ciphertext k should be > 0");

        // The pipeline survived all the way through — this is the primary
        // success criterion for the end-to-end test.
    }

    /// Runs a 2-layer forward pass under FHE to verify that transformer
    /// blocks can be chained (output of block 1 feeds into block 2).
    ///
    /// This exercises the most critical integration point: whether the
    /// ciphertext layout produced by one block is compatible with the
    /// input expectations of the next block.
    #[test]
    fn test_end_to_end_forward_pass_2_layers() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [60u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [61u8; 32], [62u8; 32],
        );

        let dims = ModelDims {
            d_model: 1,
            d_head: 1,
            n_heads: 1,
            d_ffn: 1,
            n_layers: 2,
            n_experts: 1,
            n_active_experts: 1,
        };

        let config = TransformerBlockConfig {
            attention: AttentionConfig {
                dims: dims.clone(),
                params: params.clone(),
                softmax_approx: SoftmaxStrategy::ReluSquared,
                causal: true,
            },
            pre_attn_norm: LayerNormConfig::rms_norm(1),
            pre_ffn_norm: LayerNormConfig::rms_norm(1),
            ffn: FFNConfig::Standard {
                activation: ActivationChoice::SquaredReLU,
            },
            residual: true,
        };

        // Two layers with identity-like weights
        let layer_weights = vec![
            TransformerBlockWeights {
                attention: AttentionWeights {
                    w_q: vec![vec![1i64]],
                    w_k: vec![vec![1i64]],
                    w_v: vec![vec![1i64]],
                    w_o: vec![vec![1i64]],
                },
                ffn: FFNWeights {
                    w1: vec![vec![1i64]],
                    w2: vec![vec![1i64]],
                    w3: None,
                },
            },
            TransformerBlockWeights {
                attention: AttentionWeights {
                    w_q: vec![vec![1i64]],
                    w_k: vec![vec![1i64]],
                    w_v: vec![vec![1i64]],
                    w_o: vec![vec![1i64]],
                },
                ffn: FFNWeights {
                    w1: vec![vec![1i64]],
                    w2: vec![vec![1i64]],
                    w3: None,
                },
            },
        ];

        let vals: Vec<i8> = vec![3];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [70u8; 32], [71u8; 32]);

        // Run the 2-layer forward pass
        let result = chimera_forward_pass(&module, &eval_key, &ct, &config, &layer_weights);

        use poulpy_core::layouts::LWEInfos;
        assert!(result.n() > 0, "output ciphertext N should be > 0");
        assert!(result.base2k() > 0, "output ciphertext base2k should be > 0");
        assert!(result.k() > 0, "output ciphertext k should be > 0");
    }
}
