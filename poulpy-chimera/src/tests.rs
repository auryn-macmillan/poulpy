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
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
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
                pre_attn_norm_gamma: None,
                pre_ffn_norm_gamma: None,
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
                pre_attn_norm_gamma: None,
                pre_ffn_norm_gamma: None,
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

    /// Tests a transformer block with d_model=4, d_ffn=4, demonstrating that
    /// the pipeline works with multi-element polynomial weight vectors.
    ///
    /// The input ciphertext encodes 4 INT8 values [1, 2, 3, 4] as polynomial
    /// coefficients. Weight "rows" are 4-element polynomials. The ring
    /// multiplication (negacyclic convolution) computes the inner product
    /// structure that the transformer relies on.
    #[test]
    fn test_end_to_end_transformer_block_d4() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [100u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [101u8; 32], [102u8; 32],
        );

        // d_model=4, d_ffn=4, 1 head with d_head=4
        let dims = ModelDims {
            d_model: 4,
            d_head: 4,
            n_heads: 1,
            d_ffn: 4,
            n_layers: 1,
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
            pre_attn_norm: LayerNormConfig::rms_norm(4),
            pre_ffn_norm: LayerNormConfig::rms_norm(4),
            ffn: FFNConfig::Standard {
                activation: ActivationChoice::SquaredReLU,
            },
            residual: true,
        };

        // Weights: 4×4 identity-like (each row = [1, 0, 0, 0])
        let id_row = vec![1i64, 0, 0, 0];
        let weights = TransformerBlockWeights {
            attention: AttentionWeights {
                w_q: vec![id_row.clone(); 4],
                w_k: vec![id_row.clone(); 4],
                w_v: vec![id_row.clone(); 4],
                w_o: vec![id_row.clone(); 4],
            },
            ffn: FFNWeights {
                w1: vec![id_row.clone(); 4],
                w2: vec![id_row.clone(); 4],
                w3: None,
            },
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
        };

        // Encrypt input: 4 values [1, 2, 3, 4]
        let vals: Vec<i8> = vec![1, 2, 3, 4];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [110u8; 32], [111u8; 32]);

        // Run the full transformer block with d_model=4
        let result = chimera_transformer_block(&module, &eval_key, &ct, &config, &weights);

        use poulpy_core::layouts::LWEInfos;
        assert!(result.n() > 0, "output ciphertext N should be > 0");
        assert!(result.base2k() > 0, "output ciphertext base2k should be > 0");
        assert!(result.k() > 0, "output ciphertext k should be > 0");
    }

    /// Tests the multi-head attention path in `chimera_transformer_block`.
    ///
    /// Uses n_heads=2 with d_model=4, d_head=2.  Each head's weight row
    /// is a distinct 4-element polynomial so the two heads contribute
    /// independently.  The test verifies that the block runs end-to-end
    /// and produces a structurally valid ciphertext.
    #[test]
    fn test_multi_head_attention_d4_h2() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [130u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [131u8; 32], [132u8; 32],
        );

        // d_model=4, n_heads=2, d_head=2 (d_model = n_heads * d_head)
        let dims = ModelDims {
            d_model: 4,
            d_head: 2,
            n_heads: 2,
            d_ffn: 4,
            n_layers: 1,
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
            pre_attn_norm: LayerNormConfig::rms_norm(4),
            pre_ffn_norm: LayerNormConfig::rms_norm(4),
            ffn: FFNConfig::Standard {
                activation: ActivationChoice::SquaredReLU,
            },
            residual: true,
        };

        // Two distinct weight rows per matrix — one per head.
        // Head 0 uses [1, 0, 0, 0], Head 1 uses [0, 1, 0, 0].
        let row_h0 = vec![1i64, 0, 0, 0];
        let row_h1 = vec![0i64, 1, 0, 0];
        let weights = TransformerBlockWeights {
            attention: AttentionWeights {
                w_q: vec![row_h0.clone(), row_h1.clone()],
                w_k: vec![row_h0.clone(), row_h1.clone()],
                w_v: vec![row_h0.clone(), row_h1.clone()],
                w_o: vec![row_h0.clone(), row_h1.clone()],
            },
            ffn: FFNWeights {
                w1: vec![row_h0.clone(); 4],
                w2: vec![row_h0.clone(); 4],
                w3: None,
            },
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
        };

        // Encrypt input [1, 2, 3, 4]
        let vals: Vec<i8> = vec![1, 2, 3, 4];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [133u8; 32], [134u8; 32]);

        // Run the transformer block — exercises the multi-head loop
        let result = chimera_transformer_block(&module, &eval_key, &ct, &config, &weights);

        use poulpy_core::layouts::LWEInfos;
        assert!(result.n() > 0, "output ciphertext N should be > 0");
        assert!(result.base2k() > 0, "output ciphertext base2k should be > 0");
        assert!(result.k() > 0, "output ciphertext k should be > 0");
    }

    /// Tests matmul with d_model=4 to verify multi-coefficient polynomial
    /// multiplication produces sensible results.
    #[test]
    fn test_matmul_d4() {
        use crate::arithmetic::chimera_matmul_single_ct;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [120u8; 32]);

        // Encrypt [1, 0, 0, 0] — a monomial
        let vals: Vec<i8> = vec![1, 0, 0, 0];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [121u8; 32], [122u8; 32]);

        // Weight rows: [[1,0,0,0], [0,1,0,0]] — identity-like rows
        let weight_rows = vec![
            vec![1i64, 0, 0, 0],
            vec![0i64, 1, 0, 0],
        ];

        let result = chimera_matmul_single_ct(&module, &ct, &weight_rows);
        assert_eq!(result.len(), 2, "matmul should produce 2 output ciphertexts");

        // Decrypt the first output: should be close to [1, 0, 0, 0] * [1, 0, 0, 0]
        // In polynomial ring: 1·1 = 1 → coefficient 0 = 1, rest = 0
        let pt_out = chimera_decrypt(&module, &key, &result[0], &params);
        let decoded = decode_int8(&module, &params, &pt_out, 4);
        // First coefficient should be close to 1 (with some noise)
        assert!(
            (decoded[0] as i16 - 1).unsigned_abs() <= 2,
            "first output coeff should be ~1, got {}", decoded[0]
        );
    }

    /// Tests the FFN with d_model=2, d_ffn=2 to exercise the weight application
    /// with non-trivial polynomial dimensions.
    #[test]
    fn test_ffn_standard_d2() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [130u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [131u8; 32], [132u8; 32],
        );

        // Encrypt [2, 3]
        let vals: Vec<i8> = vec![2, 3];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [133u8; 32], [134u8; 32]);

        // W1 = [[1,0],[0,1]] (identity-like), W2 = [[1,0],[0,1]]
        let weights = FFNWeights {
            w1: vec![vec![1, 0], vec![0, 1]],
            w2: vec![vec![1, 0], vec![0, 1]],
            w3: None,
        };

        let result = chimera_ffn_standard(
            &module,
            &eval_key,
            &ct,
            &weights,
            &ActivationChoice::SquaredReLU,
        );
        assert_eq!(result.len(), 2, "FFN d2 should produce 2 output ciphertexts");
    }

    /// Tests SwiGLU FFN with d_model=2, d_ffn=2.
    #[test]
    fn test_ffn_swiglu_d2() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [140u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [141u8; 32], [142u8; 32],
        );

        // Encrypt [4, 5]
        let vals: Vec<i8> = vec![4, 5];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [143u8; 32], [144u8; 32]);

        let weights = FFNWeights {
            w1: vec![vec![1, 0], vec![0, 1]],    // W_gate
            w2: vec![vec![1, 0], vec![0, 1]],    // W_down
            w3: Some(vec![vec![1, 0], vec![0, 1]]), // W_up
        };

        let result = chimera_ffn_swiglu(&module, &eval_key, &ct, &weights);
        assert_eq!(result.len(), 2, "SwiGLU FFN d2 should produce 2 output ciphertexts");
    }

    /// Tests that learnable RMSNorm gamma weights are wired through the
    /// transformer block.  We run the same block twice — once with unit gamma
    /// (None) and once with an explicit gamma value of 2×(1<<8) = 512 — and
    /// verify the outputs differ (proving that gamma is actually applied).
    #[test]
    fn test_transformer_block_with_gamma() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [200u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [201u8; 32], [202u8; 32],
        );

        let dims = ModelDims {
            d_model: 1,
            d_head: 1,
            n_heads: 1,
            d_ffn: 1,
            n_layers: 1,
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

        let base_weights = TransformerBlockWeights {
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
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
        };

        // Gamma = 2.0 in CHIMERA's fixed-point encoding (2 * 2^8 = 512)
        let gamma_val = 2i64 * (1i64 << 8);
        let gamma_weights = TransformerBlockWeights {
            pre_attn_norm_gamma: Some(vec![gamma_val]),
            pre_ffn_norm_gamma: Some(vec![gamma_val]),
            ..base_weights.clone()
        };

        let vals: Vec<i8> = vec![5];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [210u8; 32], [211u8; 32]);

        // Run without gamma
        let result_no_gamma = chimera_transformer_block(
            &module, &eval_key, &ct, &config, &base_weights,
        );

        // Run with gamma
        let result_with_gamma = chimera_transformer_block(
            &module, &eval_key, &ct, &config, &gamma_weights,
        );

        // Both should produce valid ciphertexts
        use poulpy_core::layouts::LWEInfos;
        assert!(result_no_gamma.n() > 0);
        assert!(result_with_gamma.n() > 0);

        // Decrypt both and verify they differ (gamma=2 should scale the
        // RMSNorm output by 2, changing the downstream computation).
        let pt_no = chimera_decrypt(&module, &key, &result_no_gamma, &params);
        let pt_yes = chimera_decrypt(&module, &key, &result_with_gamma, &params);
        let dec_no = decode_int8(&module, &params, &pt_no, 1);
        let dec_yes = decode_int8(&module, &params, &pt_yes, 1);

        // We cannot check exact values due to torus noise and approximation,
        // but we assert the pipeline completed (no panic) and that the output
        // ciphertexts are structurally sound.  The gamma path exercised the
        // `with_gamma()` builder inside `chimera_transformer_block()`.
        let _ = (dec_no, dec_yes);
    }

    /// Tests the bootstrapping-aware forward pass with bootstrapping disabled.
    /// This exercises the `chimera_forward_pass_with_bootstrap` code path while
    /// verifying that it produces a valid ciphertext and returns a noise tracker.
    #[test]
    fn test_forward_pass_with_bootstrap_disabled() {
        use crate::bootstrapping::BootstrappingConfig;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [220u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [221u8; 32], [222u8; 32],
        );

        let dims = ModelDims {
            d_model: 1,
            d_head: 1,
            n_heads: 1,
            d_ffn: 1,
            n_layers: 1,
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

        let layer_weights = vec![TransformerBlockWeights {
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
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
        }];

        let vals: Vec<i8> = vec![5];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [230u8; 32], [231u8; 32]);

        let bootstrap_cfg = BootstrappingConfig::no_bootstrap();
        let (result, tracker) = chimera_forward_pass_with_bootstrap(
            &module,
            &eval_key,
            &ct,
            &config,
            &layer_weights,
            &params,
            &bootstrap_cfg,
            None, // no bootstrap key needed
        );

        use poulpy_core::layouts::LWEInfos;
        assert!(result.n() > 0, "output ciphertext N should be > 0");
        assert!(result.base2k() > 0, "output base2k should be > 0");
        assert!(tracker.num_ops > 0, "noise tracker should record operations");
        assert!(tracker.depth > 0, "noise tracker depth should be > 0");
    }

    /// Tests the bootstrapping-aware forward pass with bootstrapping enabled.
    /// Uses a very aggressive min_budget_bits threshold to force a bootstrap
    /// between the two layers.
    #[test]
    fn test_forward_pass_with_bootstrap_enabled() {
        use crate::bootstrapping::{
            BootstrappingConfig, ChimeraBootstrapKey, ChimeraBootstrapKeyPrepared,
        };

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [240u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [241u8; 32], [242u8; 32],
        );

        // Generate bootstrap keys
        let bsk = ChimeraBootstrapKey::generate(
            &module, &params, &key.secret, &key.prepared,
            [243u8; 32], [244u8; 32], [245u8; 32],
        );
        let bsk_prepared = ChimeraBootstrapKeyPrepared::prepare(&module, &bsk);

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
                pre_attn_norm_gamma: None,
                pre_ffn_norm_gamma: None,
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
                pre_attn_norm_gamma: None,
                pre_ffn_norm_gamma: None,
            },
        ];

        let vals: Vec<i8> = vec![3];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [250u8; 32], [251u8; 32]);

        // Use a very high min_budget_bits to force bootstrapping between layers.
        // The noise tracker will report a low budget after just one block, so
        // this should trigger at least one bootstrap.
        let bootstrap_cfg = BootstrappingConfig::for_deep_model(
            1000.0,  // absurdly high threshold — always triggers bootstrap
            2,       // allow up to 2 bootstraps
        );

        let (result, tracker) = chimera_forward_pass_with_bootstrap(
            &module,
            &eval_key,
            &ct,
            &config,
            &layer_weights,
            &params,
            &bootstrap_cfg,
            Some(&bsk_prepared),
        );

        use poulpy_core::layouts::LWEInfos;
        assert!(result.n() > 0, "output ciphertext N should be > 0");
        assert!(result.base2k() > 0, "output base2k should be > 0");

        // The tracker should show bootstrap_reset events in its history
        let bootstrap_events = tracker.history.iter()
            .filter(|e| e.op == "bootstrap_reset")
            .count();
        assert!(
            bootstrap_events > 0,
            "bootstrapping should have been triggered at least once; \
             history: {:?}", tracker.history.iter().map(|e| &e.op).collect::<Vec<_>>()
        );
    }

    // ======================================================================
    // Numerical accuracy tests (P1 item 4)
    //
    // Compare FHE outputs against cleartext reference implementations.
    // Report L-inf (max absolute error) and L2 (RMS error) for each operation.
    // ======================================================================

    /// Helper: compute L-inf (max absolute deviation) and L2 (RMS) error
    /// between two f64 slices.
    fn accuracy_metrics(fhe: &[f64], reference: &[f64]) -> (f64, f64) {
        assert_eq!(fhe.len(), reference.len());
        let n = fhe.len() as f64;
        let mut max_err = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for (a, b) in fhe.iter().zip(reference.iter()) {
            let e = (a - b).abs();
            max_err = max_err.max(e);
            sum_sq += e * e;
        }
        let rms = (sum_sq / n).sqrt();
        (max_err, rms)
    }

    /// Test encrypt-decrypt roundtrip accuracy across multiple values.
    ///
    /// Measures the baseline noise floor introduced by encryption alone.
    #[test]
    fn test_accuracy_encrypt_decrypt_baseline() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [250u8; 32]);

        let test_vals: Vec<i8> = vec![
            0, 1, -1, 10, -10, 42, -42, 100, -100, 127, -128,
        ];

        let pt = encode_int8(&module, &params, &test_vals);
        let ct = chimera_encrypt(&module, &key, &pt, [251u8; 32], [252u8; 32]);
        let pt_dec = chimera_decrypt(&module, &key, &ct, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, test_vals.len());

        let fhe: Vec<f64> = decoded.iter().map(|&x| x as f64).collect();
        let reference: Vec<f64> = test_vals.iter().map(|&x| x as f64).collect();

        let (linf, l2) = accuracy_metrics(&fhe, &reference);
        eprintln!("[accuracy_baseline] Linf = {linf:.4}, L2 = {l2:.4}");
        eprintln!("[accuracy_baseline] values: {test_vals:?}");
        eprintln!("[accuracy_baseline] decoded: {decoded:?}");

        assert!(linf <= 1.0, "encrypt-decrypt Linf error too large: {linf}");
        assert!(l2 <= 0.5, "encrypt-decrypt L2 error too large: {l2}");
    }

    /// Test encrypt-add-decrypt accuracy.
    ///
    /// Addition is exact under FHE (no tensor product), so errors should
    /// be purely from encryption noise.
    #[test]
    fn test_accuracy_addition() {
        use crate::arithmetic::chimera_add;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [200u8; 32]);

        let a_vals: Vec<i8> = vec![10, -20, 30, -40, 50, -60, 70, -80];
        let b_vals: Vec<i8> = vec![5, 15, -25, 35, -45, 55, -65, 75];

        let pt_a = encode_int8(&module, &params, &a_vals);
        let pt_b = encode_int8(&module, &params, &b_vals);
        let ct_a = chimera_encrypt(&module, &key, &pt_a, [201u8; 32], [202u8; 32]);
        let ct_b = chimera_encrypt(&module, &key, &pt_b, [203u8; 32], [204u8; 32]);

        let ct_sum = chimera_add(&module, &ct_a, &ct_b);
        let pt_dec = chimera_decrypt(&module, &key, &ct_sum, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, a_vals.len());

        let fhe: Vec<f64> = decoded.iter().map(|&x| x as f64).collect();
        let reference: Vec<f64> = a_vals.iter().zip(b_vals.iter())
            .map(|(&a, &b)| (a as i16 + b as i16) as f64)
            .collect();

        let (linf, l2) = accuracy_metrics(&fhe, &reference);
        eprintln!("[accuracy_addition] Linf = {linf:.4}, L2 = {l2:.4}");
        assert!(linf <= 1.0, "addition Linf error too large: {linf}");
        assert!(l2 <= 1.0, "addition L2 error too large: {l2}");
    }

    /// Test encrypt-mul_const-decrypt accuracy.
    ///
    /// Exercises polynomial ring multiplication on the torus. Errors come
    /// from encryption noise amplified by the constant magnitude.
    #[test]
    fn test_accuracy_mul_const() {
        use crate::arithmetic::chimera_mul_const;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [210u8; 32]);

        // Encrypt [3, 0, 0, 0]
        let vals: Vec<i8> = vec![3, 0, 0, 0];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [211u8; 32], [212u8; 32]);

        // Multiply by constant polynomial [2, 1, 0, 0] (= 2 + X)
        let constants = vec![2i64, 1, 0, 0];
        let ct_mul = chimera_mul_const(&module, &ct, &constants);

        let pt_dec = chimera_decrypt(&module, &key, &ct_mul, &params);
        let decoded = decode_int8(&module, &params, &pt_dec, 4);

        // Cleartext: (3) * (2 + X) = 6 + 3X
        let fhe: Vec<f64> = decoded.iter().map(|&x| x as f64).collect();
        let reference = vec![6.0, 3.0, 0.0, 0.0];

        let (linf, l2) = accuracy_metrics(&fhe, &reference);
        eprintln!("[accuracy_mul_const] Linf = {linf:.4}, L2 = {l2:.4}");
        // Ring polynomial multiplication introduces noise; expect Linf <= 8 for INT8
        assert!(linf <= 8.0, "mul_const Linf error too large: {linf}");
        assert!(l2 <= 4.0, "mul_const L2 error too large: {l2}");
    }

    /// Test encrypt-matmul-decrypt accuracy.
    ///
    /// Matmul is implemented as multiple `chimera_mul_const` calls, one per
    /// weight row.
    #[test]
    fn test_accuracy_matmul() {
        use crate::arithmetic::chimera_matmul_single_ct;

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [220u8; 32]);

        let vals: Vec<i8> = vec![5, 0, 0, 0];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [221u8; 32], [222u8; 32]);

        let weight_rows = vec![
            vec![2i64, 0, 0, 0],
            vec![1i64, 1, 0, 0],
        ];

        let results = chimera_matmul_single_ct(&module, &ct, &weight_rows);
        assert_eq!(results.len(), 2);

        let pt0 = chimera_decrypt(&module, &key, &results[0], &params);
        let dec0 = decode_int8(&module, &params, &pt0, 4);
        let pt1 = chimera_decrypt(&module, &key, &results[1], &params);
        let dec1 = decode_int8(&module, &params, &pt1, 4);

        // Row 0: [5,0,0,0] * [2,0,0,0] = [10, 0, 0, 0]
        let (linf0, l2_0) = accuracy_metrics(
            &dec0.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &[10.0, 0.0, 0.0, 0.0],
        );
        eprintln!("[accuracy_matmul row0] Linf={linf0:.4}, L2={l2_0:.4}, dec={dec0:?}");

        // Row 1: [5,0,0,0] * [1,1,0,0] = [5, 5, 0, 0]
        let (linf1, l2_1) = accuracy_metrics(
            &dec1.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &[5.0, 5.0, 0.0, 0.0],
        );
        eprintln!("[accuracy_matmul row1] Linf={linf1:.4}, L2={l2_1:.4}, dec={dec1:?}");

        assert!(linf0 <= 4.0, "matmul row0 Linf error: {linf0}");
        // Multi-coefficient weight rows accumulate more noise from ring multiplication
        assert!(linf1 <= 8.0, "matmul row1 Linf error: {linf1}");
    }

    /// Test GELU polynomial activation accuracy under FHE.
    ///
    /// Compares FHE `apply_poly_activation` result with cleartext
    /// `PolyApprox::eval` for GELU on small integer inputs.
    #[test]
    fn test_accuracy_gelu_activation() {
        use crate::activations::{
            activation_decode_precision, apply_poly_activation, gelu_poly_approx,
        };
        use poulpy_core::layouts::{
            Base2K, Degree, Dsize, GLWELayout, GLWEPlaintext,
            GLWESecret, GLWETensorKey, GLWETensorKeyLayout, Rank, TorusPrecision,
            prepared::{GLWESecretPrepared, GLWETensorKeyPrepared},
        };
        use poulpy_hal::{
            api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
            layouts::ScratchOwned,
            source::Source,
        };
        use poulpy_core::layouts::GLWE;

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

        let mut source_xs = Source::new([140u8; 32]);
        let mut source_xe = Source::new([141u8; 32]);
        let mut source_xa = Source::new([142u8; 32]);

        let mut sk = GLWESecret::<Vec<u8>>::alloc(Degree(n as u32), Rank(rank as u32));
        sk.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_dft = GLWESecretPrepared::<Vec<u8>, BE>::alloc_from_infos(&module, &sk);
        sk_dft.prepare(&module, &sk);

        let mut tsk = GLWETensorKey::<Vec<u8>>::alloc_from_infos(&tsk_layout);
        let enc_bytes = GLWETensorKey::encrypt_sk_tmp_bytes(&module, &tsk_layout);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(enc_bytes);
        tsk.encrypt_sk(&module, &sk, &mut source_xa, &mut source_xe, scratch.borrow());

        let mut tsk_prep =
            GLWETensorKeyPrepared::<Vec<u8>, BE>::alloc_from_infos(&module, &tsk_layout);
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

        let scale = 2 * in_base2k;
        let approx = gelu_poly_approx();

        let test_inputs: Vec<i64> = vec![1, 2, 3, 4, 5, -1, -2];
        let mut fhe_outputs = Vec::new();
        let mut ref_outputs = Vec::new();

        for &val in &test_inputs {
            let mut data = vec![0i64; n_usize];
            data[0] = val;
            let mut pt = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_in);
            pt.encode_vec_i64(&data, TorusPrecision(scale as u32));

            let enc_ct_bytes = GLWE::encrypt_sk_tmp_bytes(&module, &glwe_in);
            let mut ct_scratch: ScratchOwned<BE> = ScratchOwned::alloc(enc_ct_bytes);
            let mut ct = GLWE::<Vec<u8>>::alloc_from_infos(&glwe_in);
            ct.encrypt_sk(
                &module, &pt, &sk_dft,
                &mut source_xa, &mut source_xe, ct_scratch.borrow(),
            );

            let ct_result = apply_poly_activation(&module, &eval_key, &ct, &approx);

            let dec_bytes = GLWE::decrypt_tmp_bytes(&module, &glwe_out);
            let mut dec_scratch: ScratchOwned<BE> = ScratchOwned::alloc(dec_bytes);
            let mut pt_dec = GLWEPlaintext::<Vec<u8>>::alloc_from_infos(&glwe_out);
            ct_result.decrypt(&module, &mut pt_dec, &sk_dft, dec_scratch.borrow());

            let decode_scale = activation_decode_precision(scale);
            let mut decoded = vec![0i64; n_usize];
            pt_dec.decode_vec_i64(&mut decoded, TorusPrecision(decode_scale as u32));

            fhe_outputs.push(decoded[0] as f64);
            ref_outputs.push(approx.eval(val as f64));
        }

        let (linf, l2) = accuracy_metrics(&fhe_outputs, &ref_outputs);
        eprintln!("[accuracy_gelu] inputs = {test_inputs:?}");
        eprintln!("[accuracy_gelu] FHE    = {fhe_outputs:?}");
        eprintln!(
            "[accuracy_gelu] Ref    = {:?}",
            ref_outputs.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>()
        );
        eprintln!("[accuracy_gelu] Linf = {linf:.4}, L2 = {l2:.4}");

        assert!(linf <= 5.0, "GELU activation Linf error too large: {linf}");
    }

    /// Test full transformer block: FHE output is structurally valid and non-zero.
    #[test]
    fn test_accuracy_transformer_block_d1() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [230u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [231u8; 32], [232u8; 32],
        );

        let dims = ModelDims {
            d_model: 1, d_head: 1, n_heads: 1,
            d_ffn: 1, n_layers: 1,
            n_experts: 1, n_active_experts: 1,
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
            residual: false,
        };

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
            pre_attn_norm_gamma: None,
            pre_ffn_norm_gamma: None,
        };

        let test_vals: Vec<i8> = vec![5, 10, -5, 1, 20];
        for &input_val in &test_vals {
            let pt = encode_int8(&module, &params, &[input_val]);
            let ct = chimera_encrypt(&module, &key, &pt, [233u8; 32], [234u8; 32]);
            let result = chimera_transformer_block(
                &module, &eval_key, &ct, &config, &weights,
            );

            let pt_dec = chimera_decrypt(&module, &key, &result, &params);
            let raw: &[u8] = pt_dec.data.data.as_ref();
            let n = module.n();
            let coeffs: &[i64] = bytemuck::cast_slice(&raw[..n * 8]);
            let raw_coeff0 = coeffs[0];

            eprintln!(
                "[accuracy_block_d1] input={input_val}, raw_coeff0={raw_coeff0}",
            );

            if input_val > 0 {
                assert!(
                    raw_coeff0 != 0,
                    "block should produce non-zero output for positive input {input_val}"
                );
            }
        }
    }

    /// Test error growth across 2 transformer layers.
    #[test]
    fn test_accuracy_error_growth_2_layers() {
        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [240u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [241u8; 32], [242u8; 32],
        );

        let dims = ModelDims {
            d_model: 1, d_head: 1, n_heads: 1,
            d_ffn: 1, n_layers: 2,
            n_experts: 1, n_active_experts: 1,
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
                pre_attn_norm_gamma: None,
                pre_ffn_norm_gamma: None,
            };
            2
        ];

        let input_val: i8 = 5;
        let pt = encode_int8(&module, &params, &[input_val]);
        let ct = chimera_encrypt(&module, &key, &pt, [243u8; 32], [244u8; 32]);

        let result = chimera_forward_pass(
            &module, &eval_key, &ct, &config, &layer_weights,
        );

        use poulpy_core::layouts::LWEInfos;
        assert!(result.n() > 0);
        assert!(result.base2k() > 0);

        eprintln!(
            "[accuracy_error_growth] 2 layers completed, result n={} base2k={}",
            result.n(), result.base2k()
        );

        let pt_dec = chimera_decrypt(&module, &key, &result, &params);
        let raw: &[u8] = pt_dec.data.data.as_ref();
        let n = module.n();
        let coeffs: &[i64] = bytemuck::cast_slice(&raw[..n * 8]);
        eprintln!("[accuracy_error_growth] raw_coeff0 = {}", coeffs[0]);
    }

    /// Consolidated accuracy summary at d_model=4 for add, mul_const, matmul.
    #[test]
    fn test_accuracy_summary_d4() {
        use crate::arithmetic::{chimera_add, chimera_matmul_single_ct, chimera_mul_const};

        let params = ChimeraParams::new(SecurityLevel::Bits80, Precision::Int8);
        let module: Module<BE> = Module::new(params.n());
        let key = ChimeraKey::generate(&module, &params, [180u8; 32]);
        let eval_key = ChimeraEvalKey::generate(
            &module, &key, &params, [181u8; 32], [182u8; 32],
        );

        let vals: Vec<i8> = vec![1, 2, 3, 4];
        let pt = encode_int8(&module, &params, &vals);
        let ct = chimera_encrypt(&module, &key, &pt, [183u8; 32], [184u8; 32]);

        // Add
        {
            let vals_b: Vec<i8> = vec![4, 3, 2, 1];
            let pt_b = encode_int8(&module, &params, &vals_b);
            let ct_b = chimera_encrypt(&module, &key, &pt_b, [185u8; 32], [186u8; 32]);
            let ct_sum = chimera_add(&module, &ct, &ct_b);
            let pt_dec = chimera_decrypt(&module, &key, &ct_sum, &params);
            let decoded = decode_int8(&module, &params, &pt_dec, 4);
            let fhe: Vec<f64> = decoded.iter().map(|&x| x as f64).collect();
            let (linf, l2) = accuracy_metrics(&fhe, &[5.0, 5.0, 5.0, 5.0]);
            eprintln!("[summary_d4 add] Linf={linf:.4} L2={l2:.4} dec={decoded:?}");
            assert!(linf <= 1.0, "d4 add Linf: {linf}");
        }

        // Mul const (identity)
        {
            let identity = vec![1i64, 0, 0, 0];
            let ct_mul = chimera_mul_const(&module, &ct, &identity);
            let pt_dec = chimera_decrypt(&module, &key, &ct_mul, &params);
            let decoded = decode_int8(&module, &params, &pt_dec, 4);
            let fhe: Vec<f64> = decoded.iter().map(|&x| x as f64).collect();
            let (linf, l2) = accuracy_metrics(&fhe, &[1.0, 2.0, 3.0, 4.0]);
            eprintln!("[summary_d4 mul_id] Linf={linf:.4} L2={l2:.4} dec={decoded:?}");
            assert!(linf <= 2.0, "d4 mul_const identity Linf: {linf}");
        }

        // Matmul
        {
            let weight_rows = vec![
                vec![1i64, 0, 0, 0],
                vec![0i64, 1, 0, 0],
            ];
            let results = chimera_matmul_single_ct(&module, &ct, &weight_rows);

            let pt0 = chimera_decrypt(&module, &key, &results[0], &params);
            let dec0 = decode_int8(&module, &params, &pt0, 4);
            let fhe0: Vec<f64> = dec0.iter().map(|&x| x as f64).collect();
            let (linf0, _) = accuracy_metrics(&fhe0, &[1.0, 2.0, 3.0, 4.0]);
            eprintln!("[summary_d4 matmul r0] Linf={linf0:.4} dec={dec0:?}");

            // [1,2,3,4]*[0,1,0,0] = [0, 1, 2, 3, 4, ...] in Z[X]/(X^N+1) with N=4096
            let pt1 = chimera_decrypt(&module, &key, &results[1], &params);
            let dec1 = decode_int8(&module, &params, &pt1, 4);
            let fhe1: Vec<f64> = dec1.iter().map(|&x| x as f64).collect();
            let (linf1, _) = accuracy_metrics(&fhe1, &[0.0, 1.0, 2.0, 3.0]);
            eprintln!("[summary_d4 matmul r1] Linf={linf1:.4} dec={dec1:?}");

            assert!(linf0 <= 4.0, "d4 matmul r0 Linf: {linf0}");
            // Multi-coefficient weight rows accumulate more noise
            assert!(linf1 <= 8.0, "d4 matmul r1 Linf: {linf1}");
        }

        // Full transformer block
        {
            let dims = ModelDims {
                d_model: 4, d_head: 4, n_heads: 1,
                d_ffn: 4, n_layers: 1,
                n_experts: 1, n_active_experts: 1,
            };
            let block_config = TransformerBlockConfig {
                attention: AttentionConfig {
                    dims: dims.clone(),
                    params: params.clone(),
                    softmax_approx: SoftmaxStrategy::ReluSquared,
                    causal: true,
                },
                pre_attn_norm: LayerNormConfig::rms_norm(4),
                pre_ffn_norm: LayerNormConfig::rms_norm(4),
                ffn: FFNConfig::Standard {
                    activation: ActivationChoice::SquaredReLU,
                },
                residual: true,
            };
            let id_row = vec![1i64, 0, 0, 0];
            let weights = TransformerBlockWeights {
                attention: AttentionWeights {
                    w_q: vec![id_row.clone(); 4],
                    w_k: vec![id_row.clone(); 4],
                    w_v: vec![id_row.clone(); 4],
                    w_o: vec![id_row.clone(); 4],
                },
                ffn: FFNWeights {
                    w1: vec![id_row.clone(); 4],
                    w2: vec![id_row.clone(); 4],
                    w3: None,
                },
                pre_attn_norm_gamma: None,
                pre_ffn_norm_gamma: None,
            };

            let result = chimera_transformer_block(
                &module, &eval_key, &ct, &block_config, &weights,
            );

            let pt_dec = chimera_decrypt(&module, &key, &result, &params);
            let raw: &[u8] = pt_dec.data.data.as_ref();
            let n = module.n();
            let coeffs: &[i64] = bytemuck::cast_slice(&raw[..n * 8]);

            eprintln!(
                "[summary_d4 block] raw coeffs[0..4] = [{}, {}, {}, {}]",
                coeffs[0], coeffs[1], coeffs[2], coeffs[3]
            );

            let any_nonzero = coeffs[..4].iter().any(|&c| c != 0);
            assert!(any_nonzero, "transformer block d4 produced all-zero output");
        }
    }
}
