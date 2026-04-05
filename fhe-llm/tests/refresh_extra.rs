#[test]
fn extra_refresh_preserves_ciphertext_layout() {
    use poulpy_FHE_LLM::inference::{InferenceConfig, InferencePipeline, ModelSpec};
    use poulpy_FHE_LLM::params::{ModelDims, Precision, SecurityLevel};
    use std::sync::Arc;

    // Minimal 1-layer spec just to get a pipeline
    let spec = ModelSpec {
        dims: ModelDims {
            d_model: 128,
            d_head: 32,
            n_heads: 4,
            n_kv_heads: 2,
            d_ffn: 512,
            n_layers: 1,
            n_experts: 1,
            n_active_experts: 1,
        },
        embed_name: "model.embed_tokens.weight".to_string(),
        lm_head_name: "model.embed_tokens.weight".to_string(),
        final_norm_name: "model.norm.weight".to_string(),
        rope_theta: 10000.0,
        max_seq_len: 2048,
        bos_token_id: 1,
        eos_token_id: 2,
    };

    let cfg = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(1),
        trunc_d_model: Some(128),
        trunc_d_ffn: Some(512),
        num_heads: Some(4),
        num_kv_heads: Some(2),
        softmax_strategy: poulpy_FHE_LLM::attention::SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 1,
        key_seed: [0; 32],
        eval_seed_a: [0; 32],
        eval_seed_b: [0; 32],
        fhe_frequent_bootstrap: false,
        fhe_extra_refresh: false,
        fhe_silu_log_msg_mod: Some(20),
        fhe_identity_log_msg_mod: Some(20),
        ..InferenceConfig::default()
    };

    let pipeline = InferencePipeline::load(
        "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors",
        "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json",
        spec,
        cfg,
    )
    .expect("pipeline load");

    // encrypt a tiny vector (4 ciphertexts)
    let ct_vec = pipeline.encrypt_vec(vec![1u64.into(), 2, 3, 4]);

    // apply the *only* refresh that the extra‑refresh false path performs
    let ct_refreshed = pipeline.refresh_vec(&ct_vec, true /* after_residual */).expect("refresh");

    // layout must stay identical
    assert_eq!(ct_refreshed.base2k(), ct_vec.base2k());
    assert_eq!(ct_refreshed.k(), ct_vec.k());
}
