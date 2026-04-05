use poupy_FHE_LLM::attention::SoftmaxStrategy;
use poupy_FHE_LLM::inference::VEC_EFFECTIVE_DECODE_SCALE;
use poupy_FHE_LLM::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use poupy_FHE_LLM::params::{Precision, SecurityLevel};
use std::time::Instant;

const MODEL_PATH: &str = "/home/dev/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InferenceConfig {
        security: SecurityLevel::Bits128,
        precision: Precision::Fp16,
        num_layers: Some(1),
        trunc_d_model: Some(32),
        trunc_d_ffn: Some(128),
        num_heads: Some(1),
        num_kv_heads: Some(1),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        fhe_silu_log_msg_mod: Some(18),
        fhe_identity_log_msg_mod: Some(18),
        fhe_frequent_bootstrap: true,
        ..InferenceConfig::default()
    };

    let load_start = Instant::now();
    eprintln!("[test] Loading pipeline with FP16 and frequent bootstrap...");
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::tinyllama_1_1b(), config)?;
    let load_elapsed = load_start.elapsed();
    eprintln!("[test] Loaded in {:.2}s", load_elapsed.as_secs_f64());

    let prompt = "2+2=";
    let prompt_tokens = pipeline.tokenize(prompt)?;
    let last_token = *prompt_tokens.last().unwrap();
    let start = Instant::now();
    let result = pipeline.step_refreshed(last_token)?;
    let elapsed = start.elapsed();

    eprintln!("[test] FHE step {:.2}s", elapsed.as_secs_f64());
    eprintln!("[test] Token ID (FHE): {}", result.token_id);
    eprintln!("[test] Top-5 logits: {:?}", result.top_logits);
    eprintln!(
        "[test] Hidden state range: [{:?}, {:?}]",
        result.hidden_state.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
        result.hidden_state.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(),
    );

    // Diagnostics: before scaling, after scaling, after LUT
    let params = pipeline.params();
    let before_scaling = pipeline.diagnose_first_block_stage_for_token_at_precision(last_token, "input", params.k_pt.0);
    let after_scaling = pipeline.refreshed_hidden_range_at_precision(last_token, VEC_EFFECTIVE_DECODE_SCALE);
    let after_lut = pipeline.diagnose_first_block_stage_for_token_at_precision(last_token, "ffn_out", VEC_EFFECTIVE_DECODE_SCALE);
    eprintln!("[test] Diagnostics:");
    eprintln!(
        "  Max ciphertext value before scaling (input): {} (stage: {})",
        before_scaling.max, before_scaling.stage
    );
    eprintln!(
        "  Max ciphertext value after scaling (refreshed final): {} (stage: {})",
        after_scaling.max, after_scaling.stage
    );
    eprintln!(
        "  Max ciphertext value after LUT (ffn_out): {} (stage: {})",
        after_lut.max, after_lut.stage
    );

    // Compute max_ct_val after refresh
    let raw_hidden = pipeline.raw_hidden_state_for_token(last_token);
    let max_ct_val_after_refresh = raw_hidden.iter().map(|&v| (v as f64).abs()).fold(0.0f64, f64::max);
    eprintln!("[test] Max CT val after refresh: {}", max_ct_val_after_refresh);

    // Verify top-5 logits are non-zero
    let nonzero_logits: Vec<bool> = result.top_logits.iter().map(|(_, logit)| *logit != 0).collect();
    let all_nonzero = nonzero_logits.iter().all(|&b| b);
    if all_nonzero {
        eprintln!("[test] All top-5 logits are non-zero: ✅");
    } else {
        eprintln!("[test] Some top-5 logits are zero: ❌");
    }

    // Log refresh scaling factor
    eprintln!(
        "[test] Refresh scaling factor: {} bits (effective decode scale), SiLU LUT precision: {} bits, identity LUT precision: {} bits",
        VEC_EFFECTIVE_DECODE_SCALE,
        config.fhe_silu_log_msg_mod.unwrap_or(11),
        config.fhe_identity_log_msg_mod.unwrap_or(11)
    );

    // Compare with plaintext shadow
    let plaintext_result = pipeline.plaintext_step_refreshed_multilayer(last_token);
    eprintln!(
        "[test] Plaintext shadow token ID: {}, token text: {:?}",
        plaintext_result.token_id, plaintext_result.token_text
    );
    eprintln!("  Token match: {}", result.token_id == plaintext_result.token_id);

    Ok(())
}
