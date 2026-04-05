/// Minimal test to isolate where 1-layer SmolLM2 hangs.
use std::process;
use std::time::Instant;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = "2+2=";

    let config = InferenceConfig {
        security: SecurityLevel::Bits100,
        precision: Precision::Int8,
        num_layers: Some(1),
        trunc_d_model: None,
        trunc_d_ffn: None,
        num_heads: Some(9),
        num_kv_heads: Some(3),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: false,
        max_new_tokens: 1,
        key_seed: [42u8; 32],
        eval_seed_a: [43u8; 32],
        eval_seed_b: [44u8; 32],
    };

    eprintln!("[1] Loading model...");
    let load_start = Instant::now();
    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    eprintln!("[1] Loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    eprintln!("[2] Tokenizing prompt...");
    let prompt_tokens = pipeline.tokenize(prompt)?;
    eprintln!("[2] Tokens: {:?}\n", prompt_tokens);

    eprintln!("[3] Encrypting input (embedding)...");
    let encrypt_start = Instant::now();
    let input = pipeline.encrypt_input(prompt_tokens[0])?;
    eprintln!(
        "[3] Encrypted in {:.1}s, {} cts",
        encrypt_start.elapsed().as_secs_f64(),
        input.len()
    );

    eprintln!("[4] Running attention (QKV projection)...");
    let attn_start = Instant::now();
    let hidden_after_attn = pipeline.fhe_forward_attention_only(&input);
    eprintln!(
        "[4] Attention in {:.1}s, hidden shape: {}",
        attn_start.elapsed().as_secs_f64(),
        hidden_after_attn.len()
    );

    eprintln!("[5] Running FFN...");
    let ffn_start = Instant::now();
    let hidden_after_ffn = pipeline.fhe_forward_ffn_only(&hidden_after_attn);
    eprintln!("[5] FFN in {:.1}s", ffn_start.elapsed().as_secs_f64());

    eprintln!("[6] Decrypting output...");
    let decrypt_start = Instant::now();
    let hidden = pipeline.decrypt_hidden(&hidden_after_ffn)?;
    eprintln!("[6] Decrypted in {:.1}s", decrypt_start.elapsed().as_secs_f64());

    eprintln!("[7] Running LM head...");
    let lm_start = Instant::now();
    let logits = pipeline.cleartext_lm_head_forward(&hidden)?;
    eprintln!("[7] LM head in {:.1}s", lm_start.elapsed().as_secs_f64());

    eprintln!("[8] Argmax...");
    let max_idx = logits.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0;
    eprintln!(
        "[8] Max logit idx: {}, token: {}",
        max_idx,
        pipeline.decode_token(max_idx as u32).unwrap_or("??".to_string())
    );

    Ok(())
}
