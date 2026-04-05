use std::process;

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
    let mut args = std::env::args().skip(1);
    let user_prompt = args
        .next()
        .unwrap_or_else(|| "What is 2+2? Answer with one token.".to_string());
    let scale_bits: Vec<usize> = args
        .next()
        .map(|s| {
            s.split(',')
                .filter(|part| !part.trim().is_empty())
                .map(|part| part.trim().parse::<usize>().expect("invalid scale bits"))
                .collect()
        })
        .unwrap_or_else(|| vec![0, 1, 2, 3, 4, 5, 6]);

    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(30),
        trunc_d_model: Some(576),
        trunc_d_ffn: Some(1536),
        num_heads: Some(9),
        num_kv_heads: Some(3),
        softmax_strategy: SoftmaxStrategy::ReluSquared,
        apply_final_norm: true,
        max_new_tokens: 2,
        ..InferenceConfig::default()
    };

    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    let prompt = pipeline.format_chat_prompt(
        Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face."),
        &user_prompt,
    );
    let diag = pipeline.diagnose_prompt_gate_activation_with_scales(&prompt, &scale_bits)?;

    println!("prompt: {:?}", diag.prompt);
    println!("tokenized_prompt_len: {}", diag.prompt_tokens.len());
    println!(
        "message_domain: log_message_modulus={} range=[{}, {}]",
        diag.log_message_modulus, diag.message_range_min, diag.message_range_max
    );
    println!(
        "gate_input_range: min={:.6} max={:.6} rounded_min={} rounded_max={} overflow_dims={}/{}",
        diag.gate_min, diag.gate_max, diag.gate_rounded_min, diag.gate_rounded_max, diag.overflow_dims, diag.total_dims
    );
    for row in diag.rows {
        println!(
            "variant={} scale_bits={} overflow={}: gate_act(L-inf={:.6} L2={:.6} MAE={:.6}) hidden(L-inf={:.6} L2={:.6} MAE={:.6}) mlp_out(L-inf={:.6} L2={:.6} MAE={:.6})",
            row.variant,
            row.input_scale_bits,
            row.overflow_dims,
            row.gate_act_linf,
            row.gate_act_l2,
            row.gate_act_mae,
            row.hidden_linf,
            row.hidden_l2,
            row.hidden_mae,
            row.mlp_out_linf,
            row.mlp_out_l2,
            row.mlp_out_mae,
        );
    }

    Ok(())
}
