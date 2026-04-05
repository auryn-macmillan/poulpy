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
    let prompts: Vec<String> = std::env::args().skip(1).collect();
    let prompts = if prompts.is_empty() {
        vec![
            "What is 2+2? Answer with one token.".to_string(),
            "Complete the sequence with one token: red, blue, red, blue,".to_string(),
            "The capital of France is".to_string(),
            "Answer with one token: yes or no. Is water wet?".to_string(),
            "Finish with one token: 1, 2, 3,".to_string(),
        ]
    } else {
        prompts
    };

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

    for user_prompt in prompts {
        let prompt = pipeline.format_chat_prompt(
            Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face."),
            &user_prompt,
        );
        let result = pipeline.compare_prompt_output_impact(&prompt)?;

        println!("prompt: {:?}", user_prompt);
        println!(
            "exact: token={} text={:?} runner_up={} {:?} margin={:.6}",
            result.exact.token_id,
            result.exact.token_text,
            result.exact_top2_token_id,
            result.exact_top2_token_text,
            result.exact_top_margin_f64,
        );
        for row in result.rows {
            println!(
                "  variant={} match={} token={} text={:?} rank={} hidden_mae={:.6} pred_logit={:.6} exact_logit={:.6}",
                row.variant,
                row.token_match,
                row.token_id,
                row.token_text,
                row.exact_token_rank,
                row.hidden_mae,
                row.predicted_logit_f64,
                row.exact_token_logit_f64,
            );
        }
        println!();
    }

    Ok(())
}
