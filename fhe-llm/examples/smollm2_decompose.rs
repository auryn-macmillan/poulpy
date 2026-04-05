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

fn parse_softmax(arg: Option<String>) -> SoftmaxStrategy {
    match arg.as_deref() {
        Some("poly") | Some("poly4") | Some("polynomial") => SoftmaxStrategy::PolynomialDeg4,
        Some("linear") => SoftmaxStrategy::Linear,
        Some("relu2") | Some("relu_squared") | None => SoftmaxStrategy::ReluSquared,
        Some(other) => panic!("unknown softmax strategy: {other}"),
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let user_prompt = args
        .next()
        .unwrap_or_else(|| "What is 2+2? Answer with one token.".to_string());
    let softmax_strategy = parse_softmax(args.next());

    let config = InferenceConfig {
        security: SecurityLevel::Bits80,
        precision: Precision::Int8,
        num_layers: Some(30),
        trunc_d_model: Some(576),
        trunc_d_ffn: Some(1536),
        num_heads: Some(9),
        num_kv_heads: Some(3),
        softmax_strategy,
        apply_final_norm: true,
        max_new_tokens: 2,
        ..InferenceConfig::default()
    };

    let pipeline = InferencePipeline::load(MODEL_PATH, TOKENIZER_PATH, ModelSpec::smollm2_135m_instruct(), config)?;
    let prompt = pipeline.format_chat_prompt(
        Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face."),
        &user_prompt,
    );

    let comparison = pipeline.compare_prompt_plaintext_paths(&prompt)?;

    println!("prompt: {:?}", user_prompt);
    println!("tokenized_prompt_len: {}", comparison.prompt_tokens.len());
    println!();
    println!(
        "exact_loaded_weights: token={} {:?} top5={:?}",
        comparison.exact.token_id, comparison.exact.token_text, comparison.exact.top_logits
    );
    println!(
        "refreshed__FHE_LLM_plain: token={} {:?} top5={:?}",
        comparison.refreshed.token_id, comparison.refreshed.token_text, comparison.refreshed.top_logits
    );
    println!("token_match: {}", comparison.token_match);
    println!(
        "refreshed_vs_exact_hidden: L-inf={:.3} L2={:.3} MAE={:.3}",
        comparison.refreshed_vs_exact.0, comparison.refreshed_vs_exact.1, comparison.refreshed_vs_exact.2
    );

    Ok(())
}
