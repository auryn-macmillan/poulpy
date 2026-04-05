use std::process;

use fhe_llm::attention::SoftmaxStrategy;
use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};

const MODEL_PATH: &str = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
const TOKENIZER_PATH: &str =
    "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

fn parse_softmax(arg: Option<String>) -> SoftmaxStrategy {
    match arg.as_deref() {
        Some("poly") | Some("poly4") | Some("polynomial") => SoftmaxStrategy::PolynomialDeg4,
        Some("linear") => SoftmaxStrategy::Linear,
        Some("relu2") | Some("relu_squared") | None => SoftmaxStrategy::ReluSquared,
        Some(other) => panic!("unknown softmax strategy: {other}"),
    }
}

fn parse_precisions(arg: Option<String>) -> Vec<u32> {
    arg.map(|s| {
        s.split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u32>().expect("invalid precision"))
            .collect()
    })
    .unwrap_or_else(|| vec![4, 6, 8, 10, 12])
}

fn parse_mode(arg: Option<String>) -> bool {
    matches!(arg.as_deref(), Some("exact-softmax") | Some("exact_softmax") | Some("exact"))
}

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
    let softmax_strategy = parse_softmax(args.next());
    let precisions = parse_precisions(args.next());
    let exact_softmax = parse_mode(args.next());

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
    let sweep = if exact_softmax {
        pipeline.sweep_prompt_refreshed_quantization_exact_softmax(&prompt, &precisions)?
    } else {
        pipeline.sweep_prompt_refreshed_quantization(&prompt, &precisions)?
    };

    println!("prompt: {:?}", user_prompt);
    println!("tokenized_prompt_len: {}", sweep.prompt_tokens.len());
    println!(
        "attention_mode: {}",
        if exact_softmax { "exact_softmax" } else { "configured" }
    );
    println!("exact: token={} {:?}", sweep.exact_token_id, sweep.exact_token_text);
    for row in sweep.rows {
        println!(
            "precision={}: token={} {:?} | L-inf={:.3} L2={:.3} MAE={:.3} | exact_rank={} pred_logit={} exact_logit={}",
            row.precision,
            row.token_id,
            row.token_text,
            row.linf,
            row.l2,
            row.mae,
            row.exact_token_rank,
            row.predicted_logit,
            row.exact_token_logit,
        );
    }

    Ok(())
}
