use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use fhe_llm::attention::SoftmaxStrategy;
use tokenizers::Tokenizer;
use poulpy_core::layouts::LWEInfos;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let prompt = "2+2=".to_string();
    let num_layers = 1usize;
    let trunc_d_model = 128usize;

    println!("=== Before/After Bootstrap Debug ===");

    let model_path = "/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/model.safetensors";
    let tokenizer_path = "/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main/tokenizer.json";

    let pipeline = InferencePipeline::load(
        model_path,
        tokenizer_path,
        ModelSpec::smollm2_135m_instruct(),
        InferenceConfig {
            security: SecurityLevel::Bits100,
            precision: Precision::Int8,
            num_layers: Some(num_layers),
            trunc_d_model: Some(trunc_d_model),
            trunc_d_ffn: Some(256usize),
            num_heads: Some(2usize),
            num_kv_heads: Some(1usize),
            softmax_strategy: SoftmaxStrategy::ReluSquared,
            apply_final_norm: true,
            max_new_tokens: 1,
            ..InferenceConfig::default()
        },
    )?;

    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Tokenizer");
    let encoding = tokenizer.encode(prompt.clone(), true).expect("Encode");
    let token_ids = encoding.get_ids().to_vec();
    let first_token = token_ids[0];

    let embeddings = pipeline.embedding();
    let full_embedding = embeddings.lookup(first_token as usize).to_vec();
    let input_embeddings: Vec<i64> = full_embedding[..trunc_d_model].to_vec();
    
    println!("Input embedding first 4: {:?}", &input_embeddings[..4]);

    let hidden_encrypted = pipeline.fhe_forward_pass_to_hidden(&input_embeddings);
    
    // Check ciphertext before refresh
    let hidden_before_refresh = pipeline.decrypt_vec_at_precision(&hidden_encrypted, 10);
    println!("Ciphertext before refresh (10-bit): first 16 = {:?}", &hidden_before_refresh[..16.min(hidden_before_refresh.len())]);

    // Now manually refresh and check
    let hidden_after_refresh = pipeline.refresh_vec_at_precision(&hidden_encrypted, 10);
    let hidden_after = pipeline.decrypt_vec_at_precision(&hidden_after_refresh, 10);
    println!("Ciphertext after refresh (10-bit): first 16 = {:?}", &hidden_after[..16.min(hidden_after.len())]);

    Ok(())
}
