use fhe_llm::inference::{InferenceConfig, InferencePipeline, ModelSpec};
use fhe_llm::params::{Precision, SecurityLevel};
use fhe_llm::attention::SoftmaxStrategy;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let prompt = "2+2=".to_string();
    let num_layers = 1usize;
    let trunc_d_model = 128usize;

    println!("=== Roundtrip Debug ===");

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
    let input_i64: Vec<i64> = full_embedding[..trunc_d_model].to_vec();
    let input_i8: Vec<i8> = input_i64.iter().map(|&v| v as i8).collect();
    
    println!("Input first 16: {:?}", &input_i8[..16.min(input_i8.len())]);

    // Encrypt
    let input_ct = pipeline.encrypt_embedding(&input_i8);
    
    // Decrypt at different precisions
    let dec_8 = pipeline.decrypt_vec_at_precision(&input_ct, 8);
    println!("Decrypt 8-bit: first 16 = {:?}", &dec_8[..16.min(dec_8.len())]);
    
    let dec_10 = pipeline.decrypt_vec_at_precision(&input_ct, 10);
    println!("Decrypt 10-bit: first 16 = {:?}", &dec_10[..16.min(dec_10.len())]);
    
    let dec_14 = pipeline.decrypt_vec_at_precision(&input_ct, 14);
    println!("Decrypt 14-bit: first 16 = {:?}", &dec_14[..16.min(dec_14.len())]);
    
    let dec_26 = pipeline.decrypt_vec_at_precision(&input_ct, 26);
    println!("Decrypt 26-bit: first 16 = {:?}", &dec_26[..16.min(dec_26.len())]);

    Ok(())
}
