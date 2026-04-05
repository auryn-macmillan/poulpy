#!/usr/bin/env python3
"""
Create a toy safetensors model for EncryptedLMHead testing.
- vocab_size = 10
- d_model = 4
- Deterministic weights for reproducibility
"""

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# Set seed for reproducibility
np.random.seed(42)

# Model dimensions
vocab_size = 10
d_model = 4

# Create deterministic weights
# Use simple patterns that are easy to verify

# Embedding table: [vocab_size, d_model]
# Use identity-like pattern for first few tokens
embedding = np.zeros((vocab_size, d_model), dtype=np.float32)
for i in range(min(vocab_size, d_model)):
    embedding[i, i] = 1.0  # Identity for first 4 tokens
# Add some noise to other entries
embedding[4:, :] = np.random.randn(vocab_size - 4, d_model) * 0.1

# LM head weights: [vocab_size, d_model]
# Use simple pattern: first row is all positive, rest are mixed
lm_head = np.zeros((vocab_size, d_model), dtype=np.float32)
lm_head[0, :] = np.array([1.0, 2.0, 3.0, 4.0])  # First logit: positive weights
lm_head[1, :] = np.array([-1.0, -2.0, -3.0, -4.0])  # Second logit: negative weights
lm_head[2:, :] = np.random.randn(vocab_size - 2, d_model) * 0.5

# Save as INT8 quantized
def quantize_to_int8(arr, scale=1.0):
    """Quantize float array to INT8."""
    quantized = (arr / scale).astype(np.int8)
    return quantized, scale

# Quantize weights
embedding_quant, emb_scale = quantize_to_int8(embedding, scale=1.0)
lm_head_quant, lm_scale = quantize_to_int8(lm_head, scale=1.0)

# Save model
model = {
    "model.embed_tokens.weight": embedding_quant,
    "lm_head.weight": lm_head_quant,
}

save_file(model, "toy_model.safetensors")

# Create minimal tokenizer (just a simple mapping)
import json
tokenizer = {
    "vocab": {f"<token_{i}>": i for i in range(vocab_size)},
    "bos_token": "<token_0>",
    "eos_token": "<token_1>",
}
with open("tokenizer.json", "w") as f:
    json.dump(tokenizer, f, indent=2)

print(f"Created toy model: vocab_size={vocab_size}, d_model={d_model}")
print(f"Embedding shape: {embedding.shape}, scale: {emb_scale}")
print(f"LM head shape: {lm_head.shape}, scale: {lm_scale}")
print("Files created: toy_model.safetensors, tokenizer.json")