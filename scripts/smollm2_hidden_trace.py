#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path("/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main")
TOKENIZER_DIR = Path("/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main")


def build_prompt(user_prompt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant named SmolLM, trained by Hugging Face."},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="What is 2+2? Answer with one token.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    prompt = build_prompt(args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    position_ids = torch.arange(inputs["input_ids"].shape[1], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        out = model.model(**inputs, output_hidden_states=True, use_cache=False)

        block = model.model.layers[0]
        hidden0 = out.hidden_states[0]
        ln1 = block.input_layernorm(hidden0)
        pos_emb = model.model.rotary_emb(ln1, position_ids)
        q = block.self_attn.q_proj(ln1)
        k = block.self_attn.k_proj(ln1)
        v = block.self_attn.v_proj(ln1)
        attn_out, _ = block.self_attn(
            hidden_states=ln1,
            position_embeddings=pos_emb,
            attention_mask=None,
            past_key_values=None,
            cache_position=position_ids[0],
        )
        residual1 = hidden0 + attn_out
        ln2 = block.post_attention_layernorm(residual1)
        gate = block.mlp.gate_proj(ln2)
        up = block.mlp.up_proj(ln2)
        mlp_out = block.mlp(ln2)
        residual2 = residual1 + mlp_out

    hidden_states = out.hidden_states
    last_token_idx = inputs["input_ids"].shape[1] - 1
    rows = []
    for i, hs in enumerate(hidden_states):
        vec = hs[0, last_token_idx, :].cpu().tolist()
        stage = "embed_last" if i == 0 else f"layer_{i}"
        rows.append({"stage": stage, "hidden": vec})

    sublayers = [
        ("layer1_pre_attn_norm", ln1),
        ("layer1_q", q),
        ("layer1_k", k),
        ("layer1_v", v),
        ("layer1_attn_out", attn_out),
        ("layer1_residual1", residual1),
        ("layer1_pre_ffn_norm", ln2),
        ("layer1_gate", gate),
        ("layer1_up", up),
        ("layer1_mlp_out", mlp_out),
        ("layer1_residual2", residual2),
    ]
    sublayer_rows = [
        {"stage": name, "hidden": tensor[0, last_token_idx, :].cpu().tolist()} for name, tensor in sublayers
    ]

    payload = {
        "prompt": args.prompt,
        "chat_prompt": prompt,
        "prompt_len": inputs["input_ids"].shape[1],
        "rows": rows,
        "sublayers": sublayer_rows,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
