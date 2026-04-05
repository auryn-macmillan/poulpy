#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path("/home/dev/models/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main")
TOKENIZER_DIR = Path("/home/dev/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/main")


def build_prompt(user_prompt: str):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant named SmolLM, trained by Hugging Face."},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="What is 2+2? Answer with one token.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()

    messages = build_prompt(args.prompt)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    input_len = inputs["input_ids"].shape[1]
    new_ids = generated[0, input_len:].tolist()
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    full = tokenizer.decode(generated[0], skip_special_tokens=True)

    payload = {
        "prompt": args.prompt,
        "chat_prompt": text,
        "generated_ids": new_ids,
        "response": response,
        "full_text": full,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
