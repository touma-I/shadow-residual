#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the special-token surface of a tokenizer.

Diagnostic tool: given an HF model id, dump the special-tokens map,
additional special tokens, and probe a handful of common role/EOS markers
to see which encode to exactly one token id (candidates for
adapter.last_context_token / adapter.last_token in the SR config).
"""
from __future__ import annotations
import argparse
import json
import sys


CANDIDATE_MARKERS = [
    # Granite 4.1 canonical (works if tokenizer inherits those)
    "<|end_of_role|>",
    "<|start_of_role|>",
    "<|end_of_text|>",
    # Common alternatives
    "<|im_end|>",
    "<|im_start|>",
    "<|end|>",
    "<|assistant|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end_of_turn|>",
    "</s>",
    "<|eos|>",
    # Rare / model-family specific
    "<end_of_turn>",
    "<eos>",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"=== tokenizer: {args.model} ===")
    print(f"class: {type(tok).__name__}")
    print(f"vocab_size: {tok.vocab_size}")
    print(f"model_max_length: {tok.model_max_length}")
    print(f"bos_token: {tok.bos_token!r}  id={tok.bos_token_id}")
    print(f"eos_token: {tok.eos_token!r}  id={tok.eos_token_id}")
    print(f"pad_token: {tok.pad_token!r}  id={tok.pad_token_id}")
    print(f"unk_token: {tok.unk_token!r}  id={tok.unk_token_id}")

    print()
    print("=== special_tokens_map ===")
    print(json.dumps(tok.special_tokens_map, indent=2, default=str))

    print()
    print("=== additional_special_tokens ===")
    ast = getattr(tok, "additional_special_tokens", None) or []
    ast_ids = getattr(tok, "additional_special_tokens_ids", None) or []
    for tok_str, tid in zip(ast, ast_ids):
        print(f"  {tok_str!r} -> {tid}")

    print()
    print("=== candidate marker probe (must encode to 1 id) ===")
    for cand in CANDIDATE_MARKERS:
        ids = tok.encode(cand, add_special_tokens=False)
        status = "OK single-token" if len(ids) == 1 else f"NO ({len(ids)} tokens)"
        print(f"  {cand!r:30s} -> {ids}   {status}")

    print()
    print("=== rendered sample chat (first 200 + last 400 chars) ===")
    sample = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    try:
        rendered = tok.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        print(f"[rendered length: {len(rendered)} chars]")
        print("--- head ---")
        print(rendered[:200])
        print("--- tail ---")
        print(rendered[-400:])
    except Exception as e:
        print(f"apply_chat_template failed: {e}")

    print()
    print("=== chat_template (first 2000 chars) ===")
    ct = getattr(tok, "chat_template", None) or ""
    print(ct[:2000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
