#!/usr/bin/env python3
"""
Generate greedy (temp=0) decoding fixtures for end-to-end inference verification.
Produces one file per prompt with token IDs, text, and step-by-step logits.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path(__file__).parent.parent / "hf_models" / "Qwen3-0.6B"
OUT_DIR = Path(__file__).parent.parent / "fixtures" / "greedy-decoding"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Hello how are you?",
    "Do you know the story of this proverb '磨刀不误砍柴功' and why is it so similar to Abraham Lincoln quote?",
]

MAX_NEW_TOKENS = 20  # short fixtures for fast verification


def main():
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()

    for prompt in PROMPTS:
        safe_name = prompt.replace(" ", "_").replace("'", "").replace("?", "")[:40]
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"{'='*70}")

        # Use greedy generation with output_scores so we can inspect step logits
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids  # [1, seq_len]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy (temp = 0)
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,       # per-step logits
                return_dict_in_generate=True,
            )

        # outputs.sequences: [1, total_len]  (prompt + generated)
        # outputs.scores: tuple of per-step logits, one per generated token
        #   each element: [batch=1, vocab_size]

        full_ids = outputs.sequences[0].tolist()
        prompt_ids = input_ids[0].tolist()
        generated_ids = full_ids[len(prompt_ids):]

        full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Collect per-step logits (top-10 values) for verification
        step_logits = []
        for i, score in enumerate(outputs.scores):
            # score: [1, vocab_size] — logits for position (prompt_len + i)
            top10_vals, top10_idxs = score[0].topk(10)
            step_logits.append({
                "step": i,
                "chosen_token": generated_ids[i],
                "chosen_logit": float(score[0][generated_ids[i]].item()),
                "top10_tokens": top10_idxs.tolist(),
                "top10_logits": top10_vals.tolist(),
            })

        fixture = {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "full_ids": full_ids,
            "generated_ids": generated_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "num_prompt_tokens": len(prompt_ids),
            "num_generated_tokens": len(generated_ids),
            "eos_token_id": tokenizer.eos_token_id,
            "steps": step_logits,
        }

        out_path = OUT_DIR / f"{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=2, ensure_ascii=False)

        print(f"  Prompt tokens:  {len(prompt_ids)}")
        print(f"  Generated:      {len(generated_ids)} tokens")
        print(f"  Output:         {generated_text!r}")
        print(f"  Fixture saved:  {out_path}")


if __name__ == "__main__":
    main()
