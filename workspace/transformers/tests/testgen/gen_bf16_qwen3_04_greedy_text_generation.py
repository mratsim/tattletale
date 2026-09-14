#!/usr/bin/env python3
"""Generate greedy (temp=0) decoding fixtures for end-to-end inference
verification under the ttt-tf-001-greedy-steps-h2 schema:

- per step the top-32 ids with f32 logits, the argmax margin
- the softmax tail probability beyond the top-32 support

The installed transformers is the source of truth, no vendored checkout is consulted.

Payloads ship as single-entry deflate .json.zst archives
(FIXTURE_GENERATION.md rule 10).
"""

import json
import os
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa, the E402 import follows the path insert
    argmax_record_from_step, write_argmax_decisions,
    write_json_zst)

MODEL_PATH = Path(__file__).parent.parent / "hf_models" / "Qwen3-0.6B"
OUT_DIR = Path(__file__).parent.parent / "fixtures" / "bf16-04-greedy-text-generation" / "Qwen3-0.6B"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Hello how are you?",
    "Do you know the story of this proverb '磨刀不误砍柴功' and why is it so similar to Abraham Lincoln quote?",
]

GREEDY_STEPS_SCHEMA = "ttt-tf-001-greedy-steps-h2"
MAX_NEW_TOKENS = 20  # short fixtures for fast verification

TOP_K = 32

DECISION_ULP_DATATYPE = "bf16"
    # The ulp datatype of the chain, serialized as the 005 decisions
    # frame's "ulp_datatype" key, the unit every decision band check consumes.



def main():
    """Records the greedy chain fixtures and the decision frames."""
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
    )
    model.eval()

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "recorded_from": "m4max-cpu",
        "num_threads": torch.get_num_threads(),
        "dtype": "bfloat16",
        "device": "cpu",
    }

    for prompt in PROMPTS:
        safe_name = prompt.replace(" ", "_").replace("'", "").replace("?", "")[:40]
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"{'='*70}")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids  # [1, seq_len] tensor shape

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy (temp = 0)
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,       # per-step f32 logits captured
                return_dict_in_generate=True,
            )

        full_ids = outputs.sequences[0].tolist()
        prompt_ids = input_ids[0].tolist()
        generated_ids = full_ids[len(prompt_ids):]

        full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        steps = []
        margins = []
        for i, score in enumerate(outputs.scores):
            last_f32 = score[0].float()
            top_vals, top_idxs = torch.topk(last_f32, TOP_K)
            probs = torch.softmax(last_f32, dim=-1)
            tail = float(1.0 - probs[top_idxs].sum().item())
            margin = float(top_vals[0].item() - top_vals[1].item())
            margins.append(margin)
            steps.append({
                "step": i,
                "chosen_token": generated_ids[i],
                "top32_ids": top_idxs.tolist(),
                "top32_logits": [float(v) for v in top_vals.tolist()],
                "argmax_margin": margin,
                "tail_probability": tail,
            })

        fixture = {
            "schema": GREEDY_STEPS_SCHEMA,
            "model": "Qwen3-0.6B",
            "env": env,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "full_ids": full_ids,
            "generated_ids": generated_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "num_prompt_tokens": len(prompt_ids),
            "num_generated_tokens": len(generated_ids),
            "eos_token_id": tokenizer.eos_token_id,
            "steps": steps,
        }

        out_path = OUT_DIR / f"{safe_name}.json.zst"
        write_json_zst(str(out_path), fixture)
        write_argmax_decisions(
            str(OUT_DIR / f"{safe_name}.decisions.json.zst"), safe_name,
            [argmax_record_from_step(step)
             for step in steps], DECISION_ULP_DATATYPE)

        print(f"  Prompt tokens:  {len(prompt_ids)}")
        print(f"  Generated:      {len(generated_ids)} tokens")
        print(f"  Output:         {generated_text!r}")
        print(f"  Fixture saved:  {out_path}")

    print("\nMargin distribution (top1 - top2, f32):")
    print(f"  min {min(margins):.6f}  median {sorted(margins)[len(margins)//2]:.6f}  max {max(margins):.6f}")


if __name__ == "__main__":
    main()
