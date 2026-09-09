#!/usr/bin/env python3
"""Generate greedy (temp=0) decoding fixtures for end-to-end inference
verification under the tt-greedy-2 schema:
per step the top-32 ids with f32 logits, the argmax margin, and the
softmax tail probability beyond the top-32 support.

the installed transformers is the source of truth,
no vendored checkout is consulted. Payloads ship as single-entry deflate
.json.zst archives (FIXTURE_GENERATION.md rule 10), and the generator stamps
PROVENANCE.md at record time through fixture_stats.write_provenance.
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

from fixture_stats import recording_env, write_json_zst, write_provenance  # noqa: E402

MODEL_PATH = Path(__file__).parent.parent / "hf_models" / "Qwen3-0.6B"
OUT_DIR = Path(__file__).parent.parent / "fixtures" / "bf16-04-greedy-text-generation" / "Qwen3-0.6B"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "Hello how are you?",
    "Do you know the story of this proverb '磨刀不误砍柴功' and why is it so similar to Abraham Lincoln quote?",
]

MAX_NEW_TOKENS = 20  # short fixtures for fast verification

TOP_K = 32


def main():
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

    provenance = recording_env(
        model="Qwen3-0.6B",
        generator="testgen/gen_bf16_qwen3_04_greedy_text_generation.py",
        seed="none (greedy temp=0, no sampling)",
        extra={"dtype": "bfloat16"},
    )
    write_provenance(str(OUT_DIR / "PROVENANCE.md"), list(provenance.items()))

    for prompt in PROMPTS:
        safe_name = prompt.replace(" ", "_").replace("'", "").replace("?", "")[:40]
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"{'='*70}")

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
            "schema": "tt-greedy-2",
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

        print(f"  Prompt tokens:  {len(prompt_ids)}")
        print(f"  Generated:      {len(generated_ids)} tokens")
        print(f"  Output:         {generated_text!r}")
        print(f"  Fixture saved:  {out_path}")

    print("\nMargin distribution (top1 - top2, f32):")
    print(f"  min {min(margins):.6f}  median {sorted(margins)[len(margins)//2]:.6f}  max {max(margins):.6f}")


if __name__ == "__main__":
    main()
