#!/usr/bin/env python3
"""
Greedy (temperature 0) decoding fixtures for the Qwen3.5-0.8B text stack,
reference transformers modeling on CPU torch bf16, one JSON file per prompt
following the gen_bf16_qwen3_04_greedy_text_generation.py conventions.

Decode entry, explicit because the Qwen3.5 config has no bos_token_id, no
generation_config.json:

  - generation starts from the prompt tokens directly (no special token prepended)
  - decoding stops at the config eos_token_id 248044
  - the tokenizer's own eos (248046, im_end) is not used

One prompt ("The resume is ready", decomposed e + U+0301) carries combining marks in its token stream:

  - the reference pre-tokenizer regex includes the \p{M} class, so the marks merge into letter tokens ("résumé")
  - the fixture locks that token stream so the Nim tokenizer must handle combining marks the same way

Generated under tests/fixtures/bf16-04-greedy-text-generation/Qwen3.5-0.8B/,
one <safe_name>.json file per prompt.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import platform
import torch
import transformers

from fixture_stats import (  # noqa, the E402 import follows the path insert
    argmax_record_from_step, write_argmax_decisions,
    write_json_zst)
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from safetensors import safe_open

# Determinism (called once at import time).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config, schema id and model path constants.
GREEDY_STEPS_SCHEMA = "ttt-tf-001-greedy-steps-h2"

DECISION_ULP_DATATYPE = "bf16"
    # The ulp datatype of the chain, serialized as the 005 decisions
    # frame's "ulp_datatype" key, the unit every decision band check consumes.


MODEL_NAME = "Qwen3.5-0.8B"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-04-greedy-text-generation", MODEL_NAME
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), f"tests/hf_models/{MODEL_NAME}"
)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# PROMPTS entries carry (prompt, output file name, max_new_tokens, seed).
# - greedy decoding is sampling-free, so the seeds are fixed per prompt
#   for reproducible RNG state on reruns
# - the second prompt uses decomposed e + U+0301 combining acute,
#   exercising the pre-tokenizer \p{M} class and decoding to text
#   identical to the precomposed form
PROMPTS = [
    ("Hello, how are you?", "Hello_how_are_you", 8, 81),
    ("The résumé is ready", "The_resume_is_ready", 8, 82),
    ("What is the capital of France?", "What_is_the_capital_of_France", 8, 83),
]


def load_wrapper_config() -> Qwen3_5Config:
    """Load the wrapper Qwen3_5Config from the model config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5Config.from_dict(wrapper)
    cfg._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    return cfg


def build_model(cfg: Qwen3_5Config) -> Qwen3_5ForConditionalGeneration:
    """Wrapper model with real checkpoint weights, bf16, eval, CPU.

    Returns the model with the rotary inv_freq buffer restored to f32
    after the dtype cast, because bf16 storage would round the frequency
    values (~1e-3 per element) while the reference computes cos/sin in f32.
    """
    model = Qwen3_5ForConditionalGeneration(cfg)
    rotary = model.model.language_model.rotary_emb
    inv_freq = rotary.inv_freq.float()
    original_inv_freq = rotary.original_inv_freq.float()

    model.eval().to(torch.bfloat16)

    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = original_inv_freq

    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    del weights
    if len(missing) != 1 or missing[0] != "lm_head.weight":
        raise SystemExit(
            f"[gen_bf16_qwen35dense_04_greedy_text_generation] unexpected missing tensors: {missing}")
    if len(unexpected) != 15:
        raise SystemExit(
            f"[gen_bf16_qwen35dense_04_greedy_text_generation] unexpected foreign tensors: {unexpected}")
    return model


def main() -> None:
    """Generates the bf16-04 greedy fixtures and the decision records."""
    print(f"Generating {MODEL_NAME} bf16-04-greedy-text-generation fixtures")
    print("=" * 60)
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    cfg = load_wrapper_config()
    model = build_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    eos_token_id = cfg.text_config.eos_token_id
    assert eos_token_id == 248044

    for prompt, out_name, max_new_tokens, seed in PROMPTS:
        torch.manual_seed(seed)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        prompt_ids = input_ids[0].tolist()
        print(f"\nPrompt ({len(prompt_ids)} tokens): {prompt!r}")

        # ttt-tf-001-greedy-steps-h2, replaying generation step by step so
        # each deciding row is captured in f32, greedy decoding being argmax
        # over the raw logits row.
        scores = []
        generated_ids = []
        with torch.no_grad():
            cur = input_ids
            for i in range(max_new_tokens):
                out = model(cur)
                last_f32 = out.logits[0, -1].float()
                scores.append(last_f32)
                nxt = int(last_f32.argmax().item())
                generated_ids.append(nxt)
                if nxt == eos_token_id:
                    break
                cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

        full_ids = prompt_ids + generated_ids
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        steps = []
        margins = []
        for i, last_f32 in enumerate(scores):
            top_vals, top_idxs = torch.topk(last_f32, 32)
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

        env = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "recorded_from": "m4max-cpu",
            "num_threads": torch.get_num_threads(),
            "dtype": "bfloat16",
            "device": "cpu",
        }

        fixture = {
            "schema": GREEDY_STEPS_SCHEMA,
            "model": "Qwen3.5-0.8B",
            "env": env,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "full_ids": prompt_ids + generated_ids,
            "generated_ids": generated_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "num_prompt_tokens": len(prompt_ids),
            "num_generated_tokens": len(generated_ids),
            "eos_token_id": eos_token_id,
            "note": "decode entry is the prompt tokens (no bos token exists). "
                    "eos is the config eos_token_id 248044",
            "steps": steps,
        }

        out_path = os.path.join(FIXTURE_DIR, f"{out_name}.json.zst")
        write_json_zst(out_path, fixture)
        write_argmax_decisions(
            os.path.join(FIXTURE_DIR, f"{out_name}.decisions.json.zst"),
            out_name,
            [argmax_record_from_step(step)
             for step in steps], DECISION_ULP_DATATYPE)
        print(f"  Generated: {len(generated_ids)} tokens -> {generated_text!r}")
        print(f"  Fixture saved: {out_path}")

    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
