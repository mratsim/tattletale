#!/usr/bin/env python3
"""
Generate greedy (temperature 0) decoding fixtures for the Qwen3.5-0.8B text
stack using the reference transformers modeling on CPU torch bf16.

Reference: gen_bf16_qwen3_04_greedy_text_generation.py conventions (one JSON file per prompt).

Decode entry convention: the Qwen3.5 config has no bos_token_id and no
generation_config.json, so the decode entry is defined explicitly here:
generation starts from the prompt tokens directly (no special token is
prepended) and stops at config eos_token_id 248044. The tokenizer's own
eos (248046, im_end) is not used.

One prompt ("The resume is ready", decomposed e + U+0301) carries combining
marks in its token stream. The reference pre-tokenizer regex includes the
\\p{M} class, so the marks merge into letter tokens ("résumé"). The fixture
locks that token stream so the Nim tokenizer must handle combining marks the
same way.

What is generated (under tests/fixtures/bf16-04-greedy-text-generation/Qwen3.5-0.8B/):

  <safe_name>.json   per prompt: prompt, prompt_ids, full_ids,
    generated_ids, full_text, generated_text, num_prompt_tokens,
    num_generated_tokens, eos_token_id.

environments missing the """

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import platform  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from fixture_stats import recording_env, write_json_zst, write_provenance  # noqa: E402
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from safetensors import safe_open

# Determinism (called once at import time).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config.
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

# (prompt, output file name, max_new_tokens, seed). The second prompt
# uses decomposed e + U+0301 combining acute to exercise pre-tokenizer
# \p{M} class and decodes to text identical to the precomposed
# form. Greedy decoding is sampling-free, so the seeds
# are fixed per prompt for reproducible RNG state on reruns.
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

    The rotary inv_freq buffer is restored to f32 after the dtype cast: the
    reference rotary forward computes cos/sin in f32 and bf16 storage would
    round the frequency values (~1e-3 per element).
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

        # tt-greedy-2: replay generation step by step so each deciding row
        # is captured in f32. Greedy is argmax over the raw logits row.
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
            "schema": "tt-greedy-2",
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
        print(f"  Generated: {len(generated_ids)} tokens -> {generated_text!r}")
        print(f"  Fixture saved: {out_path}")

    print("=" * 60)
    provenance = recording_env(
        model=MODEL_NAME,
        generator="testgen/gen_bf16_qwen35dense_04_greedy_text_generation.py",
        seed="none (greedy temp=0, no sampling)",
        extra={"dtype": "bfloat16"},
    )
    write_provenance(os.path.join(FIXTURE_DIR, "PROVENANCE.md"),
                     list(provenance.items()))
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
