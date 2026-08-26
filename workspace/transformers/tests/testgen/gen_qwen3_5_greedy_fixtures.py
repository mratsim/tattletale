#!/usr/bin/env python3
"""
Generate greedy (temperature 0) decoding fixtures for the Qwen3.5-0.8B text
stack using the VENDORED prod transformers modeling on CPU torch bf16.

Reference: gen_07_greedy_fixture.py conventions (one JSON file per prompt).

Fixtures are the ground truth for the Nim q_bf16 greedy test
The Nim implementation
is fixed to match these fixtures, never the other way around.

Decode entry convention: the Qwen3.5 config has no bos_token_id and no
generation_config.json, so the decode entry is defined explicitly here:
generation starts from the prompt tokens directly (no special token is
prepended) and stops at config eos_token_id 248044. The tokenizer's own
eos (248046, im_end) is not used.

One prompt ("The resume is ready", decomposed e + U+0301) carries combining
marks in its token stream. The vendored pre-tokenizer regex includes the
\\p{M} class, so the marks merge into letter tokens ("résumé"). The fixture
locks that token stream so the Nim tokenizer must handle combining marks the
same way.

What is generated (under tests/fixtures/greedy-decoding/Qwen3.5-0.8B/):

  <safe_name>.json   per prompt: prompt, prompt_ids, full_ids,
    generated_ids, full_text, generated_text, num_prompt_tokens,
    num_generated_tokens, eos_token_id.

Determinism: torch.manual_seed per prompt section, CPU only, greedy decoding
(no sampling), and the tokenizer is deterministic, so reruns are byte-identical.
"""

import json
import os
import sys
import torch

# Vendored prod transformers is the source of truth.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
VENDORED_SRC = os.environ.get(
    "QWEN35_VENDORED_SRC",
    os.path.join(_REPO_ROOT, "_references_prod", "transformers", "src"))
if not os.path.isdir(VENDORED_SRC):
    raise SystemExit(
        f"[gen_qwen3_5_greedy_fixtures] vendored modeling not found at {VENDORED_SRC}. "
        "Set QWEN35_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, VENDORED_SRC)

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
    GRANDPARENT_DIR, "fixtures", "greedy-decoding", MODEL_NAME
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), f"tests/hf_models/{MODEL_NAME}"
)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# (prompt, output file name, max_new_tokens, seed). The second prompt
# uses the decomposed e + U+0301 combining acute to exercise the
# pre-tokenizer \p{M} class. It decodes to the same text as the
# precomposed form. Greedy decoding is sampling-free, so the seeds
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
    """Wrapper model with real shard weights, bf16, eval, CPU.

    The rotary inv_freq buffer is restored to f32 after the dtype cast: the
    vendored rotary forward computes cos/sin in f32 and bf16 storage would
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
            f"[gen_qwen3_5_greedy_fixtures] unexpected missing tensors: {missing}")
    if len(unexpected) != 15:
        raise SystemExit(
            f"[gen_qwen3_5_greedy_fixtures] unexpected foreign tensors: {unexpected}")
    return model


def main() -> None:
    print(f"Generating {MODEL_NAME} greedy-decoding fixtures")
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

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=eos_token_id,
            )

        full_ids = outputs[0].tolist()
        generated_ids = full_ids[len(prompt_ids):]
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        fixture = {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "full_ids": full_ids,
            "generated_ids": generated_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "num_prompt_tokens": len(prompt_ids),
            "num_generated_tokens": len(generated_ids),
            "eos_token_id": eos_token_id,
            "note": "decode entry is the prompt tokens (no bos token exists). "
                    "eos is the config eos_token_id 248044",
        }

        out_path = os.path.join(FIXTURE_DIR, f"{out_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fixture, f, indent=2, ensure_ascii=False)
        print(f"  Generated: {len(generated_ids)} tokens -> {generated_text!r}")
        print(f"  Fixture saved: {out_path}")

    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
