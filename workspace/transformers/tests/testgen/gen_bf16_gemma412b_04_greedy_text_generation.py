#!/usr/bin/env python3
"""Tier-04 greedy-text-generation fixture generator, gemma-4-12B-it,
torch bf16 on Metal (mps), token chains argmax-decoded through the installed transformers modeling,

- consumer tests/q_bf16/t_bf16_gemma412b_04_greedy_text_generation.nim
- <prompt>_<horizon>_steps.json.zst per prompt, the ttt-tf-001-greedy-steps-h2 chain
- plus the ttt-tf-005-argmax-decisions frame over the same steps

- hand-rolled single-token decode over the whole-prompt prefill, single unpadded chains, each record carries the argmax pick

- the top-32 ids with f32 logits, the argmax margin and the softmax tail probability
- a chain pick equal to a configured eos id fails the recording, every recorded pick stays a live argmax pick
- the third prompt prefill-crosses the 1024 sliding window, the sliding layers record the windowed-mask path over the grown cache

- recorded_from defaults to m4max-metal (TTT_RECORD_FROM overrides), `--chain N` records one chain, replay on the consuming suite's device
- the run refuses the weight load under 48 GiB free+inactive+speculative pool, other python/torch processes holding RAM block it

Regenerate from the worktree root:

  uv run python workspace/transformers/tests/testgen/gen_bf16_gemma412b_04_greedy_text_generation.py
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa: E402, the path insert precedes the import
    argmax_record_from_step,
    recording_env,
    write_argmax_decisions,
    write_json_zst,
)


def _load_sibling(filename: str):
    """Executes and returns one testgen generator module by file path, cached
    under a derived module name so a second call returns the same module."""
    name = filename[:-3] if filename.endswith(".py") else filename
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load sibling generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The bf16-03-full-forward-to-logits generator carries the shared model
# checks (RAM guard, tokenizer loader, identity constants), determinism
# locks at import time in the sibling module through one intra-op thread.
_ids = _load_sibling("gen_bf16_gemma412b_03_full_forward_to_logits.py")
NUM_THREADS = _ids.NUM_THREADS
GREEDY_STEPS_SCHEMA = "ttt-tf-001-greedy-steps-h2"
MODEL_NAME = _ids.MODEL_NAME
check_ram = _ids.check_ram
build_model = _ids.build_model
load_tokenizer = _ids.load_tokenizer

import torch  # noqa: E402

# Recording box of this run, recording_env reads the variable at call time.
os.environ.setdefault("TTT_RECORD_FROM", "m4max-metal")

GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-04-greedy-text-generation", MODEL_NAME
)

# Prompt corpus, one prompt per row with its explicit fixture name:
# - prompt one reuses the exact token ids already recorded in the bf16-03
#   full-forward family, both fixture families share the deciding row
# - the dragon story repeats one sentence to a prefill past the 1024
#   sliding window, its decode steps record the windowed-mask path
# - horizons stay inside 32 greedy steps
PROMPT_SPEC = [
    ("The sky is", 32, "The_sky_is_32_steps"),
    ("To be or not to be, that is", 32, "To_be_or_not_to_be_that_is_32_steps"),
    ("Once upon a time there was a small dragon. " * 105
     + "The dragon looked at", 32, "Dragon_story_crosses_1024_window_32_steps"),
]
MAX_HORIZON = 32

DECISION_ULP_DATATYPE = "bf16"
    # The ulp datatype of the chain, serialized as the 005 decisions
    # frame's "ulp_datatype" key, the unit every decision band check consumes.


def prompt_token_ids(tokenizer, text: str) -> tuple:
    """Prompt ids under the checkpoint tokenizer, no special tokens prepended
    (the checkpoint tokenizer adds no BOS)."""
    ids = tokenizer(text)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = tuple(int(t) for t in ids)
    return ids


def greedy_chain(model, token_ids: tuple, max_new_tokens: int) -> dict:
    """Greedy-decodes one unpadded chain through the installed forward.

    Takes the loaded reference model, the prompt ids and the chain horizon.

    Returns the generated ids plus one record per step over the deciding
    last-position logits row, each record carrying the argmax pick, top-32
    f32 logits, margin and tail probability.

    The prefill covers the whole prompt and returns the past-key cache, each
    step runs one single-token forward over that cache, the previous argmax
    id feeds back as the next input.

    torch.topk orders equal values by a selection-internal detail that flips
    with k, while torch.argmax returns the first-max index, so the recorded
    chosen_token stays the argmax pick and an exact tie records margin 0.0.
    """
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = []
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device="mps")
    generated = []
    steps = []
    with torch.no_grad():
        out = model(input_ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        for step in range(max_new_tokens):
            last_f32 = out.logits[0, -1].float()
            argmax_id = int(last_f32.argmax().item())
            generated.append(argmax_id)
            assert argmax_id not in eos_ids, (
                f"chain hit a configured eos id {argmax_id} at step {step}, "
                "prompt or horizon must be respecified")
            top_vals, top_idxs = torch.topk(last_f32, 32)
            probs = torch.softmax(last_f32, dim=-1)
            tail = float(1.0 - probs[top_idxs].sum().item())
            margin = float(top_vals[0].item() - top_vals[1].item())
            steps.append({
                "step": step,
                "chosen_token": argmax_id,
                "top32_ids": top_idxs.tolist(),
                "top32_logits": [float(v) for v in top_vals.tolist()],
                "argmax_margin": margin,
                "tail_probability": tail,
            })
            if step + 1 < max_new_tokens:
                nxt = torch.tensor([[argmax_id]], dtype=torch.long, device="mps")
                out = model(nxt, past_key_values=cache, use_cache=True,
                            logits_to_keep=1)
                cache = out.past_key_values
    return {
        "generated_ids": generated,
        "steps": steps,
    }


def main() -> None:
    """Records the greedy chain fixtures and the decision frames."""
    chain_arg = None
    if "--chain" in sys.argv:
        chain_arg = int(sys.argv[sys.argv.index("--chain") + 1])
    print(f"Generating {MODEL_NAME} bf16-04-greedy-text-generation fixtures"
          + (f", chain {chain_arg} only" if chain_arg is not None else ""))
    check_ram()

    model = build_model()
    tokenizer = load_tokenizer()

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    for chain_idx, (prompt_text, max_new_tokens, fixture_name) \
            in enumerate(PROMPT_SPEC):
        if chain_arg is not None and chain_idx != chain_arg:
            continue
        assert 0 < max_new_tokens <= MAX_HORIZON, (
            f"horizon for {fixture_name!r} must stay inside {MAX_HORIZON} steps")
        token_ids = prompt_token_ids(tokenizer, prompt_text)
        if prompt_text == PROMPT_SPEC[2][0]:
            assert len(token_ids) > 1024, (
                "the dragon-story prompt must tokenize past the 1024 sliding "
                f"window, got {len(token_ids)} tokens")
        chain = greedy_chain(model, token_ids, max_new_tokens)
        assert len(chain["generated_ids"]) == max_new_tokens
        assert len(chain["steps"]) == max_new_tokens
        full_ids = list(token_ids) + chain["generated_ids"]
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        generated_text = tokenizer.decode(
            chain["generated_ids"], skip_special_tokens=True)
        fixture = {
            "schema": GREEDY_STEPS_SCHEMA,
            "model": MODEL_NAME,
            "env": recording_env(
                model=MODEL_NAME,
                generator="testgen/gen_bf16_gemma412b_04_greedy_text_generation.py",
                extra={"dtype": "bfloat16", "device": "mps",
                       "num_threads": NUM_THREADS,
                       "attn_implementation": model.config.text_config
                       ._attn_implementation}),
            "prompt": prompt_text,
            "prompt_ids": list(token_ids),
            "num_prompt_tokens": len(token_ids),
            "generated_ids": chain["generated_ids"],
            "num_generated_tokens": max_new_tokens,
            "full_ids": full_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "steps": chain["steps"],
            "note": "single unpadded chains, the serialized values are the "
                "single-chain values, no batched pass backs them",
        }
        out_path = os.path.join(FIXTURE_DIR, f"{fixture_name}.json.zst")
        write_json_zst(out_path, fixture)
        decisions_path = str(out_path)[:-len(".json.zst")] + ".decisions.json.zst"
        write_argmax_decisions(
            decisions_path,
            os.path.basename(decisions_path)[:-len(".decisions.json.zst")],
            [argmax_record_from_step(step) for step in chain["steps"]],
            DECISION_ULP_DATATYPE)
        digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        print(f"  prompt ({len(token_ids)} tokens): {fixture_name}")
        print(f"    generated: {chain['generated_ids']}")
        print(f"    fixture: {out_path}")
        print(f"    sha256: {digest}")
        print(f"    step0 margin {chain['steps'][0]['argmax_margin']:.4f}")

    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
