#!/usr/bin/env python3
"""Ling-3.0-tiny fixture generator over the real checkpoint shards, resolved through the gitignored tests/hf_models symlink.

bf16-04 greedy-text-generation records, token chains argmax-decoded through the reference modeling on torch bf16:
- fixture dir tests/fixtures/bf16-04-greedy-text-generation/Ling-3.0-tiny/
- consumer tests/q_bf16/t_bf16_ling3_04_greedy_text_generation.nim

- <prompt>_<horizon>_steps.json.zst, the ttt-tf-001-greedy-steps-h2 chain with the env frame
- <prompt>_<horizon>_steps.decisions.json.zst, the ttt-tf-005-argmax-decisions frame over the same steps

Chain contract:
- hand-rolled single-token decode over the whole-prompt prefill, single unpadded chains, no batched pass backs the values
- every decode step runs the KDA fused_recurrent dispatch (q_len == 1)
  with the causal conv window of the correct width
- a divergence is a near-tie iff the diverging pick equals the recorded runner-up id and the top-2 gap sits within a few bf16 ulps

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root, PYTHONPATH pointing at the kimi-linear reference worktree src with the torch fallback kernels:
  PYTHONPATH=<kimi-linear-reference-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_ling3_04_greedy_text_generation.py
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa: E402, the path insert precedes the import
    argmax_record_from_step,
    recording_env,
    write_argmax_decisions,
)


def _load_sibling(filename: str):
    """Execute and return a testgen generator module by file path under a derived module name, a second call returns the cached module."""
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
# build (shims, fla stubs, from_pretrained load, dtype checks), checkpoint
# counts and the tokenizer loader:
# - it loads the layer-internals sibling for the zstd writer and the ram check
# - determinism locks at import time through one intra-op torch thread
_ling03 = _load_sibling("gen_bf16_ling3_03_full_forward_to_logits.py")
_ling01 = _ling03._ling01
NUM_THREADS = _ling01.NUM_THREADS
GREEDY_STEPS_SCHEMA = "ttt-tf-001-greedy-steps-h2"
MODEL_NAME = _ling03.MODEL_NAME
MODEL_DIR = _ling03.MODEL_DIR
INDEX_PATH = _ling03.INDEX_PATH
FIXTURE_DIR = os.path.join(
    _ling01.TESTS_DIR, "fixtures", "bf16-04-greedy-text-generation", MODEL_NAME
)
write_json_zst = _ling01.write_json_zst
check_ram = _ling01.check_ram
build_model = _ling03.build_model
load_tokenizer = _ling03.load_tokenizer
index_census = _ling03.index_census
TRANSFORMERS_VERSION = _ling03.TRANSFORMERS_VERSION

import torch  # noqa: E402

# 3 distinct simple English prompts, each tokenized to at most 12 tokens
# by this checkpoint's tokenizer, hard-asserted against recorded corpus
# ids verified over the AutoTokenizer fast path:
# - horizons stay inside 32 greedy steps
# - prompt one reuses the exact token ids already recorded in the bf16-03
#   full-forward fixture, both fixture families share the deciding row
PROMPT_SPEC = [
    ("Hello, how are you?", 32),
    ("The capital of France is", 32),
    ("Big blue whales eat krill.", 32),
]
RECORDED_PROMPT_IDS = {
    "Hello, how are you?": [14455, 11, 1099, 449, 362, 30],
    "The capital of France is": [678, 7706, 300, 11406, 341],
    "Big blue whales eat krill.": [12888, 7182, 63700, 7638, 636, 42153, 13],
}
MAX_PROMPT_TOKENS = 12
MAX_HORIZON = 32


def prompt_token_ids(tokenizer, text: str) -> tuple[int, ...]:
    """Prompt ids under the checkpoint tokenizer, no special tokens added, no bos prepended, hard-asserted against the recorded corpus ids."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = tuple(int(t) for t in ids)
    expected = RECORDED_PROMPT_IDS[text]
    assert list(ids) == expected, (
        f"prompt {text!r} tokenized to {list(ids)}, recorded corpus ids are {expected}")
    assert 1 <= len(ids) <= MAX_PROMPT_TOKENS, (
        f"prompt {text!r} holds {len(ids)} tokens, expected 1..{MAX_PROMPT_TOKENS}")
    return ids


def generation_stop_ids() -> list[int]:
    """Stop ids from config.json, the single-int eos value asserted equal to the recorded value, the chains assert none of these ids."""
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        raw = json.load(f)["eos_token_id"]
    if raw is None:
        return []
    return [int(raw)] if isinstance(raw, int) else [int(t) for t in raw]


def assert_dtype_plan(model, weight_map: dict) -> None:
    """Value-level dtype-plan asserts on the loaded model, the dtype contract the whole greedy chain replay walks:
    - every routed expert_bias buffer sits on the bf16 grid, bitwise equal
      to the checkpoint f32 tensor round-tripped through bf16, the reference
      load casts the untyped register_buffer under the bf16 deploy context
    - every KDA A_log and dt_bias keeps its checkpoint f32 values, each
      created at an explicit f32 dtype, the meta dtype wins the load"""
    from safetensors import safe_open as _open

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        if (i + 1) % 4 == 0:
            continue
        prefix = f"model.layers.{i}.attention"
        with _open(os.path.join(MODEL_DIR, weight_map[f"{prefix}.A_log"]),
                   framework="pt") as f:
            raw_a_log = f.get_tensor(f"{prefix}.A_log")
        assert layer.attention.A_log.dtype == torch.float32, (
            f"layer {i} A_log drifted off the checkpoint f32 grid")
        assert torch.equal(layer.attention.A_log.detach(), raw_a_log), (
            f"layer {i} A_log values drifted off the checkpoint f32 values")
        with _open(os.path.join(MODEL_DIR, weight_map[f"{prefix}.dt_bias"]),
                   framework="pt") as f:
            raw_dt_bias = f.get_tensor(f"{prefix}.dt_bias")
        assert layer.attention.dt_bias.dtype == torch.float32, (
            f"layer {i} dt_bias drifted off the checkpoint f32 grid")
        assert torch.equal(layer.attention.dt_bias.detach(), raw_dt_bias), (
            f"layer {i} dt_bias values drifted off the checkpoint f32 values")
    for i in range(1, len(layers)):
        prefix = f"model.layers.{i}.mlp.gate.expert_bias"
        bias = layers[i].mlp.gate.expert_bias
        assert bias.dtype == torch.bfloat16, (
            f"layer {i} expert_bias drifted off the reference bf16 buffer "
            f"dtype, dtype {bias.dtype!r} breaks the recorded chain")
        with _open(os.path.join(MODEL_DIR, weight_map[prefix]),
                   framework="pt") as f:
            raw_bias = f.get_tensor(prefix)
        round_tripped = raw_bias.to(torch.bfloat16)
        assert torch.equal(bias.detach(), round_tripped), (
            f"layer {i} expert_bias values sit off the bf16 round trip of "
            "the checkpoint f32 buffer, the reference load grid moved")
        # On-grid proof by the idempotence form:
        #   re-rounding a bf16 value
        # through bf16 is the identity, the grid never widens under it.
        assert torch.equal(round_tripped, round_tripped.to(torch.bfloat16)), (
            f"layer {i} expert_bias round trip is not idempotent")


def greedy_chain(model, token_ids: tuple[int, ...], max_new_tokens: int) -> dict:
    """Greedy-decode one unpadded chain through the installed forward:
    - the prefill covers the whole prompt over an explicit DynamicCache,
      each step runs one single-token forward, the previous argmax id feeds
      back as the next input
    - every decode step runs the natural recurrent dispatch at the cached one-token state
    - returns the generated ids and the per-step records of the deciding
      last-position logits row, the top-32 ids with f32 logits, the argmax
      margin and the softmax tail probability beyond the top-32 support
    - exactly max_new_tokens argmax picks are recorded, the cache makes
      each step one one-token forward"""
    # Remote modeling targets the transformers 4.45 cache protocol, the model
    # never creates a cache, so the replay passes one explicit DynamicCache
    # through every forward, prefill and decode alike:
    # - a direct forward with use_cache=True alone returns past_key_values None
    # - the MLA layers run past_key_values.update, the KDA layers read and write cache.layers[idx].keys plus .values,
    #   the recurrent state and the three conv states, the same
    #   preloaded-cache protocol the layer-internals recording verified
    # - the model forward ignores logits_to_keep, the kwarg disappears into
    #   **kwargs, the lm_head runs over every position and casts f32
    # - the deciding row therefore reads logits[0, -1] off the full tensor,
    #   the grid the bf16-03 recording consumed
    from transformers.cache_utils import DynamicCache
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=DEVICE)
    generated = []
    steps = []
    stop = generation_stop_ids()
    with torch.no_grad():
        cache = DynamicCache()
        out = model(input_ids, past_key_values=cache, use_cache=True)
        for step in range(max_new_tokens):
            last_f32 = out.logits[0, -1].float()
            # torch.topk over the f32 logits records the top-32 support, the argmax
            # margin uses the top-2 values:
            # - torch.topk orders equal values by a selection-internal detail
            #   that flips with k, torch.argmax returns the first-max index
            # - the recorded chosen_token stays the argmax pick, an exact tie
            #   records margin 0.0, the consumer checks classify the divergence
            #   at that step
            argmax_id = int(last_f32.argmax().item())
            generated.append(argmax_id)
            assert argmax_id not in stop, (
                f"chain hit a configured stop id {argmax_id} at step {step}, "
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
                nxt = torch.tensor([[argmax_id]], dtype=torch.long, device=DEVICE)
                out = model(nxt, past_key_values=cache, use_cache=True)
    return {
        "generated_ids": generated,
        "steps": steps,
    }


def prompt_fixture_name(text: str, max_new_tokens: int) -> str:
    """File name for one prompt fixture, alphanumerics only, horizon suffix, one
file per prompt."""
    safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    safe = "_".join(part for part in safe.split("_") if part)
    return f"{safe}_{max_new_tokens}_steps.json.zst"


def main() -> None:
    """Records the greedy chain fixtures and the decision frames."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    print(f"Generating {MODEL_NAME} bf16-04-greedy-text-generation fixtures")
    print("=" * 60)
    check_ram()

    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    index_census(weight_map)
    model = build_model()
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    assert_dtype_plan(model, weight_map)
    tokenizer = load_tokenizer()
    stop = generation_stop_ids()
    assert stop == [156895], (
        f"stop set {stop} moved off the recorded single-int config value [156895]")

    texts = [spec[0] for spec in PROMPT_SPEC]
    assert len(set(texts)) == len(texts), "prompts must be distinct"

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    for prompt_text, max_new_tokens in PROMPT_SPEC:
        assert 0 < max_new_tokens <= MAX_HORIZON, (
            f"horizon for {prompt_text!r} must stay inside {MAX_HORIZON} steps")
        token_ids = prompt_token_ids(tokenizer, prompt_text)
        chain = greedy_chain(model, token_ids, max_new_tokens)
        assert len(chain["generated_ids"]) == max_new_tokens
        assert len(chain["steps"]) == max_new_tokens
        full_ids = list(token_ids) + chain["generated_ids"]
        full_text = tokenizer.decode(full_ids)
        generated_text = tokenizer.decode(chain["generated_ids"])
        fixture = {
            "schema": GREEDY_STEPS_SCHEMA,
            "model": MODEL_NAME,
            "env": recording_env(
                model=MODEL_NAME,
                generator="testgen/gen_bf16_ling3_04_greedy_text_generation.py",
                extra={"dtype": "bfloat16", "device": DEVICE,
                       "num_threads": NUM_THREADS,
                       "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu")}),
            "prompt": prompt_text,
            "prompt_ids": list(token_ids),
            "num_prompt_tokens": len(token_ids),
            "generated_ids": chain["generated_ids"],
            "num_generated_tokens": max_new_tokens,
            "full_ids": full_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "steps": chain["steps"],
            "moe_dispatch": "eager",
            "attn_implementation": "eager",
            "stop_ids": stop,
            "note": "single unpadded chains; the serialized values are the "
                    "single-chain values, no batched pass backs them. The "
                    "reference runs the eager token-gather MoE dispatch "
                    "(moe_infer, no grouped_mm alternative) and the eager "
                    "attention spelling, both hardcoded on the template.",
        }
        out_path = os.path.join(
            FIXTURE_DIR, prompt_fixture_name(prompt_text, max_new_tokens))
        write_json_zst(out_path, fixture)
        decisions_path = str(out_path)[:-len(".json.zst")] + ".decisions.json.zst"
        write_argmax_decisions(
            decisions_path,
            os.path.basename(decisions_path)[:-len(".decisions.json.zst")],
            [argmax_record_from_step(step) for step in chain["steps"]],
            "bf16")
        digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        print(f"  prompt ({len(token_ids)} tokens): {prompt_text!r}")
        print(f"    generated: {chain['generated_ids']}")
        print(f"    fixture: {out_path}")
        print(f"    sha256: {digest}")
        print(f"    step0 margin {chain['steps'][0]['argmax_margin']:.4f}")

    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
