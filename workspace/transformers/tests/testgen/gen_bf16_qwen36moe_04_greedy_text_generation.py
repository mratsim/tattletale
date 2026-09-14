#!/usr/bin/env python3
"""
Qwen3.6-35B-A3B bf16-04 greedy fixtures, token chains argmax-decoded from the real checkpoint
through the installed transformers modeling on CPU torch bf16, one JSON fixture per prompt.

Decode entry:

  - generation starts from the prompt tokens directly (no bos)
  - single unpadded chains, no padding and no batching, the serialized values are the single-chain values
  - no batched pass backs them, the batched left-pad pass is a different reduction order in bf16, not a reference

Generated under tests/fixtures/bf16-04-greedy-text-generation/Qwen3.6-35B-A3B/:

  - <prompt>_<horizon>_steps.json, one file per prompt
  - prompt_ids carries the checkpoint tokenizer ids, no special tokens added
  - generated_ids carries the horizon greedy tokens, the argmax of the last-position logits at every decode step
  - steps carries the ttt-tf-001-greedy-steps-h2 per-step records
  - each record holds chosen_token, top32_ids, top32_logits in f32, argmax_margin, tail_probability beyond the top-32 support
  - the identity block keys (model, prompt text, horizon, the thread/eager/attn settings, and the torch / transformers versions)

Near-tie clause (recorded for the consumer):

  - a token-chain divergence between two faithful CPU bf16 ports is a decode-argmax near-tie
  - the condition is the diverging pick equal to the recorded runner-up id
  - the recorded top-2 logit gap at that step is at most 2 bf16 ulps, the ulp scaled at the recorded max |logit| of the step's top-2 pair
  - past such a within-band flip the two chains legitimately diverge, so the consumer stops comparing that prompt
  - the top-2 pair is recorded at every step, so the consumer never needs a second rule

Locked configuration:

  - intra-op threads = 1
  - the expert dispatch locked to `eager` (the default backend resolution picks `grouped_mm`, a different accumulation formulation)
  - the `_attn_implementation` setting of `sdpa`
  - the model checks (loading, untied-head, per-layer eager asserts) come shared from gen_bf16_qwen36moe_03_full_forward_to_logits.py
  - both generators verify the identical configuration, its error messages carry that module name

  - run `cd <worktree root> && .venv/bin/python workspace/transformers/tests/testgen/gen_bf16_qwen36moe_04_greedy_text_generation.py` twice
  - the bytes of both runs must be identical before the fixtures are installed

One model-resident process globally:

  - the script verifies a free+inactive+speculative pool above 32 GiB and no other python/torch process before it loads anything
  - the floor is sized to the measured anonymous peak under the mmap-backed from_pretrained loader
  - the 70 GB weight stack uses file-backed pages, the same measurement the bf16-03-full-forward-to-logits fixtures recorded
"""

import hashlib
import json
import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}


def write_json_zst(path, obj, ensure_ascii=True):
    """Writes obj as one zstd frame.

    - level 19, content size and checksum recorded in the frame header
    - JSON fixtures stay reviewable via the generator, the frame stays
      out of text diffs
    """
    payload = json.dumps(
        obj, sort_keys=True, indent=2, ensure_ascii=ensure_ascii
    ).encode("utf-8") + b"\n"
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))

import os
import subprocess
import sys

def _load_sibling(filename: str):
    """Loads and returns a testgen generator module by filename.

    - the naming rule allows dots and hyphens that import syntax rejects,
      so the module loads from its file path under a derived name
    - a second call returns the cached module, never a second execution
      of its top level
    """
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



# The bf16-03-full-forward-to-logits generator carries the shared model,
# version and identity checks, so both fixture generators verify one
# identical configuration of the loaded weights.
# - naming-rule filename characters are beyond import syntax, so the module
#   comes from its path and unpacks the shared names by attribute

_ids_inference = _load_sibling("gen_bf16_qwen36moe_03_full_forward_to_logits.py")
NUM_THREADS = _ids_inference.NUM_THREADS
GREEDY_STEPS_SCHEMA = "ttt-tf-001-greedy-steps-h2"

DECISION_ULP_DATATYPE = "bf16"
    # The ulp datatype of the chain, serialized as the 005 decisions
    # frame's "ulp_datatype" key, the unit every decision band check consumes.

MODEL_NAME = _ids_inference.MODEL_NAME
MODEL_DIR = _ids_inference.MODEL_DIR
INDEX_PATH = _ids_inference.INDEX_PATH
load_wrapper_config = _ids_inference.load_wrapper_config
build_model = _ids_inference.build_model

import transformers
import torch

torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import AutoTokenizer

TRANSFORMERS_VERSION = transformers.__version__

GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-04-greedy-text-generation", "Qwen3.6-35B-A3B"
)

# 3 distinct simple English prompts, each tokenized to at most 12 tokens
# by this checkpoint's tokenizer (asserted below). Horizons stay inside
# 32 greedy steps.
PROMPT_SPEC = [
    ("Hello, how are you?", 32),
    ("The capital of France is", 32),
    ("Big blue whales eat krill.", 32),
]
MAX_PROMPT_TOKENS = 12
MAX_HORIZON = 32
# Pool floor, same formula and same constant as the bf16-03-full-forward-to-logits generator.
MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_qwen36moe_04_greedy_text_generation] vm_stat gave no page size line")


def pool_bytes() -> int:
    """Free+inactive+speculative physical memory in bytes from vm_stat."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    wanted = ("Pages free:", "Pages inactive:", "Pages speculative:")
    pool = 0
    for line in out.stdout.splitlines():
        for label in wanted:
            if line.startswith(label):
                pool += int(line.split()[2].rstrip(".")) * page
    if pool == 0:
        raise SystemExit("[gen_bf16_qwen36moe_04_greedy_text_generation] vm_stat gave no pool lines")
    return pool


def ancestor_pids() -> set:
    """PIDs of this process and its ancestors, up to init."""
    chain = set()
    pid = os.getpid()
    for _ in range(16):
        if pid <= 1:
            break
        chain.add(pid)
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True)
        try:
            pid = int(out.stdout.strip())
        except ValueError:
            break
    return chain


def check_ram() -> None:
    """Refuses to load weights when the memory pool is low or another
    python/torch process holds RAM.

    - this process chain is excluded from the pgrep match, whose command
      line spells the torch dependency of this run
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_04_greedy_text_generation] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_04_greedy_text_generation] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def prompt_token_ids(tokenizer, text: str) -> tuple[int, ...]:
    """Prompt ids under the checkpoint tokenizer, no special tokens added.

    - from a single input with no bos prepended, the transformers tokenizer
      returns a BatchEncoding (a UserDict with no ids attribute) whose
      input_ids are a flat list for one unpadded prompt
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    assert len(ids) >= 1 and len(ids) <= MAX_PROMPT_TOKENS, (
        f"prompt {text!r} holds {len(ids)} tokens, expected 1..{MAX_PROMPT_TOKENS}")
    return tuple(int(t) for t in ids)


def generation_eos_ids() -> list[int]:
    """Stop ids of the wrapper, from generation_config.json.

    - the greedy chains below assert none of these ids, every step must
      stay an argmax over live vocabulary
    - a chain that reached eos could not be extended token-exactly
    """
    path = os.path.join(MODEL_DIR, "generation_config.json")
    with open(path) as f:
        raw = json.load(f)["eos_token_id"]
    if raw is None:
        return []
    return [int(raw)] if isinstance(raw, int) else [int(t) for t in raw]


def greedy_chain(model, token_ids: tuple[int, ...], max_new_tokens: int) -> dict:
    """Greedy-decodes one unpadded chain through the installed forward,
    prefill over the whole prompt with the returned past-key cache, then
    one single-token forward per step feeding back the argmax id.

    Returns the generated ids and the ttt-tf-001-greedy-steps-h2 per-step
    records of the deciding last-position logits row:

      - the top-32 ids with f32 logits, the argmax margin
      - the softmax tail probability beyond the top-32 support
      - exactly max_new_tokens argmax picks, the cache makes each step
        one O(1)-length forward
    """
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long)
    generated = []
    steps = []
    eos = generation_eos_ids()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        for step in range(max_new_tokens):
            last_f32 = out.logits[0, -1].float()
            # torch.topk over the f32 router logits records the top-32
            # support and the argmax margin uses the top-2 values.
            # - torch.topk orders equal values by a selection-internal
            #   detail that flips with k, while torch.argmax returns
            #   the first-max index
            # - the recorded chosen_token stays the argmax pick, an exact
            #   tie records margin 0.0 and the consumer's checks classify
            #   the divergence at that step
            argmax_id = int(last_f32.argmax().item())
            generated.append(argmax_id)
            assert argmax_id not in eos, (
                f"chain hit a configured eos id {argmax_id} at step {step}; "
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
                nxt = torch.tensor([[argmax_id]], dtype=torch.long)
                out = model(nxt, past_key_values=cache, use_cache=True,
                            logits_to_keep=1)
                cache = out.past_key_values
    return {
        "generated_ids": generated,
        "steps": steps,
    }


def prompt_fixture_name(text: str, max_new_tokens: int) -> str:
    """File name for one prompt fixture, alphanumerics only with the horizon suffix, one file per prompt."""
    safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    safe = "_".join(part for part in safe.split("_") if part)
    return f"{safe}_{max_new_tokens}_steps.json.zst"


def main() -> None:
    """Records the greedy chain fixtures and the decision frames."""
    print(f"Generating {MODEL_NAME} bf16-04-greedy-text-generation fixtures")
    print("=" * 60)
    check_ram()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fixture_stats import argmax_record_from_step, write_argmax_decisions

    cfg = load_wrapper_config()
    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    model = build_model(cfg, weight_map)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

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
        fixture = {
            "schema": GREEDY_STEPS_SCHEMA,
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "prompt_ids": list(token_ids),
            "num_prompt_tokens": len(token_ids),
            "generated_ids": chain["generated_ids"],
            "num_generated_tokens": max_new_tokens,
            "steps": chain["steps"],
            "num_threads": NUM_THREADS,
            "experts_implementation": model.config.text_config._experts_implementation,
            "attn_implementation": cfg._attn_implementation,
            "torch_version": torch.__version__,
            "transformers_version": TRANSFORMERS_VERSION,
            "dtype": "bfloat16",
            "device": "cpu",
            "note": "single unpadded chains; the serialized values are the "
                    "single-chain values, no batched pass backs them",
        }
        out_path = os.path.join(
            FIXTURE_DIR, prompt_fixture_name(prompt_text, max_new_tokens))
        write_json_zst(out_path, fixture, ensure_ascii=False)
        chain_stem = os.path.basename(out_path).removesuffix(".json.zst")
        write_argmax_decisions(
            os.path.join(FIXTURE_DIR, f"{chain_stem}.decisions.json.zst"),
            chain_stem,
            [argmax_record_from_step(step)
             for step in chain["steps"]], DECISION_ULP_DATATYPE)
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
