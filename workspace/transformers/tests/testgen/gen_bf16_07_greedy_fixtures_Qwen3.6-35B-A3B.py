#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B greedy-decoding fixtures: token chains
argmax-decoded from the real checkpoint through the vendored transformers
modeling on CPU torch bf16, one JSON fixture per prompt.

What is generated (under tests/fixtures/greedy-decoding/Qwen3.6-35B-A3B/):

  <prompt>_<horizon>_steps.json   per prompt:
    input_tokens            the prompt ids, from the checkpoint tokenizer,
                            no special tokens added
    generated_ids           the horizon greedy tokens, argmax of the
                            last-position logits at every decode step
    step_logits_checksum    per step, the float32 sum over the vocab
                            row the step was decided from
    near_tie.logits         per step, float32 pairs [step][2]: the argmax
                            id's logit and the runner-up's logit
    near_tie.ids            per step, [argmax id, runner-up id]; exact
                            fp32 argmax-boundary ties are structural in
                            this checkpoint and torch.topk's tie order
                            there is a selection detail that flips with
                            k, so the pair is anchored on the argmax id
                            the chain feeds and names the true runner-up
    near_tie.tied_steps     per prompt, how many steps tied exactly
                            (logit gap 0.0)

  remaining keys are the identity block: model, prompt text, horizon,
  thread/eager/attn pins, torch / transformers versions, the vendored
  tree sha, and the near-tie clause the consumer applies.

Decode entry: generation starts from the prompt tokens directly (no bos).
The decode loop is hand-rolled: one prefill forward over the whole prompt,
then one single-token forward per step feeding back the previous argmax,
with the past-key cache the vendored model returns. Single chains only:
no padding, no batching, and the serialized contract is the single-chain
values; no batched pass validates them (the batched left-pad pass is a
different reduction order in bf16, not a reference for these chains).

Near-tie clause (recorded for the consumer, provenance R12): a token-chain
divergence between two faithful CPU bf16 ports is a decode-argmax near-tie
when the diverging pick equals the recorded runner-up id and the recorded
top-2 logit gap at that step is at most 2 bf16 ulps, the ulp scaled at
the recorded max |logit| of the step's top-2 pair. Past such a
within-band flip the two chains legitimately diverge, so the consumer
stops comparing that prompt. The top-2 pair is recorded at every step,
so the consumer never needs a second rule.

Pins: intra-op threads = 1, the expert dispatch pinned to `eager` (the
unpinned `from_pretrained` default resolves to `grouped_mm`, a different
accumulation formulation), and the M06 `_attn_implementation` pin of
`sdpa`. The model battery (loading, untied-head and per-layer eager
asserts) is shared from gen_bf16_05_ids_to_logits_fixtures_Qwen3.6-35B-A3B.py, so
both generators probe the identical pin set; its error messages carry that
module name.

Run from the worktree root under the declared project interpreter, twice;
the bytes of both runs must be identical before the fixtures are
installed:
  cd <worktree root> && .venv/bin/python \
    workspace/transformers/tests/testgen/gen_bf16_07_greedy_fixtures_Qwen3.6-35B-A3B.py


RAM: one model-resident process globally. The script verifies a
free+inactive+speculative pool above 32 GiB and that no other
python/torch process is running before it loads anything; the floor is
sized to the measured anonymous peak under the mmap-backed vendored
loader (the 70 GB weight stack rides file-backed pages), the same
measurement the ids-inference fixtures recorded.
"""

import hashlib
import json
import zipfile


def write_json_zip(path, obj, ensure_ascii=True):
    """Write obj as a single-entry deflate .json.zip, entry named after the
    uncompressed file. JSON fixtures stay reviewable via the generator and
    the archive stays out of text diffs."""
    payload = json.dumps(
        obj, sort_keys=True, indent=2, ensure_ascii=ensure_ascii
    ).encode("utf-8") + b"\n"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr(os.path.basename(path)[:-len(".zip")], payload)

import os
import subprocess
import sys

def _load_sibling(filename: str):
    """Execute and return a testgen generator module by filename. The naming
    law allows dots and hyphens that import syntax rejects, so the module is
    loaded from its file path under a derived name. A second call returns
    the cached module, never a second execution of its top level.
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



# The shared model battery (pins, identity checks) lives in the ids-inference
# generator, keeping both fixture generators on the identical pin set.
# Naming-law filename characters are beyond import syntax, so the module
# comes from its path, with the shared names unpacked by attribute.

_ids_inference = _load_sibling("gen_bf16_05_ids_to_logits_fixtures_Qwen3.6-35B-A3B.py")
VENDORED_SHA = _ids_inference.VENDORED_SHA
NUM_THREADS = _ids_inference.NUM_THREADS
MODEL_NAME = _ids_inference.MODEL_NAME
MODEL_DIR = _ids_inference.MODEL_DIR
INDEX_PATH = _ids_inference.INDEX_PATH
load_wrapper_config = _ids_inference.load_wrapper_config
build_model = _ids_inference.build_model
check_vendored_sha = _ids_inference.check_vendored_sha

import transformers  # noqa: E402
import torch

torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import AutoTokenizer  # noqa: E402

TRANSFORMERS_VERSION = transformers.__version__

GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "greedy-decoding", "Qwen3.6-35B-A3B"
)

# 3 distinct simple English prompts, each tokenized to at most 12
# tokens by this checkpoint's tokenizer, asserted below. Horizons stay
# inside 32 greedy steps.
PROMPT_SPEC = [
    ("Hello, how are you?", 32),
    ("The capital of France is", 32),
    ("Big blue whales eat krill.", 32),
]
MAX_PROMPT_TOKENS = 12
MAX_HORIZON = 32
# Near-tie band in bf16 ulps at the step's recorded max |logit| (R12).
TIE_FLIP_BAND_ULPS = 2

# Pool floor, same formula and same constant as the ids-inference generator.
MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_07_greedy_fixtures_Qwen3.6-35B-A3B] vm_stat gave no page size line")


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
        raise SystemExit("[gen_bf16_07_greedy_fixtures_Qwen3.6-35B-A3B] vm_stat gave no pool lines")
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
    """Refuse to load weights when the memory pool is low or another
    python/torch process holds RAM (this process chain is excluded from
    the pgrep match, whose command line spells the torch dependency of
    this run)."""
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_07_greedy_fixtures_Qwen3.6-35B-A3B] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_07_greedy_fixtures_Qwen3.6-35B-A3B] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def prompt_token_ids(tokenizer, text: str) -> tuple[int, ...]:
    """Prompt ids under the checkpoint tokenizer, no special tokens added
    (no bos prepended), from a SINGLE input: transformers' tokenizer
    returns a BatchEncoding (a UserDict with no ids attribute), whose
    input_ids are a flat list for one unpadded prompt."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    assert len(ids) >= 1 and len(ids) <= MAX_PROMPT_TOKENS, (
        f"prompt {text!r} holds {len(ids)} tokens, expected 1..{MAX_PROMPT_TOKENS}")
    return tuple(int(t) for t in ids)


def generation_eos_ids() -> list[int]:
    """The wrapper's stop ids from generation_config.json. The greedy
    chains below assert none of these ids: every step must stay an argmax
    over live vocabulary, and a chain that reached eos could not be
    extended token-exactly."""
    path = os.path.join(MODEL_DIR, "generation_config.json")
    with open(path) as f:
        raw = json.load(f)["eos_token_id"]
    if raw is None:
        return []
    return [int(raw)] if isinstance(raw, int) else [int(t) for t in raw]


def greedy_chain(model, token_ids: tuple[int, ...], max_new_tokens: int) -> dict:
    """Greedy-decode one unpadded chain through the vendored forward:
    prefill over the whole prompt with the returned past-key cache, then
    one single-token forward per step feeding back the argmax id.

    Returns the generated ids, per-step float32 checksums of the
    deciding last-position logits row, and per-step top-2 pairs (float32
    values and token ids). Exactly max_new_tokens argmax picks are
    recorded; the cache makes each step one O(1)-length forward."""
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long)
    generated = []
    checksums = []
    tie_logits = []
    tie_ids = []
    tie_steps = [0]
    eos = generation_eos_ids()
    with torch.no_grad():
        out = model(input_ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        for step in range(max_new_tokens):
            last_f32 = out.logits[0, -1].float()
            checksums.append(float(last_f32.sum().item()))
            # The logits quantize onto a bf16 grid, so exact fp32 argmax
            # ties are structural. torch.topk's tie order for two equal
            # values follows a selection-internal detail that flips with k.
            # First-hand probe: topk(k=2) and topk(k=5) ordered the 17.125
            # pair of that step differently.
            # The chain feeds torch.argmax, so the recorded pair anchors on it:
            # slot 0 carries the argmax id and value, slot 1 the best
            # remaining id, from argmax with its entry masked out.
            argmax_id = int(last_f32.argmax().item())
            top1_value = float(last_f32[argmax_id].item())
            masked = last_f32.clone()
            masked[argmax_id] = float("-inf")
            runner2 = torch.topk(masked, 1)
            runner_id = int(runner2.indices[0].item())
            runner_value = float(runner2.values[0].item())
            generated.append(argmax_id)
            assert argmax_id not in eos, (
                f"chain hit a configured eos id {argmax_id} at step {step}; "
                "prompt or horizon must be respecified")
            if top1_value == runner_value:
                tie_steps[0] += 1
            tie_logits.append([top1_value, runner_value])
            tie_ids.append([argmax_id, runner_id])
            if step + 1 < max_new_tokens:
                nxt = torch.tensor([[argmax_id]], dtype=torch.long)
                out = model(nxt, past_key_values=cache, use_cache=True,
                            logits_to_keep=1)
                cache = out.past_key_values
    return {
        "generated_ids": generated,
        "step_logits_checksum": checksums,
        "near_tie_logits": tie_logits,
        "near_tie_ids": tie_ids,
        "tied_steps": tie_steps[0],
    }


def prompt_fixture_name(text: str, max_new_tokens: int) -> str:
    """File name for one prompt fixture: alphanumerics only, horizon
    suffix, one file per prompt."""
    safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    safe = "_".join(part for part in safe.split("_") if part)
    return f"{safe}_{max_new_tokens}_steps.json.zip"


def main() -> None:
    print(f"Generating {MODEL_NAME} greedy-decoding fixtures")
    print("=" * 60)
    check_ram()
    sha = check_vendored_sha()

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
        assert len(chain["step_logits_checksum"]) == max_new_tokens
        assert len(chain["near_tie_logits"]) == max_new_tokens
        assert len(chain["near_tie_ids"]) == max_new_tokens
        fixture = {
            "schema": "tt-qwen36-greedy-1",
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "input_tokens": list(token_ids),
            "num_input_tokens": len(token_ids),
            "generated_ids": chain["generated_ids"],
            "step_logits_checksum": chain["step_logits_checksum"],
            "near_tie": {
                "logits": chain["near_tie_logits"],
                "ids": chain["near_tie_ids"],
                "band_ulps": TIE_FLIP_BAND_ULPS,
                "tied_steps": chain["tied_steps"],
                "clause": "an argmax divergence is a near-tie when the "
                          "diverging pick equals near_tie.ids[step][1] and "
                          "the top-2 gap at that step is at most "
                          "band_ulps * bf16Ulp(max |near_tie.logits[step]|); "
                          "comparison stops at the flip step (R12)",
            },
            "max_new_tokens": max_new_tokens,
            "num_threads": NUM_THREADS,
            "experts_implementation": model.config.text_config._experts_implementation,
            "attn_implementation": cfg._attn_implementation,
            "torch_version": torch.__version__,
            "transformers_version": TRANSFORMERS_VERSION,
            "vendored_sha": sha,
            "dtype": "bfloat16",
            "device": "cpu",
            "note": "single unpadded chains; the serialized values are the "
                    "single-chain values, no batched pass backs them",
        }
        out_path = os.path.join(
            FIXTURE_DIR, prompt_fixture_name(prompt_text, max_new_tokens))
        write_json_zip(out_path, fixture, ensure_ascii=False)
        digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        print(f"  prompt ({len(token_ids)} tokens): {prompt_text!r}")
        print(f"    generated: {chain['generated_ids']}")
        print(f"    fixture: {out_path}")
        print(f"    sha256: {digest}")
        print(f"    step0 checksum {chain['step_logits_checksum'][0]:.4f}")

    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
