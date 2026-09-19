#!/usr/bin/env python3
"""Full-forward-to-logits fixtures of the gemma-4-26B-A4B checkpoint,
recorded through the installed transformers modeling on torch bf16, Metal (mps).
The chain covers embed, all 30 decoder layers, the final norm and the softcapped lm_head.

- fixture dir tests/fixtures/bf16-03-full-forward-to-logits/gemma-4-26B-A4B/
- consumer tests/q_bf16/t_bf16_gemma426b_03_full_forward_to_logits.nim

| file                                   | contents                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| layer-<i>.safetensor                   | the chain boundary slice layer_input_seq, the last layer also layer_output_seq |
| layer-<i>.safetensor.metadata.json.zst | the layer identity, the layer kind, the dual head dims and the k_eq_v row      |
| layer-<i>.safetensor.stats.json.zst    | the ttt-tf-004-uniform-stats frame over the payload tensors                    |
| final_logits.decisions.json.zst        | the ttt-tf-005-argmax-decisions frame, one record per input position           |

- the prompt token ids are hard-asserted against the recorded corpus ids,
  a tokenizer drift fails the recording
- the forward runs twice, every capture and the logits assert run-to-run
  equal before anything is written

The 6-token prompt keeps the per-layer payloads inside the 1.5 MiB family dir
budget over the 30 layers at the 2816 hidden width. Both mask kinds skip to the sdpa
is_causal path (mask None) at that length.

The 1024-token window does not constrain these rows.

Tier-04 carries the window behavior.

Every decoder layer pairs the dense mlp (intermediate 2112) with the routed block,
128 experts top 8 over moe_intermediate 704.

Recording environment:

- recorded_from defaults to m4max-metal, the TTT_RECORD_FROM environment
  variable overrides it
- recording runs on mps (Metal), the replay device stays the consuming suite's call

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_gemma426b_03_full_forward_to_logits.py

RAM guard:

- the script refuses the weight load when the free+inactive+speculative pool
  sits below 64 GiB
- another python/torch process holding RAM also blocks the run
"""

from collections import OrderedDict
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402, the path insert precedes the import
from safetensors import torch as st  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration  # noqa: E402

from fixture_stats import (  # noqa: E402, the path insert precedes the import
    argmax_record_from_row,
    grid_of,
    write_argmax_decisions,
    write_stats_file,
)

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity constants of the checkpoint.
MODEL_NAME = "gemma-4-26B-A4B"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-03-full-forward-to-logits", MODEL_NAME
)

NUM_THREADS = 1

# Decision records written, one per input position over the top-32 support.
NUM_POSITIONS = 6

# Prompt corpus id, measured against the checkpoint tokenizer through
# AutoTokenizer over the shipped tokenizer.json, the tokenizer prepends
# the BOS id 2 ahead of the prompt text:
# - the 6-token prompt keeps the per-layer payloads inside
#   the 1.5 MiB family dir budget over the 30 layers at hidden 2816
INPUT_TEXT = "The sky is blue today"
PROMPT_IDS = [2, 818, 7217, 563, 3730, 3124]

MIN_POOL_BYTES = 64 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_gemma426b_03_full_forward_to_logits] vm_stat gave no page size line")


def pool_bytes() -> int:
    """Free+inactive+speculative physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    wanted = ("Pages free:", "Pages inactive:", "Pages speculative:")
    pool = 0
    for line in out.stdout.splitlines():
        for label in wanted:
            if line.startswith(label):
                pool += int(line.split()[2].rstrip(".")) * page
    if pool == 0:
        raise SystemExit(
            "[gen_bf16_gemma426b_03_full_forward_to_logits] vm_stat gave no pool lines")
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
    """Refuses the weight load under low memory or a stray python/torch
    process holding RAM.

    Precondition:

    - the free+inactive+speculative pool sits above the 64 GiB floor
    - no other python/torch process holds RAM, the pgrep match excludes
      this process chain, whose own command line spells the torch dependency
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_gemma426b_03_full_forward_to_logits] free+inactive+speculative "
            f"pool {pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_gemma426b_03_full_forward_to_logits] other python/torch "
            f"processes hold RAM: {stray}, stop and retry when idle")


def load_tokenizer():
    """Tokenizes the prompt through the checkpoint tokenizer.json, which
    prepends the BOS id 2 ahead of the prompt text."""
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def build_model() -> Gemma4ForConditionalGeneration:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps).

    Returns:
    - the model in eval mode, the text decoder layers carry the capture wrappers

    Identity asserts of the checkpoint config, the dual-dim k_eq_v routed shape
    on every layer:

    | property      | recorded value                                                     |
    | ------------- | ------------------------------------------------------------------ |
    | layer pattern | 5 sliding_attention layers then 1 full_attention, window 1024      |
    | head dims     | sliding 256 over 8 kv heads, full 512 over 2 kv heads              |
    | attention     | attention_k_eq_v on the full layers, v_proj None, unscaled v_norm  |
    | block         | dense mlp 2112 plus the routed block everywhere, 128 experts top 8 |
    | routed widths | moe_intermediate 704, no kv sharing, no PLE                        |
    """
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    tcfg = model.config.text_config
    assert tcfg.layer_types[0] == "sliding_attention", (
        "layer 0 must be a sliding_attention row of the 5:1 pattern")
    assert tcfg.layer_types[5] == "full_attention", (
        "layer 5 must be a full_attention row of the 5:1 pattern")
    assert tcfg.sliding_window == 1024, (
        "the recorded rows assume the 1024 sliding window")
    assert tcfg.per_layer_config[0].head_dim == 256 and \
        tcfg.per_layer_config[5].head_dim == 512, (
        "the recorded rows assume the dual head dims 256 (sliding) / 512 (full)")
    assert tcfg.per_layer_config[5].num_key_value_heads == 2, (
        "the recorded rows assume the full layers run 2 kv heads")
    assert tcfg.attention_k_eq_v is True, (
        "the recorded rows assume attention_k_eq_v")
    assert tcfg.num_kv_shared_layers == 0 and \
        tcfg.hidden_size_per_layer_input == 0, (
        "the recorded rows assume no kv sharing and no PLE")
    assert tcfg.enable_moe_block is True, (
        "the recorded rows assume the routed block on every layer")
    assert tcfg.num_experts == 128 and tcfg.top_k_experts == 8, (
        "the recorded rows assume 128 experts with top 8")
    assert tcfg.moe_intermediate_size == 704 and tcfg.intermediate_size == 2112, (
        "the recorded rows assume moe_intermediate 704 beside the dense 2112")
    assert tcfg.final_logit_softcapping == 30.0, (
        "the recorded rows assume the 30.0 final logit softcapping")
    assert tcfg.num_hidden_layers == 30 and tcfg.hidden_size == 2816 \
        and tcfg.vocab_size == 262144, (
        "the recorded rows assume the 30-layer, 2816-hidden, 262144-vocab checkpoint")
    return model


def capture_forward(model: Gemma4ForConditionalGeneration,
                    input_ids: torch.Tensor) -> tuple:
    """Runs one model forward with a capture wrapper on every text decoder layer,
    restoring the pristine forwards after the run.

    Args:
    - model, input_ids, the loaded reference model and the [1, seq] prompt ids

    Returns:
    - the capture list, one (layer_input, layer_output) pair per text layer
    - the final logits tensor
    """
    layers = model.model.language_model.layers
    captured = [None] * len(layers)
    originals = []

    def make_wrapper(layer_idx, layer, original):
        def wrapper(hidden_states, *args, **kwargs):
            entry = {"layer_input": hidden_states.clone()}
            output = original(hidden_states, *args, **kwargs)
            entry["layer_output"] = output.clone()
            captured[layer_idx] = entry
            return output

        return wrapper

    for i, layer in enumerate(layers):
        originals.append(layer.forward)
        layer.forward = make_wrapper(i, layer, originals[-1])
    try:
        with torch.no_grad():
            output = model(input_ids, use_cache=False)
    finally:
        for layer, original in zip(layers, originals):
            layer.forward = original
    return captured, output.logits


def write_metadata_zst(path: str, metadata: dict) -> None:
    """Writes one metadata sidecar, pretty JSON inside one zstd frame, level
    19 with content size and checksum recorded in the frame header."""
    import compression.zstd

    zstd_options = {
        compression.zstd.CompressionParameter.compression_level: 19,
        compression.zstd.CompressionParameter.content_size_flag: 1,
        compression.zstd.CompressionParameter.checksum_flag: 1,
    }
    payload_json = json.dumps(
        metadata, sort_keys=True, indent=2, ensure_ascii=True
    ).encode("utf-8") + b"\n"
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload_json, options=zstd_options))


def save_fixture(name: str, metadata: dict, tensors: dict) -> str:
    """Saves one layer fixture, the safetensors payload plus the metadata
    and stats sidecars.

    Args:
    - name, the file stem
    - metadata, tensors, the metadata frame and the payload tensors
      (floating tensors get one uniform stats record each)

    Returns:
    - the written payload path
    """
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    filepath = os.path.join(FIXTURE_DIR, f"{name}.safetensor")
    sorted_tensors = OrderedDict(
        (key, tensor.detach().cpu().contiguous())
        for key, tensor in sorted(tensors.items())
        if tensor is not None
    )
    serialized = st.save(sorted_tensors, metadata=None)
    with open(filepath, "wb") as f:
        f.write(serialized)
    write_metadata_zst(filepath + ".metadata.json.zst", metadata)
    write_stats_file(filepath + ".stats.json.zst", os.path.basename(filepath), [
        (key, tensor) for key, tensor in sorted_tensors.items()
        if tensor.is_floating_point()
    ])
    return filepath


def main() -> None:
    """Records the full-forward fixture set after the RAM guard."""
    recorded_from = os.environ.get("TTT_RECORD_FROM", "m4max-metal")
    check_ram()
    print(f"Generating {MODEL_NAME} bf16-03-full-forward-to-logits fixtures")
    print(f"transformers {transformers.__version__}")

    tokenizer = load_tokenizer()
    tokenizer_ids = tokenizer(INPUT_TEXT)["input_ids"]
    assert tokenizer_ids == PROMPT_IDS, (
        f"prompt ids {tokenizer_ids} disagree with the recorded corpus "
        f"{PROMPT_IDS}, the tokenizer path moved")
    input_ids = torch.tensor([tokenizer_ids], device="mps")
    seq_len = input_ids.shape[1]

    model = build_model()
    tcfg = model.config.text_config
    num_layers = tcfg.num_hidden_layers

    captured, logits = capture_forward(model, input_ids)
    captured_again, logits_again = capture_forward(model, input_ids)
    for i in range(num_layers):
        assert torch.equal(captured[i]["layer_input"], captured_again[i]["layer_input"]), (
            f"recorded run layer {i} input is not deterministic")
        assert torch.equal(captured[i]["layer_output"], captured_again[i]["layer_output"]), (
            f"recorded run layer {i} output is not deterministic")
    assert torch.equal(logits, logits_again), "recorded run logits are not deterministic"
    for i in range(num_layers - 1):
        assert torch.equal(captured[i]["layer_output"], captured[i + 1]["layer_input"]), (
            f"layer {i} output must feed layer {i + 1} as its input")

    for i in range(num_layers):
        layer_type = tcfg.layer_types[i]
        tensors = {"layer_input_seq": captured[i]["layer_input"]}
        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{i}",
            "layer_type": layer_type,
            "head_dim": tcfg.per_layer_config[i].head_dim,
            "kv_tied": layer_type == "full_attention" and tcfg.attention_k_eq_v,
            "rope_theta": tcfg.rope_parameters[layer_type]["rope_theta"],
            "input_text": INPUT_TEXT,
            "input_tokens": tokenizer_ids,
            "batch_size": 1,
            "seq_len": seq_len,
            "num_threads": NUM_THREADS,
            "dtype": "bfloat16",
            "device": "mps",
            "recorded_from": recorded_from,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "mask_note": "the 6-token prompt stays inside the 1024 window, both "
                "mask kinds skip to the sdpa is_causal path (mask None), the "
                "window behavior is recorded at tier 04",
        }
        if i == num_layers - 1:
            tensors["layer_output_seq"] = captured[i]["layer_output"]
        filepath = save_fixture(f"layer-{i:02d}", metadata, tensors)
        print(f"  layer {i:02d} ({layer_type}, head_dim "
              f"{metadata['head_dim']}, kv_tied {metadata['kv_tied']}): {filepath}")

    if logits.shape[1] < NUM_POSITIONS:
        raise SystemExit(
            f"{MODEL_NAME}: the forward produced {logits.shape[1]} "
            f"positions, the script records {NUM_POSITIONS}")
    records = [
        argmax_record_from_row(logits[0, pos].detach().cpu().to(torch.float32))
        for pos in range(NUM_POSITIONS)]
    decisions_path = os.path.join(FIXTURE_DIR, "final_logits.decisions.json.zst")
    write_argmax_decisions(
        decisions_path, "final_logits.decisions", records, grid_of(logits))
    print(f"  decisions: {decisions_path}")

    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
