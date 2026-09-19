#!/usr/bin/env python3
"""Full-forward-to-logits fixtures of the gemma-3-270m-it checkpoint,
recorded through the installed transformers modeling on torch bf16, CPU.
The chain covers embed, all 18 decoder layers, the final norm and lm_head.

- fixture dir tests/fixtures/bf16-03-full-forward-to-logits/gemma-3-270m-it/
- consumer tests/q_bf16/t_bf16_gemma3270m_03_full_forward_to_logits.nim

| file                                   | contents                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| layer-<i>.safetensor                   | the chain boundary slice layer_input_seq, the last layer also layer_output_seq |
| layer-<i>.safetensor.metadata.json.zst | the layer identity, the layer kind and its rope theta                          |
| layer-<i>.safetensor.stats.json.zst    | the ttt-tf-004-uniform-stats frame over the payload tensors                    |
| final_logits.decisions.json.zst        | the ttt-tf-005-argmax-decisions frame, one record per input position           |

- the prompt token ids are hard-asserted against the recorded corpus ids,
  a tokenizer drift fails the recording
- the forward runs twice, every capture and the logits assert run-to-run
  equal before anything is written

At the 7-token prompt length both mask kinds skip to the sdpa is_causal
path (mask None). The 512-token window does not constrain this prompt.
The tier-04 records carry the window behavior.

Recording environment:

- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment
  variable overrides it
- recording runs on cpu, the replay device stays the consuming suite's call

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_gemma3270m_03_full_forward_to_logits.py

RAM guard:

- the script refuses the weight load when the free+inactive+speculative pool
  sits below 8 GiB
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
from transformers import AutoTokenizer, Gemma3ForCausalLM  # noqa: E402

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
MODEL_NAME = "gemma-3-270m-it"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-03-full-forward-to-logits", MODEL_NAME
)

NUM_THREADS = 1

# Decision records written, one per input position over the top-32 support.
NUM_POSITIONS = 6

# Prompt corpus id, measured against the checkpoint tokenizer through
# AutoTokenizer over the shipped tokenizer.json, BOS included.
INPUT_TEXT = "Hello, how are you?"
PROMPT_IDS = [2, 9259, 236764, 1217, 659, 611, 236881]

MIN_POOL_BYTES = 8 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_gemma3270m_03_full_forward_to_logits] vm_stat gave no page size line")


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
            "[gen_bf16_gemma3270m_03_full_forward_to_logits] vm_stat gave no pool lines")
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

    - the free+inactive+speculative pool sits above the 8 GiB floor
    - no other python/torch process holds RAM, the pgrep match excludes
      this process chain (its own command line spells the torch dependency)
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_gemma3270m_03_full_forward_to_logits] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_gemma3270m_03_full_forward_to_logits] other python/torch processes hold RAM: {stray}, "
            "stop and retry when idle")


def load_tokenizer():
    """Tokenizes the prompt through the checkpoint tokenizer.json, special
    tokens added (the BOS id 2 opens the recorded id list)."""
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def build_model() -> Gemma3ForCausalLM:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, CPU.

    Returns:
    - the model in eval mode, the decoder layers carry the capture wrappers

    Identity asserts of the checkpoint config:

    - the 5:1 sliding pattern, layer 4 sliding_attention and layer 5
      full_attention (the boundary pair of the tier-01 family)
    - dual rope theta, 1e6 on full layers and 1e4 on sliding layers
    - the 512 sliding window
    """
    model = Gemma3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()
    cfg = model.config
    assert cfg.layer_types[4] == "sliding_attention", (
        "layer 4 must be the sliding_attention row of the 5:1 pattern")
    assert cfg.layer_types[5] == "full_attention", (
        "layer 5 must be the full_attention row of the 5:1 pattern")
    assert cfg.rope_parameters["full_attention"]["rope_theta"] == 1e6, (
        "the full-attention rope theta must stay the global 1e6")
    assert cfg.rope_parameters["sliding_attention"]["rope_theta"] == 1e4, (
        "the sliding-attention rope theta must stay the local 1e4")
    assert cfg.sliding_window == 512, (
        "the recorded rows assume the 512 sliding window")
    return model


def capture_forward(model: Gemma3ForCausalLM, input_ids: torch.Tensor) -> tuple:
    """Runs one model forward with a capture wrapper on every decoder layer,
    restoring the pristine forwards after the run.

    Args:
    - model, input_ids, the loaded reference model and the [1, seq] prompt ids

    Returns:
    - the capture list, one (layer_input, layer_output) pair per layer
    - the final logits tensor
    """
    layers = model.model.layers
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
    recorded_from = os.environ.get("TTT_RECORD_FROM", "m4max-cpu")
    check_ram()
    print(f"Generating {MODEL_NAME} bf16-03-full-forward-to-logits fixtures")
    print(f"transformers {transformers.__version__}")

    tokenizer = load_tokenizer()
    tokenizer_ids = tokenizer(INPUT_TEXT)["input_ids"]
    assert tokenizer_ids == PROMPT_IDS, (
        f"prompt ids {tokenizer_ids} disagree with the recorded corpus "
        f"{PROMPT_IDS}, the tokenizer path moved")
    input_ids = torch.tensor([tokenizer_ids])
    seq_len = input_ids.shape[1]

    model = build_model()
    cfg = model.config
    num_layers = len(model.model.layers)

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
        layer_type = cfg.layer_types[i]
        tensors = {"layer_input_seq": captured[i]["layer_input"]}
        if i == num_layers - 1:
            tensors["layer_output_seq"] = captured[i]["layer_output"]
        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.layers.{i}",
            "layer_type": layer_type,
            "rope_theta": cfg.rope_parameters[layer_type]["rope_theta"],
            "input_text": INPUT_TEXT,
            "input_tokens": tokenizer_ids,
            "batch_size": 1,
            "seq_len": seq_len,
            "num_threads": NUM_THREADS,
            "dtype": "bfloat16",
            "device": "cpu",
            "recorded_from": recorded_from,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "mask_note": "the 7-token prompt stays inside the 512 window, both "
                "mask kinds skip to the sdpa is_causal path (mask None), the "
                "window behavior is recorded at tier 04",
        }
        filepath = save_fixture(f"layer-{i:02d}", metadata, tensors)
        print(f"  layer {i:02d} ({layer_type}): {filepath}")

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
