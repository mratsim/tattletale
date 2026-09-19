#!/usr/bin/env python3
"""Full-forward-to-logits fixtures of the North-Mini-Code-1.0 checkpoint,
recorded through the installed transformers modeling on torch bf16, Metal (mps).
The chain covers embed, all 49 decoder layers, the final norm and lm_head.

- fixture dir tests/fixtures/bf16-03-full-forward-to-logits/North-Mini-Code-1.0/
- consumer tests/q_bf16/t_bf16_north_03_full_forward_to_logits.nim

| file                                   | contents                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| layer-<i>.safetensor                   | the chain boundary slice layer_input_seq, the routed layers also the topk rows |
| layer-<i>.safetensor.metadata.json.zst | the layer identity, the layer kinds and the routed-decision margins            |
| layer-<i>.safetensor.stats.json.zst    | the ttt-tf-004-uniform-stats frame over the floating-point payload tensors     |
| final_logits.decisions.json.zst        | the ttt-tf-005-argmax-decisions frame, one record per input position           |

- the prompt token ids are hard-asserted against the recorded corpus ids,
  a tokenizer drift fails the recording
- the forward runs twice, every capture and the logits assert run-to-run
  equal before anything is written
- routed layers carry the topk_indices and routing_weights tensors plus
  per-layer boundary margins of the sigmoid top-k selection, the expert ids
  compare on the recorded margins, floor 1e-4

The reference runs the eager expert loop on both recording and replay
sides alike, no dispatch band measurement exists for this family.

At the 6-token prompt length both mask kinds skip to the sdpa is_causal
path (mask None). The 4096-token window does not constrain this prompt.
The tier-04 records carry the window behavior.

Recording environment:

- recorded_from defaults to m4max-metal, the TTT_RECORD_FROM environment
  variable overrides it
- recording runs on mps (Metal), the replay device stays the consuming suite's call

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_north_03_full_forward_to_logits.py

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
from transformers import AutoTokenizer, Cohere2MoeForCausalLM  # noqa: E402

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
MODEL_NAME = "North-Mini-Code-1.0"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-03-full-forward-to-logits", MODEL_NAME
)

NUM_THREADS = 1

# Decision records written, one per input position over the top-32 support.
NUM_POSITIONS = 6

# Prompt corpus id, measured against the checkpoint tokenizer through
# AutoTokenizer over the shipped tokenizer.json, BOS included:
# - the 6-token prompt keeps the per-layer payloads inside
#   the 1.5 MiB family dir budget over the 49 layers
INPUT_TEXT = "The sky is blue today"
PROMPT_IDS = [2, 669, 14198, 341, 7872, 3681]

MIN_POOL_BYTES = 64 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_north_03_full_forward_to_logits] vm_stat gave no page size line")


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
            "[gen_bf16_north_03_full_forward_to_logits] vm_stat gave no pool lines")
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
      this process chain (its own command line spells the torch dependency)
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_north_03_full_forward_to_logits] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_north_03_full_forward_to_logits] other python/torch processes hold RAM: {stray}, "
            "stop and retry when idle")


def load_tokenizer():
    """Tokenizes the prompt through the checkpoint tokenizer.json, special
    tokens added (the BOS id 2 opens the recorded id list)."""
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def build_model() -> Cohere2MoeForCausalLM:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps).

    Returns:
    - the model in eval mode, the decoder layers carry the capture wrappers

    Identity asserts of the checkpoint config, f-first 4:1 shape:

    - full_attention at layers 0, 4, ... 48, sliding_attention between, window 4096
    - the dense prefix at layer 0, intermediate 3072, the routed block everywhere else
    - the sigmoid top-k router, 128 experts with top 8, no renorm, no shared experts
    """
    model = Cohere2MoeForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    cfg = model.config
    assert cfg.layer_types[0] == "full_attention", (
        "layer 0 must be the full_attention row of the f-first 4:1 pattern")
    assert cfg.layer_types[1] == "sliding_attention", (
        "layer 1 must be a sliding_attention row of the f-first 4:1 pattern")
    assert cfg.layer_types[4] == "full_attention", (
        "layer 4 must be the full_attention row of the f-first 4:1 pattern")
    assert cfg.mlp_layer_types[0] == "dense", (
        "layer 0 must be the dense prefix row")
    assert all(t == "sparse" for t in cfg.mlp_layer_types[1:]), (
        "every layer past the dense prefix must route")
    assert cfg.prefix_dense_intermediate_size == 3072, (
        "the dense prefix intermediate must stay the checkpoint 3072")
    assert cfg.sliding_window == 4096, (
        "the recorded rows assume the 4096 sliding window")
    assert cfg.rope_parameters["rope_theta"] == 50000, (
        "the rope theta must stay the single global 50000")
    assert cfg.expert_selection_fn == "sigmoid" and cfg.norm_topk_prob is False, (
        "the recorded router is the sigmoid top-k without renorm")
    assert cfg.num_experts == 128 and cfg.num_experts_per_tok == 8, (
        "the recorded rows assume 128 experts with top 8")
    assert cfg.num_shared_experts == 0, (
        "the checkpoint ships no shared experts")
    assert cfg.num_hidden_layers == 49 and cfg.hidden_size == 2048 \
        and cfg.vocab_size == 262144, (
        "the recorded rows assume the 49-layer, 2048-hidden, 262144-vocab checkpoint")
    return model


def capture_forward(model: Cohere2MoeForCausalLM, input_ids: torch.Tensor) -> tuple:
    """Runs one model forward with a capture wrapper on every decoder layer
    plus a router hook on every routed block, restoring the pristine
    forwards after the run.

    Args:
    - model, input_ids, the loaded reference model and the [1, seq] prompt ids

    Returns:
    - the capture list, one (layer_input, layer_output) pair per layer
    - the router rows, one (logits, weights, indices) triple per routed layer, None on the dense prefix
    - the final logits tensor
    """
    layers = model.model.layers
    captured = [None] * len(layers)
    routed = [None] * len(layers)
    originals = []
    handles = []

    def make_wrapper(layer_idx, layer, original):
        def wrapper(hidden_states, *args, **kwargs):
            entry = {"layer_input": hidden_states.clone()}
            output = original(hidden_states, *args, **kwargs)
            entry["layer_output"] = output.clone()
            captured[layer_idx] = entry
            return output

        return wrapper

    def make_router_hook(layer_idx):
        def hook_impl(module, args, output):
            logits, weights, indices = output
            routed[layer_idx] = {
                "router_logits": logits.clone(),
                "topk_weights": weights.clone(),
                "topk_indices": indices.clone(),
            }

        return hook_impl

    for i, layer in enumerate(layers):
        originals.append(layer.forward)
        layer.forward = make_wrapper(i, layer, originals[-1])
        if model.config.mlp_layer_types[i] == "sparse":
            handles.append(layer.mlp.gate.register_forward_hook(make_router_hook(i)))
    try:
        with torch.no_grad():
            output = model(input_ids, use_cache=False)
    finally:
        for layer, original in zip(layers, originals):
            layer.forward = original
        for handle in handles:
            handle.remove()
    return captured, routed, output.logits


def boundary_margins(router_logits: torch.Tensor, top_k: int) -> dict:
    """Top-k selection margins under the sigmoid scoring, the checkpoint
    router adds no selection bias.

    - boundary_min, the smallest Kth-vs-K+1th gap over all rows, an exact
      f32 boundary tie reads 0.0, boundary_per_row carries one margin per token
    - tie_rows, the order-ambiguous rows of the recorded expert ids, the rows whose boundary gap ties exactly
    - inner_gap_min, the smallest adjacent gap inside the top-k set, the ambiguity floor of the recorded expert order
    """
    scores = router_logits.float().sigmoid()
    sorted_scores = torch.sort(scores, dim=-1, descending=True).values
    gaps = sorted_scores[:, top_k - 1] - sorted_scores[:, top_k]
    inner = sorted_scores[:, : top_k - 1] - sorted_scores[:, 1:top_k]
    return {
        "boundary_min": gaps.min().item(),
        "boundary_per_row": gaps.tolist(),
        "tie_rows": (gaps == 0.0).nonzero().flatten().tolist(),
        "inner_gap_min": inner.min().item(),
    }


def boundary_tie_count(router_logits: torch.Tensor, top_k: int) -> int:
    """Count tokens whose top-k boundary selection scores tie exactly
    within f32, such a tie makes the selected last expert order-sensitive."""
    scores = router_logits.float().sigmoid()
    sorted_scores = torch.sort(scores, dim=-1, descending=True).values
    return int(
        (sorted_scores[:, top_k - 1] == sorted_scores[:, top_k]).sum().item())


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
    cfg = model.config
    num_layers = len(model.model.layers)

    captured, routed, logits = capture_forward(model, input_ids)
    captured_again, routed_again, logits_again = capture_forward(model, input_ids)
    for i in range(num_layers):
        assert torch.equal(captured[i]["layer_input"], captured_again[i]["layer_input"]), (
            f"recorded run layer {i} input is not deterministic")
        assert torch.equal(captured[i]["layer_output"], captured_again[i]["layer_output"]), (
            f"recorded run layer {i} output is not deterministic")
        if routed[i] is not None:
            assert torch.equal(routed[i]["router_logits"], routed_again[i]["router_logits"]), (
                f"recorded run layer {i} router logits are not deterministic")
    assert torch.equal(logits, logits_again), "recorded run logits are not deterministic"
    for i in range(num_layers - 1):
        assert torch.equal(captured[i]["layer_output"], captured[i + 1]["layer_input"]), (
            f"layer {i} output must feed layer {i + 1} as its input")

    for i in range(num_layers):
        layer_type = cfg.layer_types[i]
        mlp_type = cfg.mlp_layer_types[i]
        rope_applied = layer_type == "sliding_attention" or (
            mlp_type == "dense" and cfg.prefix_dense_sliding_window_pattern == 1)
        tensors = {"layer_input_seq": captured[i]["layer_input"]}
        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.layers.{i}",
            "layer_type": layer_type,
            "mlp_type": mlp_type,
            "rope_applied": rope_applied,
            "rope_theta": cfg.rope_parameters["rope_theta"],
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
            "mask_note": "the 6-token prompt stays inside the 4096 window, both "
                "mask kinds skip to the sdpa is_causal path (mask None), the "
                "window behavior is recorded at tier 04",
        }
        if mlp_type == "sparse":
            gate_row = routed[i]
            top_k = cfg.num_experts_per_tok
            margins = boundary_margins(gate_row["router_logits"], top_k)
            ties = boundary_tie_count(gate_row["router_logits"], top_k)
            # A zero boundary gap is an exact f32 tie. The checkpoint router
            # adds no selection bias so ties recur across prompts, the layer
            # records tie_exception plus tie_rows as the recorded exception.
            assert margins["boundary_min"] > 0.0 or ties >= 1, (
                f"layer {i} boundary gap {margins['boundary_min']} is zero "
                "without a counted tie, inconsistent margin record")
            metadata["margins"] = {
                "topk_margin_min": margins["boundary_min"],
                "topk_margin_per_row": margins["boundary_per_row"],
                "tie_rows": margins["tie_rows"],
                "inner_gap_min": margins["inner_gap_min"],
                "boundary_tie_tokens": ties,
            }
            if margins["boundary_min"] == 0.0:
                metadata["tie_exception"] = True
            metadata["flip_budget_policy"] = (
                "exact expert-id comparisons in the consuming suites stand on "
                "a recorded positive boundary margin per row with the 1e-4 "
                "floor, the checkpoint router adds no selection bias so "
                "exact f32 boundary ties occur, a tie layer carries "
                "tie_exception true and its tie_rows name the ambiguous "
                "rows, an explicit per-fixture recorded exception, never a "
                "silent widening")
            metadata["margin_floor"] = 1e-4
            metadata["router"] = "sigmoid_topk"
            tensors["topk_indices"] = gate_row["topk_indices"]
            tensors["routing_weights"] = gate_row["topk_weights"]
        if i == num_layers - 1:
            tensors["layer_output_seq"] = captured[i]["layer_output"]
        filepath = save_fixture(f"layer-{i:02d}", metadata, tensors)
        margin_note = (f", topk_margin {margins['boundary_min']:.3e}"
                       if "margins" in metadata else "")
        print(f"  layer {i:02d} ({layer_type}, {mlp_type}){margin_note}: {filepath}")

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
