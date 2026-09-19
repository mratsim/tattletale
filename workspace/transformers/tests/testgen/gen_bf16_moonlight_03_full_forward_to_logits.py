#!/usr/bin/env python3
"""Moonlight-16B-A3B fixture generator over the real checkpoint shards, resolved
through the gitignored tests/hf_models symlink.

bf16-03 full-forward-to-logits records, the full embed -> 27 decoder layers ->
norm -> lm_head chain through installed transformers modeling on torch bf16:
- fixture dir tests/fixtures/bf16-03-full-forward-to-logits/Moonlight-16B-A3B/
- consumer tests/q_bf16/t_bf16_moonlight_03_full_forward_to_logits.nim

| file                            | contents                                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| layer-<i>.safetensor            | the chain boundary slices, layer_input_seq plus the topk and routing rows on MoE layers, the last layer output |
| .metadata.json.zst              | the layer identity, the recorded dispatch bands and margins                                                    |
| .stats.json.zst                 | the ttt-tf-004-uniform-stats frame over the floating-point payload tensors                                     |
| final_logits.decisions.json.zst | the ttt-tf-005-argmax-decisions frame, one record per position over the top-32 logits support                  |

Dispatch contract of this run:
- the reference runs the grouped_mm default
- the generator reruns with every MoE layer eager and records the divergence bands
- expert ids compare on the recorded margins, floor 1e-4

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root
  uv run python workspace/transformers/tests/testgen/gen_bf16_moonlight_03_full_forward_to_logits.py
"""
import argparse
import copy
import json
import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}


def write_json_zst(path, obj, ensure_ascii=True):
    """Write obj as one zstd frame, level 19, content size and checksum recorded in the frame header."""
    payload = json.dumps(
        obj, sort_keys=True, indent=2, ensure_ascii=ensure_ascii
    ).encode("utf-8") + b"\n"
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))

from collections import OrderedDict
import os
import subprocess
import sys


import torch  # noqa: E402
from safetensors import safe_open
from safetensors import torch as st

# The script directory first keeps the sibling generator imports working.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa: E402, the path insert precedes the import
    argmax_record_from_row,
    grid_of,
    write_argmax_decisions,
    write_stats_file,
)

from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: E402
    DeepseekV3ForCausalLM,
)

# The boundary-margin helper of the MoE-internals generator loads by file path, the testgen directory is no import package.
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


_moe_fixtures = _load_sibling("gen_bf16_moonlight_01_layer_internals.py")

import transformers  # noqa: E402
TRANSFORMERS_VERSION = transformers.__version__

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity constants of the checkpoint.
MODEL_NAME = "Moonlight-16B-A3B"

NUM_POSITIONS = 6
# Decision records written, one per input position.

INPUT_TEXT = "Hello, how are you?"
# The tiktoken corpus id of the prompt, measured against the tiktoken
# tokenizer of the checkpoint through AutoTokenizer trust_remote_code
# and verified against the tiktoken Rust engine.
PROMPT_IDS = [19180, 11, 1632, 554, 398, 30]

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-03-full-forward-to-logits", MODEL_NAME
)
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")

NUM_THREADS = 1

DEVICE = "cpu"
    # Recording device, the --device argument overrides the default, the metadata and env device rows record the value the run used.

# Counts of the real checkpoint (model.safetensors.index.json), values
# re-verified against the file at record time:
TOTAL_KEYS = 5344
NUM_LAYERS = 27
NUM_MOE_LAYERS = 26
FIRST_K_DENSE_REPLACE = 1
TOP_K = 6
NUM_EXPERTS = 64

# Eager-vs-grouped_mm band guards, the two expert formulations agree on one bf16-grid routed block, the class divergence
# accumulates through 26 routed layers of residual stream:
# - one routed block measured worst 2^-9 = 1 bf16 ulp on the gen-02 rows
# - the first recording measured the accumulated bands saturating the grown-residual bf16 grid
#   with the input band reaching 1.0 from layer 13 on and the output band reaching 3.0 at the last layer
# the bounds below carry 4x the measured scale, they catch gross
# generator or checkpoint drift, not the band itself, the true bands are
# measured per run, recorded per layer, asserted by the Nim suite:
# - the input guard 4.0
# - the output guard 12.0
# - the logits band 16.0
INPUT_BAND_GUARD = 4.00
OUTPUT_BAND_GUARD = 12.00
LOGITS_BAND_GUARD = 16.00

# Pool floor (free + inactive + speculative), the op RAM rule, the same
# 32 GiB constant the Qwen3.6-35B-A3B generator of the same fixture
# family and the MoE-internals generator of this model carry.
MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_moonlight_03_full_forward_to_logits] vm_stat gave no page size line")


def pool_bytes() -> int:
    """Free, inactive and speculative physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    wanted = ("Pages free:", "Pages inactive:", "Pages speculative:")
    pool = 0
    for line in out.stdout.splitlines():
        for label in wanted:
            if line.startswith(label):
                pool += int(line.split()[2].rstrip(".")) * page
    if pool == 0:
        raise SystemExit("[gen_bf16_moonlight_03_full_forward_to_logits] vm_stat gave no pool lines")
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
    """Refuse to load weights under low memory or a stray python/torch process, the pgrep match excludes this process chain."""
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_moonlight_03_full_forward_to_logits] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_moonlight_03_full_forward_to_logits] other python/torch processes hold RAM: {stray}, "
            "stop and retry when idle")


def index_census(weight_map: dict) -> None:
    """Check the checkpoint index against the recorded counts, 5344 tensors, all under model.* except the untied lm_head.weight."""
    total = len(weight_map)
    assert total == TOTAL_KEYS, (
        f"[gen_bf16_moonlight_03_full_forward_to_logits] checkpoint index holds {total} keys, "
        f"expected {TOTAL_KEYS}")
    lm_head = sum(1 for key in weight_map if key == "lm_head.weight")
    assert lm_head == 1, "exactly one lm_head.weight entry expected"
    model_prefixed = sum(1 for key in weight_map if key.startswith("model."))
    assert model_prefixed == TOTAL_KEYS - 1, (
        "every non-head tensor must sit under model.*, found " + str(model_prefixed))


def build_model(weight_map: dict) -> DeepseekV3ForCausalLM:
    """Full reference model from the real checkpoint through the installed
    `from_pretrained`, per-file streaming, mmap-backed, bf16, eval, CPU:
    - the expert dispatch stays at the default resolution, `grouped_mm`,
      the accumulation formulation the committed per-op MoE fixtures recorded
    - the per-layer expert configs are de-shared after load, each MoE layer
      gets a shallow config copy, pure data movement, so the eager-vs-grouped_mm
      band measurement flips one layer at a time, each copy carries the resolved dispatch value
    - raises SystemExit when head and embedding share storage or values,
      or when the resolved dispatch is not grouped_mm"""
    model = DeepseekV3ForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map=None)
    resolved = model.config._experts_implementation
    assert resolved == "grouped_mm", (
        f"from_pretrained resolved the experts dispatch to {resolved!r}, "
        "the recorded chain must run the grouped_mm accumulation the "
        "committed per-op MoE fixtures captured")
    model.eval().to(torch.bfloat16)

    for i, layer in enumerate(model.model.layers):
        if i < FIRST_K_DENSE_REPLACE:
            continue
        experts = layer.mlp.experts
        assert experts.config is model.config, (
            f"layer {i} experts config shares the model config object, "
            "the per-layer flip measurement needs de-shared configs")
        experts.config = copy.copy(experts.config)
        assert experts.config._experts_implementation == "grouped_mm", (
            f"layer {i} experts copy lost the resolved dispatch")
    for i, layer in enumerate(model.model.layers):
        if i < FIRST_K_DENSE_REPLACE:
            continue
        assert layer.mlp.experts.config is not model.config, (
            f"layer {i} experts config still shares the model config object")

    embed = model.model.embed_tokens.weight
    head = model.lm_head.weight
    if head.data_ptr() == embed.data_ptr():
        raise SystemExit(
            "[gen_bf16_moonlight_03_full_forward_to_logits] lm_head shares storage with embed_tokens, "
            "the untied head was silently tied")
    if torch.equal(head, embed):
        raise SystemExit(
            "[gen_bf16_moonlight_03_full_forward_to_logits] lm_head equals embed_tokens elementwise, "
            "the untied head was silently tied")
    assert head.shape == embed.shape, (
        f"untied head shape {tuple(head.shape)} disagrees with the embedding "
        f"{tuple(embed.shape)}")
    with safe_open(os.path.join(
            MODEL_DIR, weight_map["lm_head.weight"]), framework="pt") as f:
        raw_head = f.get_tensor("lm_head.weight")
    assert torch.equal(head, raw_head), (
        "loaded lm_head.weight disagrees bitwise with the checkpoint tensor")
    del raw_head
    return model


def install_layer_hooks(layers):
    """Wrap every decoder layer forward to record input and output tensors:
    - returns the capture list, one dict per layer, plus a restore closure
    - each install restores the pristine class forward after the run,
      so a second install never chains onto a previous wrapper"""
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

    def restore():
        for layer, original in zip(layers, originals):
            layer.forward = original

    return captured, restore


def install_router_hooks(model, layers):
    """Hook every routed-block router module to record its decision rows:
    - f32 logits, f32 renormalized-and-scaled weights and int64 indices,
      the values exactly as the reference router computed them
    - the dense layers carry no router module, only layers
      first_k_dense_replace onward carry hooks"""
    routed = [None] * len(layers)
    handles = []

    def hook(layer_idx):
        def hook_impl(module, args, output):
            logits, weights, indices = output
            routed[layer_idx] = {
                "router_logits": logits.clone(),
                "topk_weights": weights.clone(),
                "topk_indices": indices.clone(),
            }

        return hook_impl

    for i in moe_layer_indices(model):
        handles.append(layers[i].mlp.gate.register_forward_hook(hook(i)))

    def restore():
        for handle in handles:
            handle.remove()

    return routed, restore


def run_forward(model, input_ids: torch.Tensor, capture_routers: bool):
    """Run the model forward with per-layer capture, per-router capture when
    `capture_routers` is set, returning captures, router rows and logits."""
    layers = model.model.layers
    captured, restore_layer = install_layer_hooks(layers)
    if capture_routers:
        router_rows, restore_router = install_router_hooks(model, layers)
    else:
        router_rows = None
        restore_router = None
    with torch.no_grad():
        output = model(input_ids)
    restore_layer()
    if restore_router is not None:
        restore_router()
    return captured, router_rows, output.logits


def flip_experts(model, implementation: str, moe_layer_indices) -> None:
    """Set the experts dispatch on every routed layer config copy, the dense
    layers stay untouched."""
    for i in moe_layer_indices:
        model.model.layers[i].mlp.experts.config._experts_implementation = implementation


def moe_layer_indices(model) -> list:
    """Indices of the routed layers, first_k_dense_replace onward."""
    return list(range(FIRST_K_DENSE_REPLACE, len(model.model.layers)))


def boundary_margins(router_logits: torch.Tensor, bias: torch.Tensor,
                     top_k: int) -> dict:
    """Top-k selection margins under the NoauxTc sigmoid + bias scoring:
    - boundary_min, the smallest positive Kth-vs-K+1th gap over all rows
    - boundary_per_row, the per-row gap list, one margin per recorded token
      - inner_gap_min, the smallest adjacent gap inside the top-k set,
        the ambiguity floor of the recorded expert order"""
    scores = router_logits.sigmoid()
    scores_for_choice = scores + bias.unsqueeze(0)
    sorted_choice = torch.sort(
        scores_for_choice, dim=-1, descending=True).values
    gaps = sorted_choice[:, top_k - 1] - sorted_choice[:, top_k]
    inner = sorted_choice[:, : top_k - 1] - sorted_choice[:, 1:top_k]
    return {
        "boundary_min": gaps.min().item(),
        "boundary_per_row": gaps.tolist(),
        "inner_gap_min": inner.min().item(),
    }


def boundary_tie_count(router_logits: torch.Tensor, bias: torch.Tensor,
                       top_k: int) -> int:
    """Count tokens whose top-k boundary selection scores tie exactly
    within fp32, descending positions top_k-1 and top_k under sigmoid + bias:
    - the recorded topk_indices locks the torch.topk choice, such a tie
      makes the selected last expert order-sensitive"""
    scores = router_logits.sigmoid()
    scores_for_choice = scores + bias.unsqueeze(0)
    sorted_choice = torch.sort(
        scores_for_choice, dim=-1, descending=True).values
    return int(
        (sorted_choice[:, top_k - 1] == sorted_choice[:, top_k]).sum().item())


def save_fixture(name: str, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file plus the 004 stats sidecar."""
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
    metadata_path = filepath + ".metadata.json.zst"
    write_json_zst(metadata_path, metadata)
    write_stats_file(filepath + ".stats.json.zst", os.path.basename(filepath), [
        (name, tensor) for name, tensor in sorted_tensors.items()
        if tensor.is_floating_point()
    ])
    return filepath


def load_tokenizer():
    """Tokenize the prompt through the checkpoint tokenizer.json, no special token added."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def main() -> None:
    """Runs the full fixture recording end to end."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    print(f"Generating {MODEL_NAME} full-model full-forward-to-logits fixtures")
    print("=" * 60)
    check_ram()
    print(f"transformers {TRANSFORMERS_VERSION}")

    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    index_census(weight_map)
    model = build_model(weight_map)
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    resolved_impl = model.config._experts_implementation
    moe_indices = moe_layer_indices(model)
    assert len(moe_indices) == NUM_MOE_LAYERS, (
        f"{len(moe_indices)} routed layers, expected {NUM_MOE_LAYERS}")
    tokenizer = load_tokenizer()

    tokenizer_ids = tokenizer(INPUT_TEXT, add_special_tokens=False)["input_ids"]
    assert tokenizer_ids == PROMPT_IDS, (
        f"prompt ids {tokenizer_ids} disagree with the recorded corpus "
        f"{PROMPT_IDS}, the tokenizer path moved")
    input_ids = torch.tensor([tokenizer_ids], device=DEVICE)
    seq_len = input_ids.shape[1]
    layers = model.model.layers
    num_layers = len(layers)
    assert num_layers == NUM_LAYERS, (
        f"the checkpoint must carry {NUM_LAYERS} decoder layers")

    # Recorded run:
    #   the from_pretrained default dispatch, the dispatch
    # the committed per-op MoE fixtures captured. Twice, for determinism.
    captured, router_rows, logits = run_forward(model, input_ids, capture_routers=True)
    captured_again, _, logits_again = run_forward(model, input_ids, capture_routers=False)
    for i in range(num_layers):
        assert torch.equal(captured[i]["layer_output"], captured_again[i]["layer_output"]), (
            f"recorded run layer {i} is not deterministic")
    assert torch.equal(logits, logits_again), "recorded run logits are not deterministic"

    # The reference-internal eager-vs-grouped_mm class band, every routed
    # layer flips to eager, the model reruns, the accumulated per-layer
    # bands against the recorded run go to the metadata.
    flip_experts(model, "eager", moe_indices)
    eager_captured, _, logits_eager = run_forward(model, input_ids, capture_routers=False)
    flip_experts(model, "grouped_mm", moe_indices)
    restored_captured, _, logits_restored = run_forward(model, input_ids, capture_routers=False)
    for i in range(num_layers):
        assert torch.equal(captured[i]["layer_output"], restored_captured[i]["layer_output"]), (
            f"dispatch restoration diverged from the recorded run at layer {i}")
    assert torch.equal(logits, logits_restored), (
        "dispatch restoration diverged from the recorded run at the logits")

    max_input_band = 0.0
    max_output_band = 0.0
    for i in range(num_layers):
        layer_input = captured[i]["layer_input"]
        layer_output = captured[i]["layer_output"]
        eager_input = eager_captured[i]["layer_input"]
        eager_output = eager_captured[i]["layer_output"]

        if i < FIRST_K_DENSE_REPLACE:
            assert torch.equal(layer_input, eager_input) and torch.equal(layer_output, eager_output), (
                f"layer {i} is dense, the expert flip must leave it untouched")
            input_diff = 0.0
            output_diff = 0.0
        else:
            # The eager run diverges from the recorded run from the first
            # routed layer output onward:
            # - at layer 1 the input still agrees, the output band is the isolated class delta of that layer
            # - from layer 2 onward the input band is the class divergence
            #   accumulated upstream, exactly the quantity the metadata records
            input_diff = (eager_input.float() - layer_input.float()).abs().max().item()
            output_diff = (eager_output.float() - layer_output.float()).abs().max().item()
            assert input_diff < INPUT_BAND_GUARD, \
                f"eager vs grouped_mm layer {i} input diff too large: {input_diff}"
            assert output_diff < OUTPUT_BAND_GUARD, \
                f"eager vs grouped_mm layer {i} output diff too large: {output_diff}"
        max_input_band = max(max_input_band, input_diff)
        max_output_band = max(max_output_band, output_diff)

        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.layers.{i}",
            "layer_type": "dense" if i < FIRST_K_DENSE_REPLACE else "moe",
            "input_text": INPUT_TEXT,
            "input_tokens": tokenizer_ids,
            "batch_size": 1,
            "seq_len": seq_len,
            "num_threads": NUM_THREADS,
            "dtype": "bfloat16",
            "device": DEVICE,
            "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
            "experts_implementation": resolved_impl,
            "torch_version": torch.__version__,
            "transformers_version": TRANSFORMERS_VERSION,
            "bands": {
                "input_band": input_diff,
                "output_band": output_diff,
            },
            "note": "layer_input_seq is the grouped_mm recorded chain. The bands are "
                    "the reference-internal eager-vs-grouped_mm class divergence "
                    "accumulated to this boundary, measured in-generator by the "
                    "all-eager rerun. The Nim chain runs the eager-class "
                    "accumulation, so its drift against the recorded chain carries "
                    "this class on top of the op-level drift.",
        }
        tensors = {"layer_input_seq": layer_input}
        if i >= FIRST_K_DENSE_REPLACE:
            gate_row = router_rows[i]
            bias = layers[i].mlp.gate.e_score_correction_bias.detach().clone().float()
            margins = boundary_margins(gate_row["router_logits"], bias, TOP_K)
            ties = boundary_tie_count(gate_row["router_logits"], bias, TOP_K)
            assert margins["boundary_min"] > 0.0, (
                f"layer {i} top-k boundary margin {margins['boundary_min']} is not "
                "positive, the expert ids are order-ambiguous, pick a different prompt")
            metadata["margins"] = {
                "topk_margin_min": margins["boundary_min"],
                "topk_margin_per_row": margins["boundary_per_row"],
                "inner_gap_min": margins["inner_gap_min"],
                "boundary_tie_tokens": ties,
            }
            metadata["flip_budget_policy"] = (
                "exact expert-id comparisons in the consuming suites stand on a "
                "recorded positive boundary margin per row with the 1e-4 floor, "
                "a row below the floor carries "
                "an explicit per-fixture recorded exception, never a silent "
                "widening")
            metadata["margin_floor"] = 1e-4
            tensors["topk_indices"] = gate_row["topk_indices"]
            tensors["routing_weights"] = gate_row["topk_weights"]
        if i == num_layers - 1:
            tensors["layer_output_seq"] = layer_output
        save_fixture(f"layer-{i:02d}", metadata, tensors)
        margin_note = (f", topk_margin {margins['boundary_min']:.3e}"
                       if "margins" in metadata else "")
        print(f"  layer {i:02d} ({metadata['layer_type']}): input_band {input_diff:.3e}, "
              f"output_band {output_diff:.3e}{margin_note}")

    logits_band = (logits_eager.float() - logits.float()).abs().max().item()
    assert logits_band < LOGITS_BAND_GUARD, \
        f"eager vs grouped_mm logits diff too large: {logits_band}"
    # Decision projection of the recorded grouped_mm run, the raw logits tensor leaves the tree, the consumers carry the argmax id,
    # the top-32 support and the tail probability of each recorded position. The eager-vs-grouped_mm band stays
    # as the recorded metadata band.
    if logits.shape[1] < NUM_POSITIONS:
        raise SystemExit(
            f"{MODEL_NAME}: the forward produced {logits.shape[1]} "
            f"positions, the script records {NUM_POSITIONS}")
    records = [
        argmax_record_from_row(logits[0, pos].detach().cpu().to(torch.float32))
        for pos in range(NUM_POSITIONS)]
    write_argmax_decisions(
        os.path.join(FIXTURE_DIR, "final_logits.decisions.json.zst"),
        "final_logits.decisions", records, grid_of(logits))
    write_json_zst(os.path.join(FIXTURE_DIR, "final_logits.safetensor.metadata.json.zst"), {
        "model": MODEL_NAME,
        "input_text": INPUT_TEXT,
        "input_tokens": tokenizer_ids,
        "seq_len": seq_len,
        "num_threads": NUM_THREADS,
        "dtype": "bfloat16",
        "device": DEVICE,
        "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
        "experts_implementation": resolved_impl,
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
        "bands": {
            "logits_band": logits_band,
        },
        "note": "the projection is the grouped_mm recorded run, "
                "logits_band is the eager-vs-grouped_mm class band at "
                "the logits.",
    })
    # A summary of the recorded bands. The recorded bands remain
    # the only tolerance source on the Nim side, and these guards catch
    # generator or checkpoint drift at generation time.
    print(f"  max eager-vs-grouped input band {max_input_band:.3e} (guard {INPUT_BAND_GUARD})")
    print(f"  max eager-vs-grouped output band {max_output_band:.3e} (guard {OUTPUT_BAND_GUARD})")
    print(f"  eager-vs-grouped logits band {logits_band:.3e} (guard {LOGITS_BAND_GUARD})")
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print(f"Generated {num_layers} layer fixtures + final_logits under {FIXTURE_DIR}")
    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
