#!/usr/bin/env python3
"""Ling-3.0-tiny fixture generator over the real checkpoint shards, resolved through the gitignored tests/hf_models symlink.

bf16-03 full-forward-to-logits records, the full embed -> 24 decoder layers ->
norm -> lm_head chain through the reference modeling on torch bf16:
- fixture dir tests/fixtures/bf16-03-full-forward-to-logits/Ling-3.0-tiny/
- consumer tests/q_bf16/t_bf16_ling3_03_full_forward_to_logits.nim

- layer-<i>.safetensor, the chain boundary slices, layer_input_seq on every layer
- routed layers add the router input and the topk and routing rows, the last layer adds its output
- .metadata.json.zst, the layer identity, the recorded dispatch spelling and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors
- final_logits.decisions.json.zst, the ttt-tf-005-argmax-decisions frame, one record per position over the top-32 logits support

Recording contract of this run:
- every KDA layer runs the fused_recurrent dispatch at the 6-token prompt, q_len <= 64 on both sides, the port spells the same class
- the MoE dispatch is the reference eager token-gather class, no grouped_mm alternative exists
- expert ids compare on the recorded per-row router margins, floor 1e-4

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root, PYTHONPATH pointing at the kimi-linear reference worktree src with the torch fallback kernels:
  PYTHONPATH=<kimi-linear-reference-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_ling3_03_full_forward_to_logits.py
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors import torch as st  # noqa: E402

# Script-directory insertion keeps the sibling generator imports working.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa: E402, the path insert precedes the import
    argmax_record_from_row,
    grid_of,
    write_argmax_decisions,
    write_stats_file,
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


_ling01 = _load_sibling("gen_bf16_ling3_01_layer_internals_mla.py")

import transformers  # noqa: E402
TRANSFORMERS_VERSION = transformers.__version__

MODEL_NAME = "Ling-3.0-tiny"

NUM_POSITIONS = 6
# Decision records written, one per input position.

write_json_zst = _ling01.write_json_zst

INPUT_TEXT = "Hello, how are you?"
# Prompt corpus id, measured against the checkpoint tokenizer
# through AutoTokenizer over the shipped tokenizer.json, no
# trust_remote_code module involved, add_special_tokens False.
PROMPT_IDS = [14455, 11, 1099, 449, 362, 30]

FIXTURE_DIR = os.path.join(
    _ling01.TESTS_DIR, "fixtures", "bf16-03-full-forward-to-logits", MODEL_NAME
)
MODEL_DIR = _ling01.MODEL_DIR
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")

DEVICE = "cpu"
    # Recording device, the --device argument overrides the default, the metadata and env device rows record the value the run used.

# Counts of the real checkpoint (model.safetensors.index.json), values
# re-verified against the file at record time:
# - 9283 tensors, the embedding key model.word_embeddings, no draft block
#   (num_nextn_predict_layers 0), the untied lm_head.weight an independent parameter
TOTAL_KEYS = 9283
NUM_LAYERS = 24
NUM_MOE_LAYERS = 23
FIRST_K_DENSE_REPLACE = 1
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4


def index_census(weight_map: dict) -> None:
    """Check the checkpoint index against the recorded counts, 9283 tensors, all
under model.* except the untied lm_head.weight, no draft block."""
    total = len(weight_map)
    assert total == TOTAL_KEYS, (
        f"[gen_bf16_ling3_03] checkpoint index holds {total} keys, "
        f"expected {TOTAL_KEYS}")
    lm_head = sum(1 for key in weight_map if key == "lm_head.weight")
    assert lm_head == 1, "exactly one lm_head.weight entry expected"
    model_prefixed = sum(1 for key in weight_map if key.startswith("model."))
    assert model_prefixed == TOTAL_KEYS - 1, (
        "every non-head tensor must sit under model.*, found " + str(model_prefixed))
    word_embeddings = sum(
        1 for key in weight_map if key == "model.word_embeddings.weight")
    assert word_embeddings == 1, "exactly one word_embeddings entry expected"
    for key in weight_map:
        if key.startswith("model.layers."):
            rest = key[len("model.layers."):]
            layer_idx = int(rest.split(".", 1)[0])
            assert 0 <= layer_idx < NUM_LAYERS, (
                f"layer index {layer_idx} outside 0..{NUM_LAYERS - 1} in key {key}")


def build_model():
    """Full reference model from the real checkpoint through the installed
    `from_pretrained`, per-file streaming, mmap-backed, bf16, eval, CPU:
    - the rope_scaling restore runs on the config before module construction,
      the AutoConfig synthesis of the null file field would crash the MLA init
    - raises SystemExit when a dtype-plan row moves, the f32 A_log and dt_bias
      the reference creates with an explicit dtype, or the bf16 expert_bias
      buffer the reference deploys without one"""
    from transformers import AutoConfig  # noqa: E402
    from transformers.dynamic_module_utils import (  # noqa: E402
        get_class_from_dynamic_module)

    _ling01.apply_shims()
    _ling01.install_fla_stubs()
    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg.rope_scaling = None
    causal_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3.BailingMoeV3ForCausalLM", MODEL_DIR)
    model = causal_cls.from_pretrained(
        MODEL_DIR, config=cfg, dtype=torch.bfloat16, device_map=None)
    model.eval()

    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    head = model.lm_head.weight.detach()
    embed = model.model.word_embeddings.weight.detach()
    assert head.data_ptr() != embed.data_ptr(), (
        "lm_head shares storage with the embedding, the untied reading moved")
    with safe_open(os.path.join(MODEL_DIR, weight_map["lm_head.weight"]),
                   framework="pt") as f:
        raw_head = f.get_tensor("lm_head.weight")
    assert torch.equal(head, raw_head), (
        "loaded lm_head.weight disagrees bitwise with the checkpoint tensor")
    del raw_head
    for i in range(NUM_LAYERS):
        attn = model.model.layers[i].attention
        if (i + 1) % 4 == 0:
            continue
        assert attn.A_log.dtype == torch.float32, (
            f"layer {i} A_log drifted off the checkpoint f32 grid")
        assert attn.dt_bias.dtype == torch.float32, (
            f"layer {i} dt_bias drifted off the checkpoint f32 grid")
    for i in range(FIRST_K_DENSE_REPLACE, NUM_LAYERS):
        bias = model.model.layers[i].mlp.gate.expert_bias
        assert bias.dtype == torch.bfloat16, (
            f"layer {i} expert_bias drifted off the reference bf16 buffer "
            f"dtype, dtype {bias.dtype!r} breaks the recorded chain")
    return model


def install_layer_hooks(layers):
    """Wrap every decoder layer forward to record input and output tensors:
    - returns the capture list, one dict per layer, plus a restore closure
    - the Ling layer forward returns the tuple (hidden_states,) plus
      optional extras, the capture takes the first element
    - each install restores the pristine class forward after the run, so
      a second install never chains onto a previous wrapper"""
    captured = [None] * len(layers)
    originals = []

    def make_wrapper(layer_idx, layer, original):
        def wrapper(hidden_states, *args, **kwargs):
            entry = {"layer_input": hidden_states.clone()}
            output = original(hidden_states, *args, **kwargs)
            first = output[0] if isinstance(output, tuple) else output
            entry["layer_output"] = first.clone()
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
    - the Ling router module returns the tuple (topk_idx, topk_weight, logits)
    - the dense layer 0 carries no router module"""
    routed = [None] * len(layers)
    handles = []

    def hook(layer_idx):
        def hook_impl(module, args, output):
            topk_idx, topk_weight, logits = output
            routed[layer_idx] = {
                "router_input": args[0].detach().clone(),
                "router_logits": logits.clone(),
                "topk_weights": topk_weight.clone(),
                "topk_indices": topk_idx.clone(),
            }

        return hook_impl

    for i in range(FIRST_K_DENSE_REPLACE, len(layers)):
        handles.append(layers[i].mlp.gate.register_forward_hook(hook(i)))

    def restore():
        for handle in handles:
            handle.remove()

    return routed, restore


def run_forward(model, input_ids, capture_routers: bool):
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


def boundary_margins(router_logits: torch.Tensor, bias: torch.Tensor) -> dict:
    """Top-k boundary margin and the smallest adjacent gap inside the top-k, per
    token row on the group-limited biased selection score, sigmoid + bias:
    - top-2 biased scores per group summed, topk_group winning groups
    - the boundary sits between the sorted masked scores at top_k-1 and top_k
    - the op sequence mirrors the reference group_limited_topk exactly
    - one recorded positive margin locks the recorded expert-id set against selection-order ambiguity"""
    scores = router_logits.sigmoid()
    choice = scores + bias.unsqueeze(0)
    tokens, experts = choice.shape
    group_scores = choice.view(tokens, N_GROUP, -1).topk(2, dim=-1).values.sum(dim=-1)
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(tokens, N_GROUP, experts // N_GROUP)
        .reshape(tokens, -1))
    masked = choice.masked_fill(~score_mask.bool(), float("-inf"))
    sorted_choice = torch.sort(masked, dim=-1, descending=True).values
    boundary = sorted_choice[:, TOP_K - 1] - sorted_choice[:, TOP_K]
    inner_gap = (sorted_choice[:, : TOP_K - 1]
                 - sorted_choice[:, 1:TOP_K]).min(dim=-1).values
    sorted_groups = torch.sort(group_scores, dim=-1, descending=True).values
    group_ties = int((sorted_groups[:, TOPK_GROUP - 1]
                      == sorted_groups[:, TOPK_GROUP]).sum().item())
    return {
        "boundary_min": boundary.min().item(),
        "boundary_per_row": boundary.tolist(),
        "inner_gap_min": inner_gap.min().item(),
        "inner_gap_per_row": inner_gap.tolist(),
        "boundary_tie_tokens": int((boundary == 0).sum().item()),
        "group_tie_tokens": group_ties,
    }


def save_fixture(name: str, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file
plus the 004 stats sidecar."""
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
    _ling01.write_json_zst(metadata_path, metadata)
    write_stats_file(filepath + ".stats.json.zst", os.path.basename(filepath), [
        (name, tensor) for name, tensor in sorted_tensors.items()
        if tensor.is_floating_point()
    ])
    return filepath


def load_tokenizer():
    """Tokenize the prompt through the checkpoint tokenizer over the AutoTokenizer fast path, no special token added, the fixture ids
    carry the raw mergeable ranks only."""
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
    _ling01.check_ram()
    print(f"transformers {TRANSFORMERS_VERSION}")

    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    index_census(weight_map)
    model = build_model()
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    moe_indices = list(range(FIRST_K_DENSE_REPLACE, NUM_LAYERS))
    assert len(moe_indices) == NUM_MOE_LAYERS
    tokenizer = load_tokenizer()

    tokenizer_ids = tokenizer(INPUT_TEXT, add_special_tokens=False)["input_ids"]
    assert tokenizer_ids == PROMPT_IDS, (
        f"prompt ids {tokenizer_ids} disagree with the recorded corpus "
        f"{PROMPT_IDS}, the tokenizer path moved")
    input_ids = torch.tensor([tokenizer_ids], device=DEVICE)
    seq_len = input_ids.shape[1]
    layers = model.model.layers
    assert len(layers) == NUM_LAYERS, (
        f"the checkpoint must carry {NUM_LAYERS} decoder layers")

    # Recorded run:
    #   the from_pretrained defaults, twice, for determinism.
    captured, router_rows, logits = run_forward(model, input_ids, capture_routers=True)
    captured_again, _, logits_again = run_forward(model, input_ids, capture_routers=False)
    for i in range(NUM_LAYERS):
        assert torch.equal(captured[i]["layer_output"], captured_again[i]["layer_output"]), (
            f"recorded run layer {i} is not deterministic")
    assert torch.equal(logits, logits_again), "recorded run logits are not deterministic"
    assert logits.dtype == torch.float32, (
        "the reference casts the lm_head output to f32, the recorded "
        "decisions sample consumes the f32 grid")

    for i in range(NUM_LAYERS):
        layer_input = captured[i]["layer_input"]
        layer_output = captured[i]["layer_output"]
        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.layers.{i}",
            "layer_type": "dense" if i < FIRST_K_DENSE_REPLACE else "moe",
            "attention_type": "attention" if (i + 1) % 4 == 0 else "linear_attention",
            "kda_dispatch": "none" if (i + 1) % 4 == 0 else "fused_recurrent",
            "input_text": INPUT_TEXT,
            "input_tokens": tokenizer_ids,
            "batch_size": 1,
            "seq_len": seq_len,
            "num_threads": _ling01.NUM_THREADS,
            "dtype": "bfloat16",
            "device": DEVICE,
            "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
            "torch_version": torch.__version__,
            "transformers_version": TRANSFORMERS_VERSION,
            "note": "layer_input_seq is the recorded reference chain. The "
                    "reference MoE block runs the eager token-gather "
                    "dispatch (moe_infer) with no grouped_mm alternative, "
                    "the port reduces the same eager class per routed "
                    "block, no reference-internal dispatch class exists "
                    "on this family.",
        }
        tensors = {"layer_input_seq": layer_input}
        if i >= FIRST_K_DENSE_REPLACE:
            gate_row = router_rows[i]
            bias = layers[i].mlp.gate.expert_bias.detach().clone().float()
            margins = boundary_margins(gate_row["router_logits"], bias)
            assert margins["boundary_min"] > 0.0, (
                f"layer {i} top-k boundary margin {margins['boundary_min']} is not "
                "positive, the expert ids are order-ambiguous, pick a different prompt")
            assert margins["boundary_tie_tokens"] == 0, (
                f"layer {i} carries {margins['boundary_tie_tokens']} boundary ties, "
                "the recorded ids are order-ambiguous, pick a different prompt")
            assert margins["group_tie_tokens"] == 0, (
                f"layer {i} carries {margins['group_tie_tokens']} group-boundary "
                "ties, the recorded group mask is order-ambiguous, pick a "
                "different prompt")
            metadata["margins"] = margins
            metadata["margin_floor"] = 1e-4
            metadata["flip_budget_policy"] = (
                "exact expert-id comparisons in the consuming suites stand on a "
                "recorded positive boundary margin per row with the 1e-4 floor, "
                "a row below the floor carries "
                "an explicit per-fixture recorded exception, never a silent "
                "widening")
            tensors["router_input"] = gate_row["router_input"]
            tensors["topk_indices"] = gate_row["topk_indices"]
            tensors["routing_weights"] = gate_row["topk_weights"]
        if i == NUM_LAYERS - 1:
            tensors["layer_output_seq"] = layer_output
        save_fixture(f"layer-{i:02d}", metadata, tensors)
        margin_note = (f", topk_margin {margins['boundary_min']:.3e}"
                       if "margins" in metadata else "")
        print(f"  layer {i:02d} ({metadata['layer_type']}, "
              f"{metadata['attention_type']}){margin_note}")

    # Decision projection of the recorded run, the raw logits tensor
    # leaves the tree, the consumers carry the argmax id, the top-32
    # support and the tail probability of each recorded position.
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
    write_json_zst(os.path.join(FIXTURE_DIR,
                                "final_logits.safetensor.metadata.json.zst"), {
        "model": MODEL_NAME,
        "input_text": INPUT_TEXT,
        "input_tokens": tokenizer_ids,
        "seq_len": seq_len,
        "num_threads": _ling01.NUM_THREADS,
        "dtype": "bfloat16",
        "device": DEVICE,
        "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
        "note": "the projection is the recorded reference run, the "
                "reference casts the lm_head output to f32 before the "
                "sample reads it.",
    })
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print(f"Generated {NUM_LAYERS} layer fixtures + final_logits under {FIXTURE_DIR}")
    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
