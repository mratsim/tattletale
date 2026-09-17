#!/usr/bin/env python3
"""Moonlight-16B-A3B fixture generator over the real checkpoint shards, resolved
through the gitignored tests/hf_models symlink.

MoE router layer-1 record through the noaux_tc reference router, DeepseekV3TopkRouter, on torch bf16:
- fixture dir tests/fixtures/bf16-01-layer-internals/Moonlight-16B-A3B-layer-1/
- consumer tests/q_bf16/t_bf16_moonlight_01_layer_internals_router.nim

- `gate-Moonlight-16B-A3B-00.safetensor`, the recorded boundary slices of the case
- .metadata.json.zst, the case identity, the recorded geometry and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors

Recorded content:
- hidden_states [6, H] and hidden_step [1, H] bf16, the router weight and the bias buffer
- router logits (f32), topk weights (renormalized, scaled, f32) and expert indices
- the topk/group boundary margins in the metadata, floor 1e-4, a seed under the floor skips to the next seed
- no Qwen3 analog exists, no Qwen3 family fixture records the router as an isolated op with its own margin metadata

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root
  uv run python workspace/transformers/tests/testgen/gen_bf16_moonlight_01_layer_internals_router.py
"""
import argparse
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import


import transformers  # noqa: E402
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (  # noqa: E402
    DeepseekV3Config,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: E402
    DeepseekV3TopkRouter,
)
# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Every checkpoint resolves through the gitignored tests/hf_models symlink.
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_ROOT = os.path.join(GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals")
MODEL_NAME = "Moonlight-16B-A3B"
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests", "hf_models", MODEL_NAME)

NUM_THREADS = 1

DEVICE = "cpu"
    # Recording device, the --device argument overrides the default, the metadata and env device rows record the value the run used.
NUM_PREFILL_TOKENS = 6
MARGIN_FLOOR = 1e-4
MAX_SEED_TRIES = 200

MIN_FREE_BYTES = 8 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_moonlight_01_layer_internals_router] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_bf16_moonlight_01_layer_internals_router] vm_stat gave no 'Pages free' line")


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
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"[gen_bf16_moonlight_01_layer_internals_router] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_moonlight_01_layer_internals_router] other python/torch processes hold RAM: {stray}. "
            "Stop and retry when idle")


def shard_of(model_dir: str, key: str) -> str:
    """Shard filename holding one checkpoint tensor, read off the model.safetensors.index.json weight map."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise SystemExit(f"[gen_bf16_moonlight_01_layer_internals_router] key {key} missing from the weight map")
    return weight_map[key]


def load_gate(model_dir: str, gate_key: str, bias_key: str | None) -> dict:
    """Router weight [E, H] in the checkpoint dtype plus the f32 bias buffer when the checkpoint carries one, memory-mapped loads of two tensors."""
    weights = {}
    shards = {gate_key: shard_of(model_dir, gate_key)}
    if bias_key is not None:
        shards[bias_key] = shard_of(model_dir, bias_key)
    for key, shard in shards.items():
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            weights[key] = f.get_tensor(key).clone()
    return weights


def save_fixture(filepath: str, metadata: dict, tensors: dict) -> None:
    """Save one fixture safetensor plus its zstd metadata sidecar and the 004 stats sidecar."""
    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in sorted(tensors.items())
        if tensor is not None
    )
    serialized = st.save(sorted_tensors, metadata=None)
    with open(filepath, "wb") as f:
        f.write(serialized)
    write_json_zst(filepath + ".metadata.json.zst", metadata)
    write_stats_file(filepath + ".stats.json.zst", os.path.basename(filepath), [
        (name, tensor) for name, tensor in sorted_tensors.items()
        if tensor.is_floating_point()
    ])


def topk_boundary_margin(choice: torch.Tensor, top_k: int) -> float:
    """Smallest gap between the Kth and the K+1th sorted choice score, minimized over all rows, the exact-index condition of the top-k selection."""
    sorted_choice = choice.sort(dim=-1, descending=True).values
    gaps = sorted_choice[:, top_k - 1] - sorted_choice[:, top_k]
    return gaps.min().item()


def group_boundary_margin(choice: torch.Tensor, num_group: int, topk_group: int) -> float:
    """Smallest gap around the topk_group boundary of the group scores:
    - the exact-index condition of the group limiting stage
    - undefined at topk_group == num_group"""
    experts_per_group = choice.shape[1] // num_group
    group_scores = (
        choice.view(-1, num_group, experts_per_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    sorted_groups = group_scores.sort(dim=-1, descending=True).values
    gaps = sorted_groups[:, topk_group - 1] - sorted_groups[:, topk_group]
    return gaps.min().item()


def variant_definition() -> list[dict]:
    """One fixture variant, checkpoint path, reference config values and the per-variant seed, read off the checkpoint config.json
    and weight map."""
    return [
        {
            "model": 'Moonlight-16B-A3B',
            "router": 'noaux_tc',
            "module": 'deepseek_v3',
            "gate_key": 'model.layers.1.mlp.gate.weight',
            "bias_key": 'model.layers.1.mlp.gate.e_score_correction_bias',
            "num_experts": 64,
            "num_experts_per_tok": 6,
            "routed_scaling_factor": 2.446,
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            "hidden_size": 2048,
            "seed": 415,
            "gate_split": 2,
        },
    ]


def build_router(variant: dict, weights: dict):
    """Reference router module, the real checkpoint router weight and bias
    buffer assigned, eval mode:
    - DeepseekV3TopkRouter stays the noaux_tc authority
    - op-identical to the checkpoint's own router module
    - selection-only bias, the f32 renorm keeps the 1e-20 denominator"""
    cfg = DeepseekV3Config(
        num_local_experts=variant["num_experts"],
        num_experts_per_tok=variant["num_experts_per_tok"],
        hidden_size=variant["hidden_size"],
        routed_scaling_factor=variant["routed_scaling_factor"],
        n_group=variant["n_group"],
        topk_group=variant["topk_group"],
        norm_topk_prob=variant["norm_topk_prob"],
    )
    router = DeepseekV3TopkRouter(cfg)
    router.weight.data = weights[variant["gate_key"]]
    router.e_score_correction_bias.data = (
        weights[variant["bias_key"]].to(torch.float32)
    )
    router.eval()
    return router


def gate_layer_prefix(variant: dict) -> str:
    """Checkpoint prefix of the router weight parameters, metadata row."""
    return variant["gate_key"].rsplit(".weight", 1)[0]


def generate_variant(variant: dict) -> None:
    """One variant, margin-clean hidden inputs and reference routing outputs, one safetensor plus sidecars:"""
    model_dir = MODEL_DIR
    weights = load_gate(model_dir, variant["gate_key"], variant["bias_key"])
    router = build_router(variant, weights)
    router = router.to(DEVICE)

    hidden_size = variant["hidden_size"]
    top_k = variant["num_experts_per_tok"]
    grouped = variant["n_group"] > 1

    seed = variant["seed"]
    margin_topk = -1.0
    margin_group = -1.0
    for _ in range(MAX_SEED_TRIES):
        torch.manual_seed(seed)
        hidden = torch.randn(NUM_PREFILL_TOKENS, hidden_size, dtype=torch.bfloat16, device=DEVICE)
        hidden_step = torch.randn(1, hidden_size, dtype=torch.bfloat16, device=DEVICE)
        with torch.no_grad():
            logits, topk_weights, topk_indices = router(hidden)
            logits_step, topk_weights_step, topk_indices_step = router(hidden_step)
        if variant["router"] == "noaux_tc":
            choice = torch.sigmoid(logits) + router.e_score_correction_bias
            choice_step = torch.sigmoid(logits_step) + router.e_score_correction_bias
        else:
            choice = torch.softmax(logits, dim=-1, dtype=torch.float32)
            choice_step = torch.softmax(logits_step, dim=-1, dtype=torch.float32)
        margin_topk = min(
            topk_boundary_margin(choice, top_k),
            topk_boundary_margin(choice_step, top_k),
        )
        margin_group = 1.0
        if grouped:
            margin_group = min(
                group_boundary_margin(choice, variant["n_group"], variant["topk_group"]),
                group_boundary_margin(choice_step, variant["n_group"], variant["topk_group"]),
            )
        if margin_topk > MARGIN_FLOOR and margin_group > MARGIN_FLOOR:
            break
        seed += 1
    else:
        raise SystemExit(
            f"[gen_bf16_moonlight_01_layer_internals_router] no margin-clean seed found for {variant['model']}"
        )
    print(f"  {variant['model']}: seed {seed}, topk margin {margin_topk:.3e}"
          + (f", group margin {margin_group:.3e}" if grouped else ""))

    # Self-checks before saving, the renormalized rows sum to the routed
    # scaling factor and the weights stay the unbiased gathered scores
    # (selection-only bias), each within the f32 renorm rounding slack.
    gathered = torch.sigmoid(logits).gather(1, topk_indices) \
        if variant["router"] == "noaux_tc" \
        else torch.softmax(logits, dim=-1, dtype=torch.float32).gather(1, topk_indices)
    if variant["norm_topk_prob"]:
        expected = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20) \
            * variant["routed_scaling_factor"]
        row_sums = topk_weights.sum(dim=-1)
        drift = (row_sums - variant["routed_scaling_factor"]).abs().max().item()
        assert drift < 1e-6, f"renorm sum drift {drift}"
    else:
        expected = gathered * variant["routed_scaling_factor"]
    assert torch.allclose(topk_weights, expected, rtol=0.0, atol=1e-6), \
        "weights diverge from the unbiased gathered scores"

    layer_prefix = gate_layer_prefix(variant)
    gate_weight = weights[variant["gate_key"]]
    bias = None
    if variant["bias_key"] is not None:
        bias = weights[variant["bias_key"]].to(torch.float32)

    metadata = {
        "model": variant["model"],
        "layer": layer_prefix,
        "router": variant["router"],
        "num_experts": variant["num_experts"],
        "num_experts_per_tok": top_k,
        "routed_scaling_factor": variant["routed_scaling_factor"],
        "norm_topk_prob": variant["norm_topk_prob"],
        "n_group": variant["n_group"],
        "topk_group": variant["topk_group"],
        "hidden_size": hidden_size,
        "gate_weight_key": variant["gate_key"],
        "gate_weight_dtype": str(gate_weight.dtype).replace("torch.", ""),
        "hidden_dtype": "bfloat16",
        "logits_dtype": "float32",
        "weights_dtype": "float32",
        "indices_dtype": "int64",
        "num_prefill_tokens": NUM_PREFILL_TOKENS,
        "num_threads": NUM_THREADS,
        "seed": seed,
        "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
        "device": DEVICE,
        "margin_floor": MARGIN_FLOOR,
        "topk_boundary_margin": margin_topk,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    if variant.get("gate_split"):
        metadata["gate_split"] = variant["gate_split"]
    if variant["bias_key"] is not None:
        metadata["bias_key"] = variant["bias_key"]
        metadata["bias_dtype"] = str(bias.dtype).replace("torch.", "")
        metadata["bias_checkpoint_dtype"] = str(
            weights[variant["bias_key"]].dtype
        ).replace("torch.", "")
    if grouped:
        metadata["group_boundary_margin"] = margin_group

    tensors = {
        "hidden_states": hidden,
        "hidden_step": hidden_step,
        "gate_weight": gate_weight,
        "router_logits": logits,
        "router_logits_step": logits_step,
        "topk_weights": topk_weights,
        "topk_weights_step": topk_weights_step,
        "topk_indices": topk_indices,
        "topk_indices_step": topk_indices_step,
    }
    if bias is not None:
        tensors["bias"] = bias

    fixture_dir = os.path.join(
        FIXTURE_ROOT, f"{variant['model']}-layer-1")
    os.makedirs(fixture_dir, exist_ok=True)
    filepath = os.path.join(fixture_dir, f"gate-{variant['model']}-00.safetensor")
    split = variant.get("gate_split")
    if split:
        decision = {name: t for name, t in tensors.items()
                    if name != "gate_weight"}
        save_fixture(filepath, metadata, decision)
        gate = tensors["gate_weight"]
        rows = gate.shape[0]
        assert rows % split == 0, "gate split must divide the expert rows"
        step = rows // split
        for half, start in (("a", 0), ("b", step)):
            part = os.path.join(
                fixture_dir,
                f"gate-{variant['model']}-00-gate-{half}.safetensor")
            save_fixture(part, {
                "model": variant["model"],
                "content": "gate_weight rows "
                    + f"{start}..{start + step - 1} of {rows}",
                "gate_weight_dtype": str(gate.dtype).replace("torch.", ""),
                "rows": f"{start}:{start + step}",
            }, {"gate_weight": gate[start:start + step].contiguous()})
    else:
        save_fixture(filepath, metadata, tensors)

    print(f"  wrote {filepath}")


def main() -> None:
    """Records the layer-1 router fixture of the checkpoint."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    check_ram()
    for variant in variant_definition():
        generate_variant(variant)
    print(
        f"[gen_bf16_moonlight_01_layer_internals_router] torch {torch.__version__},"
        f" transformers {transformers.__version__}"
    )
    print(f"[gen_bf16_moonlight_01_layer_internals_router] wrote {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
