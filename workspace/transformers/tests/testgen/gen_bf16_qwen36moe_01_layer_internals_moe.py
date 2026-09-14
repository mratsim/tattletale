#!/usr/bin/env python3
"""Routed-block (MoE) layer-0 fixture for the Qwen3.6-35B-A3B checkpoint,
recorded on CPU torch bf16 with the installed reference modeling, from safetensors.

Consumed by tests/q_bf16/t_bf16_qwen36moe_01_layer_internals_moe.nim:

  - no Qwen3-era fixture records the router logits, top-k selection or the fused rank-3 expert tensors
  - the Qwen3 and Qwen3.5 dense families are not routed
  - moe_layer0_fixture.json, under tests/fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-0/
  - it holds one routed-block forward on T=6 deterministic bf16 tokens with real layer-0 weights

Records h, router logits (f32), top-k indices, renormalized fp32 values, routing weights after the dtype cast:

  - the shared-expert gate (post-sigmoid) and the MoE output
  - bands derived from bf16 ulp arithmetic
  - the sorted-value margins that justify exact-index asserts

Run twice, cmp proves byte determinism:

  cd <worktree root> && .venv/bin/python workspace/transformers/tests/testgen/gen_bf16_qwen36moe_01_layer_internals_moe.py

RAM guards:

  - the invoking shell runs `vm_stat` and `pgrep -f "python.*(torch|hf)"` before this script
  - the script re-runs both checks and refuses the weight load when free memory is low or another python/torch process runs
  - its own process chain stays excluded from the pgrep match
"""

import json
import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}


def write_json_zst(path, obj, ensure_ascii=True):
    """Write obj as one zstd frame, level 19, content size and checksum
    recorded in the frame header.

    Args:
    - path, obj, the destination file and the JSON-serializable payload
    - ensure_ascii, the json.dumps escaping switch
    """
    payload = json.dumps(
        obj, sort_keys=True, indent=2, ensure_ascii=ensure_ascii
    ).encode("utf-8") + b"\n"
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))

import os
import subprocess
import sys


import torch  # noqa: E402

from fixture_stats import assert_path_equivalent  # noqa: E402

import transformers  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeSparseMoeBlock,
)

# Determinism, single intra-op thread, deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Checkpoint, fixture and config paths.
MODEL_NAME = "Qwen3.6-35B-A3B"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals", "Qwen3.6-35B-A3B-layer-0"
)
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "moe_layer0_fixture.json.zst")
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
WEIGHTS_FILE_1 = os.path.join(MODEL_DIR, "model-00001-of-00026.safetensors")
WEIGHTS_FILE_2 = os.path.join(MODEL_DIR, "model-00002-of-00026.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


# First seed tried by the margin search below, the chosen seed is recorded
# in the fixture meta, so regeneration is byte-deterministic.
SEED_BASE = 71
NUM_THREADS = 1

# Routed-block sequence length, the geometry lives in the parsed config.
T = 6

PREFIX = "model.language_model.layers.0.mlp."
MIN_FREE_BYTES = 8 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals_moe] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals_moe] vm_stat gave no 'Pages free' line")


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
    """Refuse to load weights when memory is low or another python/torch
    process holds RAM.

    Guard:

    - free memory below the floor raises
    - the pgrep match excludes this process chain, its own command line
      spells the torch dependency of the run
    """
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_01_layer_internals_moe] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_01_layer_internals_moe] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def load_text_config() -> Qwen3_5MoeTextConfig:
    """Load the nested text_config from the checkpoint config.json.

    - the installed PretrainedConfig defaults `_experts_implementation` to None
    - the module built on this config runs the reference expert loop
    """
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    return Qwen3_5MoeTextConfig.from_dict(wrapper["text_config"])


def load_layer0_moe_weights() -> dict:
    """Load the layer-0 MoE tensors from the two safetensor files that hold
    them via safe_open (memory-mapped, only these tensors are copied).

    Returns:
    - the weight dict keyed by the short tensor names used below
    """
    weights = {}
    with safe_open(WEIGHTS_FILE_1, framework="pt") as f:
        weights["experts.gate_up_proj"] = f.get_tensor(
            PREFIX + "experts.gate_up_proj").clone()
    with safe_open(WEIGHTS_FILE_2, framework="pt") as f:
        for short in (
            "experts.down_proj",
            "gate.weight",
            "shared_expert.down_proj.weight",
            "shared_expert.gate_proj.weight",
            "shared_expert.up_proj.weight",
            "shared_expert_gate.weight",
        ):
            weights[short] = f.get_tensor(PREFIX + short).clone()
    return weights


def build_block(cfg, weights) -> Qwen3_5MoeSparseMoeBlock:
    """Qwen3_5MoeSparseMoeBlock with real layer-0 weights loaded."""
    block = Qwen3_5MoeSparseMoeBlock(cfg)
    with torch.no_grad():
        block.gate.weight.data = weights["gate.weight"]
        block.experts.gate_up_proj.data = weights["experts.gate_up_proj"]
        block.experts.down_proj.data = weights["experts.down_proj"]
        block.shared_expert.gate_proj.weight.data = weights["shared_expert.gate_proj.weight"]
        block.shared_expert.up_proj.weight.data = weights["shared_expert.up_proj.weight"]
        block.shared_expert.down_proj.weight.data = weights["shared_expert.down_proj.weight"]
        block.shared_expert_gate.weight.data = weights["shared_expert_gate.weight"]
    block.eval()
    return block


def ulp_bf16(m: float) -> float:
    """One bf16 ulp at magnitude m, bf16 has 8 significand bits, so for m
    in [2**e, 2**(e+1)) the ulp is 2**(e-7). Zero maps to band 0."""
    if m <= 0:
        return 0.0
    e = int(torch.floor(torch.log2(torch.tensor(m))).item())
    return 2.0 ** (e - 7)


def pick_seed(block, cfg):
    """First seed from SEED_BASE whose sorted-probability margins are positive.

    A tie leaves the top-k order ambiguous between a sort and a topk, which
    makes the fixture unusable for exact-index asserts.

    Args:
    - block, cfg, the weighted MoE block and the parsed text config

    Returns:
    - the seed, its h, its full-probability tensor and the two margins
    """
    top_k = cfg.num_experts_per_tok
    for offset in range(64):
        seed = SEED_BASE + offset
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        h = torch.randn(1, T, cfg.hidden_size, generator=gen, dtype=torch.bfloat16)
        with torch.no_grad():
            router_logits, _, _ = block.gate(h)
            probs = torch.nn.functional.softmax(
                router_logits, dtype=torch.float32, dim=-1)
        sorted_probs = torch.sort(probs, dim=-1, descending=True).values
        top9_margin = (sorted_probs[:, top_k - 1] - sorted_probs[:, top_k]).min().item()
        inner_gap = (sorted_probs[:, :top_k - 1] - sorted_probs[:, 1:top_k]).min().item()
        if top9_margin > 0 and inner_gap > 0:
            return seed, h, probs, top9_margin, inner_gap
    raise SystemExit(
        "[gen_bf16_qwen36moe_01_layer_internals_moe] no seed with unambiguous top-k margins in 64 tries")


def main() -> None:
    """Write the layer-0 routed-block fixture after the RAM guard."""
    check_ram()

    cfg = load_text_config()
    weights = load_layer0_moe_weights()
    block = build_block(cfg, weights)

    seed, h, probs, top9_margin, inner_gap = pick_seed(block, cfg)

    with torch.no_grad():
        # The router chain on its own, logits at the hidden-state dtype,
        # softmax over the f32 router logits of all experts, top-k, fp32 renorm,
        # cast back last. The fp32 renormed values are kept pre-cast.
        router_logits, router_scores, router_indices = block.gate(h)
        top_values_fp32, top_indices = torch.topk(probs, cfg.num_experts_per_tok, dim=-1)
        renorm_fp32 = top_values_fp32 / top_values_fp32.sum(dim=-1, keepdim=True)
        routing_weights = renorm_fp32.to(router_logits.dtype)

        # Generator self-check, the manual chain reproduces the module
        # router scores within the instrument, plain integer equality
        # for the indices (discrete ids, no rounding).
        assert_path_equivalent(routing_weights, router_scores,
            "[gen_bf16_qwen36moe_01_layer_internals_moe] manual router chain vs the module router scores")
        assert bool((top_indices == router_indices).all()), \
            "[gen_bf16_qwen36moe_01_layer_internals_moe] manual top-k indices != module top-k indices"

        shared_gate = torch.nn.functional.sigmoid(
            block.shared_expert_gate(h))
        moe_output = block(h)

    h_fp32 = h[0].to(torch.float32)
    logits_fp32 = router_logits.to(torch.float32)
    renorm_list = renorm_fp32.to(torch.float32)
    weights_list = routing_weights.to(torch.float32)
    gate_list = shared_gate[0, :, 0].to(torch.float32)
    output_fp32 = moe_output[0].to(torch.float32)

    # Bands from bf16 ulp arithmetic (first principles, not observed deltas):
    #
    # - one ulp at the max magnitude for the pointwise values
    # - three ulps for the MoE output, a GEMM boundary flip propagates
    #   through the multiply, the accumulate and the shared add
    logits_max = logits_fp32.abs().max().item()
    weights_max = weights_list.abs().max().item()
    gate_max = gate_list.abs().max().item()
    output_max = output_fp32.abs().max().item()
    fixture = {
        "meta": {
            "seed": seed,
            "num_threads": NUM_THREADS,
            "dtype": "bfloat16",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "num_experts": int(cfg.num_experts),
            "num_experts_per_tok": int(cfg.num_experts_per_tok),
            "hidden_size": int(cfg.hidden_size),
            "moe_intermediate_size": int(cfg.moe_intermediate_size),
            "shared_expert_intermediate_size": int(cfg.shared_expert_intermediate_size),
        },
        "h": h_fp32.tolist(),
        "router_logits": logits_fp32.tolist(),
        "topk_indices": top_indices.tolist(),
        "renorm_values": renorm_list.tolist(),
        "routing_weights": weights_list.tolist(),
        "shared_gate": gate_list.tolist(),
        "moe_output": output_fp32.tolist(),
        "bands": {
            "router_logits_band": ulp_bf16(logits_max),
            "routing_weights_band": ulp_bf16(weights_max),
            "shared_gate_band": ulp_bf16(gate_max),
            "output_band": 3.0 * ulp_bf16(output_max),
        },
        "margins": {
            "topk_margin_min": top9_margin,
            "topk_inner_gap_min": inner_gap,
        },
    }

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    write_json_zst(FIXTURE_PATH, fixture)
    print(f"[gen_bf16_qwen36moe_01_layer_internals_moe] wrote {FIXTURE_PATH}")
    print(f"[gen_bf16_qwen36moe_01_layer_internals_moe] torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"[gen_bf16_qwen36moe_01_layer_internals_moe] top9 margin {top9_margin:.3e}, "
          f"inner gap {inner_gap:.3e}")
    print(f"[gen_bf16_qwen36moe_01_layer_internals_moe] bands: {fixture['bands']}")


if __name__ == "__main__":
    main()
