#!/usr/bin/env python3
"""GLM-4.7-Flash fixture generator over the real checkpoint shards, the gitignored
tests/hf_models symlink resolves the checkpoint.

Routed-block (MoE) layer-1 records (moe-*) through the installed reference modeling,
Glm4MoeLiteForCausalLM with grouped_mm expert dispatch, on torch bf16:
- fixture dir tests/fixtures/bf16-01-layer-internals/GLM-4.7-Flash-layer-1/
- consumer tests/q_bf16/t_bf16_glm47flash_01_layer_internals_moe.nim

- moe-GLM-4.7-Flash-<case>.safetensor, the recorded boundary slices of the case
- .metadata.json.zst, the case identity, the recorded geometry and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors

Recorded content:
- moe-00 prefill and moe-01 one decode step on a live forward capture
- the hidden states the MoE block consumes, the router decision rows and the routed-block output
- the boundary margins behind the exact-index asserts, floor 1e-4, recorded in the metadata
- no Qwen3 analog exists, the Qwen3 and Qwen3.5 families are not routed, no dense router decision row exists

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root
  uv run python workspace/transformers/tests/testgen/gen_bf16_glm47flash_01_layer_internals_moe.py
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
from safetensors import torch as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import


import transformers  # noqa: E402
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (  # noqa: E402
    Glm4MoeLiteForCausalLM,
)

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The checkpoint resolves through the gitignored tests/hf_models symlink.
MODEL_NAME = "GLM-4.7-Flash"
MOE_LAYER_IDX = 1
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals",
    f"{MODEL_NAME}-layer-{MOE_LAYER_IDX}"
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests", "hf_models", MODEL_NAME)

NUM_THREADS = 1

DEVICE = "cpu"
# Recording device, the --device argument overrides the default, both the metadata and the env device rows record the value
# the run used.

# Routed-block geometry of the checkpoint, recorded literals:
# 64 experts, top-4 routing, 2048-wide (no config parse in this script).
NUM_EXPERTS = 64
TOP_K = 4
HIDDEN = 2048

# Per-generator seed, independent and order-agnostic. Both the input ids
# and the decode-step id come from this one generator in a fixed order.
SEED_MOE = 236

PREFILL_LEN = 8
DECODE_POSITION = 8
MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_glm47flash_01_layer_internals_moe] vm_stat gave no page size line")


def pool_bytes() -> int:
    """Free+inactive+speculative physical memory in bytes from `vm_stat`, the pool formula the 58 GB checkpoint load runs against."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    wanted = ("Pages free:", "Pages inactive:", "Pages speculative:")
    pool = 0
    for line in out.stdout.splitlines():
        for prefix in wanted:
            if line.startswith(prefix):
                pool += int(line.split()[2].rstrip(".")) * page
    if pool == 0:
        raise SystemExit(
            "[gen_bf16_glm47flash_01_layer_internals_moe] vm_stat gave no pool lines")
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
    """Refuse to load weights under low memory or a stray python/torch process, the 58 GB load runs against the 32 GiB pool floor."""
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_glm47flash_01_layer_internals_moe] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_glm47flash_01_layer_internals_moe] other python/torch processes hold RAM: {stray}. "
            "Stop and retry when idle")


def load_model() -> Glm4MoeLiteForCausalLM:
    """Full reference model at the checkpoint dtype on CPU, the capture needs the hidden states a live forward feeds layer 1."""
    model = Glm4MoeLiteForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map=None)
    model.eval()
    return model


def save_fixture(layer_name: str, case_num: int, metadata: dict,
                 tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file."""
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in sorted(tensors.items())
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


COMMON_META = {
    "model": MODEL_NAME,
    "layer": f"model.layers.{MOE_LAYER_IDX}.mlp",
    "moe_layer_index": MOE_LAYER_IDX,
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": TOP_K,
    "hidden_size": HIDDEN,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.8,
    "norm_topk_prob": True,
    "shared_experts": 1,
    "num_threads": NUM_THREADS,
    "dtype": "bfloat16",
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
}

POLICY_META = {
    "flip_budget_policy": "exact expert-id comparisons in the consuming "
        "suites stand on a recorded positive boundary margin with the "
        "1e-4 floor, a row below the floor would carry an explicit "
        "per-fixture recorded exception, never a silent widening",
    "margin_floor": 1e-4,
}


def boundary_margins(router_logits: torch.Tensor, bias: torch.Tensor,
                     top_k: int) -> dict:
    """Top-k boundary margin and the smallest adjacent gap inside the top-k, per token row on the selection score, sigmoid + bias."""
    scores = router_logits.sigmoid()
    scores_for_choice = scores + bias.unsqueeze(0)
    sorted_choice = torch.sort(
        scores_for_choice, dim=-1, descending=True).values
    boundary = (sorted_choice[:, top_k - 1] - sorted_choice[:, top_k])
    inner_gap = (sorted_choice[:, : top_k - 1] -
                 sorted_choice[:, 1:top_k]).min(dim=-1).values
    return {
        "boundary_min": boundary.min().item(),
        "boundary_per_row": boundary.tolist(),
        "inner_gap_min": inner_gap.min().item(),
        "inner_gap_per_row": inner_gap.tolist(),
    }


def main() -> None:
    """Records the layer-1 routed-block fixtures of the checkpoint."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    COMMON_META["recorded_from"] = os.environ.get("TTT_RECORD_FROM", "m4max-cpu")
    COMMON_META["device"] = DEVICE
    check_ram()

    model = load_model()
    if DEVICE != "cpu":
        model = model.to(DEVICE)

    # Capture on the live layer-1 routed block, the block input, the router
    # decision rows and the block output:
    # - the block input is the recorded post_attention_layernorm output
    # - the router module returns the tuple router_logits, topk_weights,
    #   topk_indices with the logits at f32
    captured = {}

    def gate_hook(module, args, output):
        captured["router_logits"], captured["topk_weights"], \
            captured["topk_indices"] = output

    def block_hook(module, args, output):
        captured["moe_input"] = args[0].detach().clone()
        captured["moe_output"] = (output[0] if isinstance(output, tuple)
                                  else output).detach().clone()

    mlp = model.model.layers[MOE_LAYER_IDX].mlp
    h1 = mlp.gate.register_forward_hook(gate_hook)
    h2 = mlp.register_forward_hook(block_hook)

    bias = mlp.gate.e_score_correction_bias.detach().clone().float()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED_MOE)
    vocab = model.config.vocab_size
    input_ids = torch.randint(
        0, vocab, (1, PREFILL_LEN), generator=gen, dtype=torch.long)
    step_id = torch.randint(
        0, vocab, (1, 1), generator=gen, dtype=torch.long)
    input_ids = input_ids.to(DEVICE)
    step_id = step_id.to(DEVICE)

    with torch.no_grad():
        out_prefill = model(input_ids, use_cache=True)
        captured_prefill = dict(captured)
        captured.clear()
        past = out_prefill.past_key_values
        out_step = model(step_id, use_cache=True,
                         past_key_values=past,
                         position_ids=torch.tensor(
                             [[DECODE_POSITION]], dtype=torch.long, device=DEVICE))
        captured_step = dict(captured)

    h1.remove()
    h2.remove()

    # Generator self-check, the recorded margins are positive and the weights carry the gathered raw scores after renorm
    # plus scale, matching the reference contract.
    m_pre = boundary_margins(
        captured_prefill["router_logits"], bias, TOP_K)
    m_step = boundary_margins(
        captured_step["router_logits"], bias, TOP_K)
    assert m_pre["boundary_min"] > 0.0 and m_pre["inner_gap_min"] > 0.0, (
        "[gen_bf16_glm47flash_01_layer_internals_moe] prefill margin ambiguity, "
        "pick a different seed")
    assert m_step["boundary_min"] > 0.0 and m_step["inner_gap_min"] > 0.0, (
        "[gen_bf16_glm47flash_01_layer_internals_moe] decode margin ambiguity, "
        "pick a different seed")

    meta_boundary = {
        "topk_boundary_margin_min": min(m_pre["boundary_min"],
                                        m_step["boundary_min"]),
        "topk_boundary_margin_prefill": m_pre["boundary_min"],
        "topk_boundary_margin_decode": m_step["boundary_min"],
        "topk_inner_gap_min": min(m_pre["inner_gap_min"],
                                  m_step["inner_gap_min"]),
    }

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    save_fixture(
        "moe", 0,
        {
            **COMMON_META, **POLICY_META, **meta_boundary,
            "case": "real_hidden_prefill_seq8",
            "seed": SEED_MOE,
            "positions": list(range(PREFILL_LEN)),
        },
        {
            "input_ids": input_ids,
            "positions": torch.arange(PREFILL_LEN, dtype=torch.long, device=DEVICE),
            "hidden_states": captured_prefill["moe_input"],
            "router_logits": captured_prefill["router_logits"],
            "topk_weights": captured_prefill["topk_weights"],
            "topk_indices": captured_prefill["topk_indices"],
            "moe_output": captured_prefill["moe_output"],
        },
    )
    save_fixture(
        "moe", 1,
        {
            **COMMON_META, **POLICY_META, **meta_boundary,
            "case": "real_hidden_decode_pos8",
            "seed": SEED_MOE,
            "position": DECODE_POSITION,
            "positions": [DECODE_POSITION],
        },
        {
            "input_ids": step_id,
            "positions": torch.tensor([DECODE_POSITION], dtype=torch.long, device=DEVICE),
            "hidden_states": captured_step["moe_input"],
            "router_logits": captured_step["router_logits"],
            "topk_weights": captured_step["topk_weights"],
            "topk_indices": captured_step["topk_indices"],
            "moe_output": captured_step["moe_output"],
        },
    )

    print(f"[gen_bf16_glm47flash_01_layer_internals_moe] wrote {FIXTURE_DIR}")
    print(f"[gen_bf16_glm47flash_01_layer_internals_moe] boundary margins "
          f"prefill {m_pre['boundary_min']:.3e}, decode "
          f"{m_step['boundary_min']:.3e}, inner gap min "
          f"{meta_boundary['topk_inner_gap_min']:.3e}")


if __name__ == "__main__":
    main()
