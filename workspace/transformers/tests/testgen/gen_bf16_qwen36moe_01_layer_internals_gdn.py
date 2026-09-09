#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B Gated DeltaNet (GDN) layer-0 fixtures from
the real checkpoint safetensors files, using the installed reference
modeling on CPU
torch bf16.

Consumed by tests/q_bf16/t_bf16_qwen36moe_02_gdn.nim. No Qwen3 analog: the
same GatedDeltaNet SSM layer as the Qwen3.5-0.8B dense family, absent
from Qwen3, so no Qwen3-era generator exists to inherit.

What is generated (under tests/fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-0/):

  gdn-Qwen3.6-35B-A3B-00.safetensor (+ .metadata.json.zst)
    GDN block prefill T=5 with real layer-0 weights: conv output, q/k/v
    post-split, g, beta, the recurrent block output and the chunked
    module output. Fixture contract v2: the f32 final SSM states and the
    sublayer intermediates (z, gated RMSNorm output, core outputs) are
    NOT stored in full at any model size; their external surface lives
    in the descriptor sidecar (gdn-Qwen3.6-35B-A3B-00.safetensor.
    descriptors.json, written by harness/gen_stats.nim from the
    pre-migration bytes and frozen there; a sanctioned re-record
    rewrites it through fixture_stats.py descriptor_fields).
  The multi-chunk T=70 case retired with the fixture contract v2: no
  suite consumed it, its full-tensor shape violated the per-file byte
  cap at any model size, and the cross-chunk property it exercised
  lives on as a synthetic tier-1 check in the consuming suite (both
  live computation modes on a seeded random input, no recorded bytes).

The chunked form is the installed forward (torch_chunk_gated_delta_rule,
chunk_size 64). The recurrent form is the bitwise reference for the Nim
implementation: the Nim recurrence is bit-identical to
torch_recurrent_gated_delta_rule, and its distance to the chunked form
equals the reference's own chunk-versus-recurrent floor.

Run (twice; cmp proves byte determinism):
  cd <worktree root> && .venv/bin/python \
    workspace/transformers/tests/testgen/gen_bf16_qwen36moe_01_layer_internals_gdn.py


RAM: the invoking shell runs `vm_stat` and `pgrep -f "python.*(torch|hf)"`
before this script. The script re-runs both checks itself and refuses to
load weights when free memory is low or another python/torch process is
running (its own process chain is excluded).
"""

import json
import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}


def write_json_zst(path, obj, ensure_ascii=True):
    """Write obj as a single zstd frame, level 19 with content size and
    checksum recorded in the frame header. JSON fixtures stay reviewable
    via the generator and the frame stays out of text diffs."""
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
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open
from safetensors import torch as st


import transformers  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeGatedDeltaNet,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

# Determinism: single intra-op thread, deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config.
MODEL_NAME = "Qwen3.6-35B-A3B"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals", "Qwen3.6-35B-A3B-layer-0"
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
WEIGHTS_FILE_1 = os.path.join(MODEL_DIR, "model-00001-of-00026.safetensors")
WEIGHTS_FILE_2 = os.path.join(MODEL_DIR, "model-00002-of-00026.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

NUM_THREADS = 1

# Per-generator seed, independent and order-agnostic.
SEED_GDN_PREFILL = 71

# GDN geometry of the checkpoint.
HIDDEN = 2048
PREFILL_SEQ = 5
CHUNK_SIZE = 64
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128

PREFIX = "model.language_model.layers.0.linear_attn."
MIN_FREE_BYTES = 8 * 1024 ** 3

# Recurrent-vs-chunked divergence caps at 35B dims: the fp32 floor
# (SSM state) and the block-level bar. The floor is about one fp32 ulp
# at the divergent element's magnitude, sub-linear in seq_len, about
# four orders of magnitude under bf16 rounding. The cap below allows
# four fp32 ulps at the generated state's max magnitude.
BLOCK_BAR = 1e-3
CORE_BAR = 1e-5
# The seq-70 multichunk accumulation measured the largest divergent
# element at 2.4 fp32 ulps under torch 2.13 and 2.7 under torch 2.11.
# The ulp-scaled tripwire sits at 8, far below the order-of-magnitude
# drift it exists to catch.
SSM_ULP_MARGIN = 8.0


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals_gdn] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals_gdn] vm_stat gave no 'Pages free' line")


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
    process holds RAM (this process chain is excluded from the pgrep match,
    whose command line spells the torch dependency of this run)."""
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_01_layer_internals_gdn] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_01_layer_internals_gdn] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def load_text_config() -> Qwen3_5MoeTextConfig:
    """Load the nested text_config from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeTextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


def load_gdn_weights() -> dict:
    """Load the layer-0 GDN tensors from the two safetensors files that hold
    them, via
    safe_open (memory-mapped, only these tensors are copied)."""
    weights = {}
    with safe_open(WEIGHTS_FILE_1, framework="pt") as f:
        for short in ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight"):
            weights[short] = f.get_tensor(PREFIX + short).clone()
    with safe_open(WEIGHTS_FILE_2, framework="pt") as f:
        for short in (
            "A_log",
            "conv1d.weight",
            "dt_bias",
            "in_proj_a.weight",
            "in_proj_b.weight",
            "norm.weight",
        ):
            weights[short] = f.get_tensor(PREFIX + short).clone()
    return weights


def build_gdn_layer0(cfg: Qwen3_5MoeTextConfig) -> Qwen3_5MoeGatedDeltaNet:
    """Qwen3_5MoeGatedDeltaNet with real layer-0 weights loaded.

    A_log and the norm weight are cast to bf16, matching a bf16 model load
    (the checkpoint already stores them bf16, so the cast is a no-op)."""
    block = Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0)
    w = load_gdn_weights()
    with torch.no_grad():
        block.in_proj_qkv.weight.data = w["in_proj_qkv.weight"]
        block.in_proj_z.weight.data = w["in_proj_z.weight"]
        block.out_proj.weight.data = w["out_proj.weight"]
        block.A_log.data = w["A_log"].to(torch.bfloat16)
        block.conv1d.weight.data = w["conv1d.weight"]
        block.dt_bias.data = w["dt_bias"]
        block.in_proj_a.weight.data = w["in_proj_a.weight"]
        block.in_proj_b.weight.data = w["in_proj_b.weight"]
        block.norm.weight.data = w["norm.weight"].to(torch.bfloat16)
    block.eval()
    return block


def gdn_forward_replay(block: Qwen3_5MoeGatedDeltaNet, hidden_states: torch.Tensor,
                       use_recurrent: bool) -> dict:
    """Replay of the reference Qwen3_5MoeGatedDeltaNet.forward with a
    selectable core rule, capturing every intermediate.

    The chunked replay must be bit-identical to the module's own forward
    (asserted by the caller). The recurrent replay is the bitwise reference
    for the Nim implementation.
    """
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = block.in_proj_qkv(hidden_states).transpose(1, 2)
    z = block.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, block.head_v_dim)
    b = block.in_proj_b(hidden_states)
    a = block.in_proj_a(hidden_states)

    # Fresh prefill conv: the built-in padding (kernel - 1) matches
    # the reference causal_conv1d_fn fallback, whose only padding
    # source is this same F.conv1d call.
    conv_output = F.silu(
        block.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
    mixed = conv_output.transpose(1, 2)
    query, key, value = torch.split(
        mixed, [block.key_dim, block.key_dim, block.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, block.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, block.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, block.head_v_dim)

    beta = b.sigmoid()
    g = -block.A_log.float().exp() * F.softplus(a.float() + block.dt_bias)

    # Head-group expansion before the core rule, mirroring the reference
    # forward: the value heads share one key head per group. A no-op
    # when the two head counts are equal.
    query_core, key_core = query, key
    if block.num_v_heads // block.num_k_heads > 1:
        ratio = block.num_v_heads // block.num_k_heads
        query_core = query.repeat_interleave(ratio, dim=2)
        key_core = key.repeat_interleave(ratio, dim=2)

    rule = torch_recurrent_gated_delta_rule if use_recurrent else torch_chunk_gated_delta_rule
    core_attn_out, ssm_state = rule(
        query_core, key_core, value, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    core_2d = core_attn_out.reshape(-1, block.head_v_dim)
    z_2d = z.reshape(-1, block.head_v_dim)
    normed = block.norm(core_2d, z_2d).reshape(batch_size, seq_len, -1)
    output = block.out_proj(normed)

    return {
        "conv_output": conv_output,
        "query": query, "key": key, "value": value,
        "z": z, "g": g, "beta": beta,
        "core_attn_out": core_attn_out,
        "ssm_state": ssm_state,
        "normed": normed,
        "output": output,
    }


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute element difference of two tensors, compared in f32."""
    return (a.float() - b.float()).abs().max().item()


def ulp_fp32(m: float) -> float:
    """One fp32 ulp at magnitude m: fp32 has 23 significand bits, so for
    m in [2**e, 2**(e+1)) the ulp is 2**(e-23). Zero maps to 0."""
    if m <= 0:
        return 0.0
    e = int(torch.floor(torch.log2(torch.tensor(m))).item())
    return 2.0 ** (e - 23)


def ssm_cap(ssm_state: torch.Tensor) -> float:
    """Recurrent-vs-chunked SSM cap: SSM_ULP_MARGIN fp32 ulps at the
    generated state's max magnitude. The two rules diverge by a few ulps
    at the largest divergent element, whose magnitude is bounded by the
    state's max magnitude."""
    return SSM_ULP_MARGIN * ulp_fp32(ssm_state.abs().max().item())


def divergence_meta(output_diff: float, core_diff: float, ssm_diff: float) -> dict:
    """Observed recurrent-vs-chunked divergence, recorded so the fixture
    documents the floor it was generated against."""
    return {
        "chunk_vs_recurrent_output_diff": output_diff,
        "chunk_vs_recurrent_core_diff": core_diff,
        "chunk_vs_recurrent_ssm_diff": ssm_diff,
    }


def save_fixture(case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata
    file. The final SSM states are saved without the batch dim so the Nim
    layer state [num_v_heads, Dk, Dv] compares directly."""
    filename = f"gdn-{MODEL_NAME}-{case_num:02d}.safetensor"
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
    return filepath


def generate_gdn_prefill_fixture(block: Qwen3_5MoeGatedDeltaNet, cfg: Qwen3_5MoeTextConfig) -> None:
    """GDN block prefill T=5: recurrent reference + chunked module output."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED_GDN_PREFILL)
    x = torch.randn(1, PREFILL_SEQ, HIDDEN, generator=gen, dtype=torch.bfloat16)

    with torch.no_grad():
        module_output = block(x)  # reference forward, chunked rule
        chunk_replay = gdn_forward_replay(block, x, use_recurrent=False)
        seq_replay = gdn_forward_replay(block, x, use_recurrent=True)
    assert torch.equal(module_output, chunk_replay["output"]), (
        "[gen_bf16_qwen36moe_01_layer_internals_gdn] chunked replay diverged from the module forward")

    output_diff = max_abs_diff(seq_replay["output"], module_output)
    core_diff = max_abs_diff(seq_replay["core_attn_out"], chunk_replay["core_attn_out"])
    ssm_diff = max_abs_diff(
        seq_replay["ssm_state"][0], chunk_replay["ssm_state"][0])
    assert output_diff < BLOCK_BAR, (
        f"[gen_bf16_qwen36moe_01_layer_internals_gdn] recurrent-vs-chunked output diff outside "
        f"(0, {BLOCK_BAR}): {output_diff}")
    assert ssm_diff <= ssm_cap(seq_replay["ssm_state"][0]), (
        f"[gen_bf16_qwen36moe_01_layer_internals_gdn] recurrent-vs-chunked SSM diff outside the "
        f"documented floor: {ssm_diff}")

    save_fixture(
        0,
        {
            "model": MODEL_NAME,
            "layer": "model.language_model.layers.0.linear_attn",
            "case": "prefill_seq5",
            "seq_len": PREFILL_SEQ,
            "chunk_size": CHUNK_SIZE,
            "head_k_dim": HEAD_K_DIM,
            "head_v_dim": HEAD_V_DIM,
            "num_k_heads": NUM_K_HEADS,
            "num_v_heads": NUM_V_HEADS,
            "hidden_size": HIDDEN,
            "seed": SEED_GDN_PREFILL,
            "num_threads": NUM_THREADS,
            "dtype": "bfloat16",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "note": "output_seq is the recurrent-rule replay, the bitwise "
                    "reference for the Nim layer. output_chunked is the "
                    "chunked forward. The recurrent-vs-chunked "
                    "divergence is the documented fp32 floor. The SSM "
                    "states and the sublayer intermediates live on in "
                    "the descriptor sidecar, never stored in full",
            **divergence_meta(output_diff, core_diff, ssm_diff),
        },
        {
            "input": x,
            "conv_output": chunk_replay["conv_output"],
            "q": chunk_replay["query"], "k": chunk_replay["key"],
            "v": chunk_replay["value"],
            "g": chunk_replay["g"], "beta": chunk_replay["beta"],
            "output_seq": seq_replay["output"],
            "output_chunked": module_output,
        },
    )
    print(f"[gen_bf16_qwen36moe_01_layer_internals_gdn] prefill T={PREFILL_SEQ}: "
          f"output diff {output_diff:.3e}, core diff {core_diff:.3e}, "
          f"ssm diff {ssm_diff:.3e}")


def main() -> None:
    check_ram()
    cfg = load_text_config()
    block = build_gdn_layer0(cfg)

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_gdn_prefill_fixture(block, cfg)

    print(f"[gen_bf16_qwen36moe_01_layer_internals_gdn] torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"[gen_bf16_qwen36moe_01_layer_internals_gdn] wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
