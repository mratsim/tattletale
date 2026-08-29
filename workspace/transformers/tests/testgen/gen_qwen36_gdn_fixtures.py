#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B Gated DeltaNet (GDN) layer-0 fixtures from
the real checkpoint shards, using the vendored reference modeling on CPU
torch bf16.

What is generated (under tests/fixtures/qwen36-gdn/):

  gdn-Qwen3.6-35B-A3B-00.safetensor (+ .metadata.json)
    GDN block prefill T=5 with real layer-0 weights: conv output, q/k/v
    post-split, z, g, beta, the recurrent-rule core output and final SSM
    state, the chunked core output and final SSM state, the gated RMSNorm
    output, the recurrent block output and the chunked module output.
  gdn-Qwen3.6-35B-A3B-01.safetensor (+ .metadata.json)
    Multi-chunk prefill T=70 (two FLA chunks, 64 + 6): the recurrent and
    chunked core outputs, final SSM states and block outputs. The generator
    asserts the recurrent-vs-chunked divergence is exercised and inside the
    documented fp32 floor: about one fp32 ulp at the divergent element's
    magnitude, sub-linear in seq_len, about four orders of magnitude under
    bf16 rounding.

The chunked form is the vendored forward (torch_chunk_gated_delta_rule,
chunk_size 64). The recurrent form is the bitwise reference for the Nim
implementation: the Nim recurrence is bit-identical to
torch_recurrent_gated_delta_rule, and its distance to the chunked form
equals the reference's own chunk-versus-recurrent floor.

Run (twice; cmp proves byte determinism):
  cd <worktree root> && uv run --no-project --python 3.12 \
    --with "transformers @ file://<root>/_references_prod/transformers" \
    --with torch \
    workspace/transformers/tests/testgen/gen_qwen36_gdn_fixtures.py

RAM: the invoking shell runs `vm_stat` and `pgrep -f "python.*(torch|hf)"`
before this script. The script re-runs both checks itself and refuses to
load weights when free memory is low or another python/torch process is
running (its own process chain is excluded).
"""

import json
from collections import OrderedDict
import os
import subprocess
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors import torch as st

# The reference transformers checkout is the source of truth, found by walking
# up from this script until a _references_prod/transformers/src directory
# appears (a worktree sits one level deeper than the repo root).
# QWEN36_VENDORED_SRC overrides the walk.
_here = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _here
VENDORED_SRC = None
for _ in range(8):
    _candidate = os.path.join(_REPO_ROOT, "_references_prod", "transformers", "src")
    if os.path.isdir(_candidate):
        VENDORED_SRC = _candidate
        break
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
if VENDORED_SRC is None:
    VENDORED_SRC = os.environ.get("QWEN36_VENDORED_SRC")
if not VENDORED_SRC or not os.path.isdir(VENDORED_SRC):
    raise SystemExit(
        f"[gen_qwen36_gdn_fixtures] vendored modeling not found above {_here}. "
        "Set QWEN36_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, VENDORED_SRC)

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
FIXTURE_DIR = os.path.join(GRANDPARENT_DIR, "fixtures", "qwen36-gdn")
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
SHARD_1 = os.path.join(MODEL_DIR, "model-00001-of-00026.safetensors")
SHARD_2 = os.path.join(MODEL_DIR, "model-00002-of-00026.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# The vendored tree sha this fixture was generated against.
VENDORED_SHA = "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701"
NUM_THREADS = 1

# Per-generator seeds, independent and order-agnostic.
SEED_GDN_PREFILL = 71
SEED_MULTICHUNK = 72

# GDN geometry of the checkpoint.
HIDDEN = 2048
PREFILL_SEQ = 5
MULTICHUNK_SEQ = 70
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
SSM_ULP_MARGIN = 4.0


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_qwen36_gdn_fixtures] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_qwen36_gdn_fixtures] vm_stat gave no 'Pages free' line")


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
            f"[gen_qwen36_gdn_fixtures] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_qwen36_gdn_fixtures] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def check_vendored_sha() -> str:
    """Record the vendored tree sha and fail loudly on drift."""
    out = subprocess.run(
        ["git", "-C", VENDORED_SRC, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    sha = out.stdout.strip()
    if sha != VENDORED_SHA:
        raise SystemExit(
            f"[gen_qwen36_gdn_fixtures] vendored tree sha {sha} differs from "
            f"{VENDORED_SHA}; the fixture truth moved")
    return sha


def load_text_config() -> Qwen3_5MoeTextConfig:
    """Load the nested text_config from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeTextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


def load_gdn_weights() -> dict:
    """Load the layer-0 GDN tensors from the two shards that hold them, via
    safe_open (memory-mapped, only these tensors are copied)."""
    weights = {}
    with safe_open(SHARD_1, framework="pt") as f:
        for short in ("in_proj_qkv.weight", "in_proj_z.weight", "out_proj.weight"):
            weights[short] = f.get_tensor(PREFIX + short).clone()
    with safe_open(SHARD_2, framework="pt") as f:
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
    """Recurrent-vs-chunked SSM cap: four fp32 ulps at the generated
    state's max magnitude. The two rules diverge by about one ulp at the
    largest divergent element, whose magnitude is bounded by the state's
    max magnitude."""
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

    metadata_path = filepath + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")
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
        "[gen_qwen36_gdn_fixtures] chunked replay diverged from the module forward")

    output_diff = max_abs_diff(seq_replay["output"], module_output)
    core_diff = max_abs_diff(seq_replay["core_attn_out"], chunk_replay["core_attn_out"])
    ssm_diff = max_abs_diff(
        seq_replay["ssm_state"][0], chunk_replay["ssm_state"][0])
    assert output_diff < BLOCK_BAR, (
        f"[gen_qwen36_gdn_fixtures] recurrent-vs-chunked output diff outside "
        f"(0, {BLOCK_BAR}): {output_diff}")
    assert ssm_diff <= ssm_cap(seq_replay["ssm_state"][0]), (
        f"[gen_qwen36_gdn_fixtures] recurrent-vs-chunked SSM diff outside the "
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
                    "vendored chunked forward. The recurrent-vs-chunked "
                    "divergence is the documented fp32 floor",
            **divergence_meta(output_diff, core_diff, ssm_diff),
        },
        {
            "input": x,
            "conv_output": chunk_replay["conv_output"],
            "q": chunk_replay["query"], "k": chunk_replay["key"],
            "v": chunk_replay["value"],
            "z": chunk_replay["z"],
            "g": chunk_replay["g"], "beta": chunk_replay["beta"],
            "core_attn_out_seq": seq_replay["core_attn_out"],
            "core_attn_out_chunked": chunk_replay["core_attn_out"],
            "ssm_state_seq": seq_replay["ssm_state"][0],
            "ssm_state_chunked": chunk_replay["ssm_state"][0],
            "rmsnorm_gated_output": seq_replay["normed"],
            "output_seq": seq_replay["output"],
            "output_chunked": module_output,
        },
    )
    print(f"[gen_qwen36_gdn_fixtures] prefill T={PREFILL_SEQ}: "
          f"output diff {output_diff:.3e}, core diff {core_diff:.3e}, "
          f"ssm diff {ssm_diff:.3e}")


def generate_multichunk_fixture(block: Qwen3_5MoeGatedDeltaNet, cfg: Qwen3_5MoeTextConfig) -> None:
    """Multi-chunk GDN prefill T=70: lock the recurrent-vs-chunked band.

    Two FLA chunks (64 + 6) exercise the cross-chunk state handoff that a
    single-chunk prefill never touches. The chunked replay is asserted
    bit-identical to the module forward, and the recurrent-vs-chunked
    divergence is asserted inside the documented fp32 floor.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED_MULTICHUNK)
    x = torch.randn(1, MULTICHUNK_SEQ, HIDDEN, generator=gen, dtype=torch.bfloat16)

    with torch.no_grad():
        module_output = block(x)  # reference forward, chunked rule, two chunks
        chunk_replay = gdn_forward_replay(block, x, use_recurrent=False)
        seq_replay = gdn_forward_replay(block, x, use_recurrent=True)
    assert torch.equal(module_output, chunk_replay["output"]), (
        "[gen_qwen36_gdn_fixtures] chunked replay diverged from the module forward")

    output_diff = max_abs_diff(seq_replay["output"], module_output)
    core_diff = max_abs_diff(
        seq_replay["core_attn_out"], chunk_replay["core_attn_out"])
    ssm_diff = max_abs_diff(
        seq_replay["ssm_state"][0], chunk_replay["ssm_state"][0])
    assert 0.0 < output_diff < BLOCK_BAR, (
        f"[gen_qwen36_gdn_fixtures] recurrent-vs-chunked output diff outside "
        f"(0, {BLOCK_BAR}): {output_diff}")
    assert core_diff < CORE_BAR, (
        f"[gen_qwen36_gdn_fixtures] recurrent-vs-chunked core diff outside "
        f"(0, {CORE_BAR}): {core_diff}")
    assert 0.0 < ssm_diff <= ssm_cap(seq_replay["ssm_state"][0]), (
        f"[gen_qwen36_gdn_fixtures] recurrent-vs-chunked SSM diff outside the "
        f"documented floor: {ssm_diff}")

    save_fixture(
        1,
        {
            "model": MODEL_NAME,
            "layer": "model.language_model.layers.0.linear_attn",
            "case": "multichunk_prefill_seq70",
            "seq_len": MULTICHUNK_SEQ,
            "chunk_size": CHUNK_SIZE,
            "head_k_dim": HEAD_K_DIM,
            "head_v_dim": HEAD_V_DIM,
            "num_k_heads": NUM_K_HEADS,
            "num_v_heads": NUM_V_HEADS,
            "hidden_size": HIDDEN,
            "seed": SEED_MULTICHUNK,
            "num_threads": NUM_THREADS,
            "dtype": "bfloat16",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "note": "output_seq is the recurrent-rule replay, the bitwise "
                    "reference for the Nim layer. output_chunked is the "
                    "chunked forward. The recurrent-vs-chunked divergence "
                    "is about one fp32 ulp at the state magnitude, "
                    "sub-linear in seq_len",
            **divergence_meta(output_diff, core_diff, ssm_diff),
        },
        {
            "input": x,
            "output_seq": seq_replay["output"],
            "output_chunked": module_output,
            "core_attn_out_seq": seq_replay["core_attn_out"],
            "core_attn_out_chunked": chunk_replay["core_attn_out"],
            "ssm_state_seq": seq_replay["ssm_state"][0],
            "ssm_state_chunked": chunk_replay["ssm_state"][0],
        },
    )
    print(f"[gen_qwen36_gdn_fixtures] multichunk T={MULTICHUNK_SEQ}: "
          f"output diff {output_diff:.3e}, core diff {core_diff:.3e}, "
          f"ssm diff {ssm_diff:.3e}")


def main() -> None:
    check_ram()
    sha = check_vendored_sha()
    cfg = load_text_config()
    block = build_gdn_layer0(cfg)

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_gdn_prefill_fixture(block, cfg)
    generate_multichunk_fixture(block, cfg)

    print(f"[gen_qwen36_gdn_fixtures] torch {torch.__version__}, vendored sha {sha[:12]}")
    print(f"[gen_qwen36_gdn_fixtures] wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
