#!/usr/bin/env python3
"""Ling-3.0-tiny fixture generator over the real checkpoint shards, resolved through the gitignored tests/hf_models symlink.

MLA layer-3 records (tables-*, norm-*, qchain-*, rope-*, attn-*) on torch
bf16, BailingMoeV3MultiLatentAttention reference modeling, eager spelling:
- fixture dir tests/fixtures/bf16-01-layer-internals/Ling-3.0-tiny-layer-3/
- consumer tests/q_bf16/t_bf16_ling3_01_layer_internals_mla.nim

- <name>-Ling-3.0-tiny-<case>.safetensor, the recorded boundary slices of the case
- .metadata.json.zst, the case identity, the recorded geometry and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors

Recorded content:
- op-for-op replay slabs, the replay asserted equal to the module's own forward
- each attention case carries an SDPA counterpart, the eager-vs-sdpa gap
  stays measurable, the drift stays inspectable
- the rope tables record f32 pre-cast, the rotation runs the reference interleaved bf16 spelling
- the gate is the head-wise sigmoid spelling, multiplied before the dense projection

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root, PYTHONPATH pointing at the kimi-linear reference worktree src with the torch fallback kernels:
  PYTHONPATH=<kimi-linear-reference-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_ling3_01_layer_internals_mla.py
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
import types

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors import torch as st  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import



# Reference kernels, the kimi-linear reference worktree, never an installed
# package. PYTHONPATH decides the resolution, an assert below locks it:
# - this import runs BEFORE the fla stubs land in sys.modules
# - the reference module resolves without fla, both packages stay absent
#   in the worktree
import transformers.models.kimi_linear.modular_kimi_linear as kda_ref  # noqa: E402

BRANCH_MARKER = "kimi-linear-branch"
REFERENCE_COMMIT = "eea35a8513"
REFERENCE_BRANCH = "origin/kimi-linear"
REFERENCE_NOTE = "branch worktree, torch fallback kernels"

if BRANCH_MARKER not in kda_ref.__file__:
    raise SystemExit(
        "[ling3] the imported kimi_linear module "
        f"resolves to {kda_ref.__file__}, not inside the {BRANCH_MARKER} "
        "worktree. Set PYTHONPATH to the reference worktree src, never to "
        "site-packages")


def branch_head() -> str:
    """HEAD of the reference worktree the module resolved from, asserted onto the recorded reference commit."""
    worktree = kda_ref.__file__.split("/src/")[0]
    out = subprocess.run(
        ["git", "-C", worktree, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    head = out.stdout.strip()
    if not head.startswith(REFERENCE_COMMIT):
        raise SystemExit(
            f"[ling3] branch worktree HEAD {head} "
            f"does not match the recorded reference commit {REFERENCE_COMMIT}")
    return head



# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The checkpoint resolves through the gitignored tests/hf_models symlink.
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "Ling-3.0-tiny"
MODEL_DIR = os.path.join(
    os.path.dirname(TESTS_DIR), "tests", "hf_models", MODEL_NAME)
MLA_LAYER_IDX = 3
KDA_LAYER_IDX = 1
MLA_PREFIX = f"model.layers.{MLA_LAYER_IDX}.attention."
KDA_PREFIX = f"model.layers.{KDA_LAYER_IDX}.attention."
MLA_FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals",
    f"{MODEL_NAME}-layer-{MLA_LAYER_IDX}")
KDA_FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals",
    f"{MODEL_NAME}-layer-{KDA_LAYER_IDX}")

NUM_THREADS = 1

DEVICE = "cpu"
    # Recording device, the --device argument overrides the default, the metadata and env device rows record the value
    # the run used.

# Per-generator seeds, independent and order-agnostic.
SEED_TABLES = 241
SEED_NORM = 242
SEED_QCHAIN = 243
SEED_ROPE = 244
SEED_ATTN = 245
SEED_KDA_70 = 252
SEED_KDA_8 = 253
SEED_KDA_DECODE = 254
SEED_ONORM = 255

# MLA geometry of the checkpoint (config.json, measured at recording time).
HIDDEN = 1536
NUM_HEADS = 16
Q_LORA_RANK = 256
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = 192
V_HEAD_DIM = 128
ROPE_THETA = 6e6
SOFTMAX_SCALING = 0.07216878364870322

# KDA geometry of the checkpoint.
KDA_HEAD_DIM = 128
KDA_LOWER_BOUND = -5.0
KDA_EPS = 1e-6
CONV_KERNEL = 4

MIN_FREE_BYTES = 8 * 1024 ** 3

MLA_TENSORS = (
    "q_a_proj.weight", "q_b_proj.weight", "q_a_layernorm.weight",
    "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
    "dense.weight", "g_proj.weight",
)
KDA_TENSORS = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight",
    "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
    "A_log", "f_proj.weight", "dt_bias", "b_proj.weight",
    "g_proj.weight", "o_norm.weight", "o_proj.weight",
)



def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_ling3_01_layer_internals_mla] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit(
        "[gen_bf16_ling3_01_layer_internals_mla] vm_stat gave no 'Pages free' line")


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
            f"[gen_bf16_ling3_01_layer_internals_mla] free memory "
            f"{free / 1024 ** 3:.1f} GiB below the "
            f"{MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry "
            "when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            "[gen_bf16_ling3_01_layer_internals_mla] other python/torch "
            f"processes hold RAM: {stray}. Stop and retry when idle")



# ── fla stubs ──────────────────────────────────────────────────────────────

KDA_CAPTURE: dict = {}
"""Per-call capture of the derived safe gate, the kernel-boundary truth the
composition fixtures record beside the composition rows."""


def derive_safe_gate(g_raw, A_log, dt_bias, lower_bound):
    """Verified fla in-kernel safe-gate formula (fla/ops/kda/gate.py naive_kda_lowerbound_gate), spelled op for op in f32:
    - g = lower_bound * sigmoid(exp(A_log).view(H,1) * (g.float() + dt_bias.view(H,-1)))
    - g_raw sits at (batch, seq, heads, head_dim)"""
    heads = g_raw.shape[-2]
    g = g_raw.float()
    g = g + dt_bias.view(heads, -1)
    g = lower_bound * F.sigmoid(A_log.view(heads, 1).float().exp() * g)
    return g


def stub_chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None,
                   initial_state=None, output_final_state=False,
                   use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                   safe_gate=True, lower_bound=KDA_LOWER_BOUND,
                   cu_seqlens=None, **_):
    """fla chunk_kda stub, the verified formula derives g_safe and the call
    lands on the reference torch fallback kernel:
    - the derived g_safe lands in KDA_CAPTURE for the fixture rows"""
    assert use_gate_in_kernel and use_qk_l2norm_in_kernel and safe_gate
    g_safe = derive_safe_gate(g, A_log, dt_bias, lower_bound)
    KDA_CAPTURE["g_safe"] = g_safe
    return kda_ref.torch_chunk_kda(
        q, k, v, g_safe, beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True)


def stub_fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None,
                             initial_state=None, output_final_state=False,
                             use_qk_l2norm_in_kernel=False,
                             use_gate_in_kernel=False,
                             lower_bound=KDA_LOWER_BOUND, cu_seqlens=None, **_):
    """fla fused_recurrent_kda stub, the same g_safe derivation onto the reference recurrent kernel."""
    assert use_gate_in_kernel and use_qk_l2norm_in_kernel
    g_safe = derive_safe_gate(g, A_log, dt_bias, lower_bound)
    KDA_CAPTURE["g_safe"] = g_safe
    return kda_ref.torch_recurrent_kda(
        q, k, v, g_safe, beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True)


class StubFusedRMSNormGated(nn.Module):
    """Parameter-bearing FusedRMSNormGated stub, weight (head_dim,), eps
    from config.rms_norm_eps, activation sigmoid. Forward spells the fused
    single-rounding formula read off the installed fla 0.5.2 source
    (fla/modules/fused_norm_gate.py layer_norm_gated_fwd_kernel):
    - x f32, var = sum(x^2)/D, rstd = 1/sqrt(var + eps) spelled as the division
    - y = x*rstd*w, y *= sigmoid(g), one rounding at the end, stored back
      to the x dtype"""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5,
                 activation="swish", device=None, dtype=None):
        super().__init__()
        assert activation == "sigmoid" and elementwise_affine
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x, g):
        x32 = x.float()
        variance = x32.pow(2).mean(-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(variance + self.eps)
        out = x32 * rstd * self.weight.float()
        out = out * torch.sigmoid(g.float())
        return out.to(x.dtype)


class StubShortConvolution(nn.Conv1d):
    """Parameter-bearing ShortConvolution stub:
    a depthwise causal conv spelling the causal_conv1d_fn torch fallback math,
    silu after the causal window, cast back to the input dtype:
    - the carried state holds the last kernel-1 branch inputs per sequence,
      shape (N, D, kernel-1), oldest first
    - at decode with a carried state the state replaces the zero left padding,
      the causal window is exactly [history, x]
    - the real causal_conv1d_update consumes the same window, weight w0..w3
      ordered oldest..newest, so the grouped conv1d runs padding=0 over
      the history-prefixed input, and the nil-state zero padding never
      stacks on top of it"""

    def __init__(self, hidden_size, kernel_size, activation="silu", **_):
        super().__init__(
            hidden_size, hidden_size, kernel_size, groups=hidden_size,
            bias=False)
        assert activation == "silu"
        self.activation = activation
        self.hidden_size = hidden_size
        self.kernel = kernel_size

    def forward(self, x, cache=None, output_final_state=False, **_):
        # x (B, T, D) checkpoint dtype, batch 1 on this recording path.
        b, t, d = x.shape
        assert b == 1
        w = self.kernel
        if cache is not None:
            # cache (1, D, 1, W-1):
            #   the last W-1 branch inputs, oldest first
            history = cache[0, :, 0, :]
            window_in = torch.cat([history, x[0].transpose(0, 1)], dim=1)
        else:
            window_in = x[0].transpose(0, 1)  # (D, T) time-major rows
        # history-prefixed window, the state replaces the zero left padding,
        # no extra padding stacks on top of it
        pad = 0 if cache is not None else w - 1
        out = F.conv1d(
            window_in.unsqueeze(0).to(self.weight.dtype),
            weight=self.weight,
            padding=pad,
            groups=d,
        )[0, :, :t]
        out = F.silu(out).transpose(0, 1).to(x.dtype)
        new_cache = window_in[:, -(w - 1):].unsqueeze(0).unsqueeze(2)
        return out.unsqueeze(0), (new_cache if output_final_state else None)


def install_fla_stubs() -> None:
    """Injects the fla package stubs into sys.modules before the remote
    module import:
    - the installed fla-core is triton-only, its kernels cannot run here
    - the stubs carry the verified formulas and the reference torch fallback kernels"""
    fla = types.ModuleType("fla")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_kda = types.ModuleType("fla.ops.kda")
    fla_ops_simple_gla = types.ModuleType("fla.ops.simple_gla")
    fla_ops_sg_fr = types.ModuleType("fla.ops.simple_gla.fused_recurrent")
    fla_ops_sg_ch = types.ModuleType("fla.ops.simple_gla.chunk")
    fla_ops_utils = types.ModuleType("fla.ops.utils")
    fla_ops_utils_index = types.ModuleType("fla.ops.utils.index")
    fla_utils = types.ModuleType("fla.utils")
    fla_modules = types.ModuleType("fla.modules")

    fla_ops_kda.chunk_kda = stub_chunk_kda
    fla_ops_kda.fused_recurrent_kda = stub_fused_recurrent_kda
    fla_ops_sg_fr.fused_recurrent_simple_gla = None
    fla_ops_sg_ch.chunk_simple_gla = None
    fla_ops_utils_index.prepare_cu_seqlens_from_mask = None
    fla_ops_utils_index.prepare_lens_from_mask = None
    fla_utils.tensor_cache = lambda f: f
    fla_modules.FusedRMSNormGated = StubFusedRMSNormGated
    fla_modules.ShortConvolution = StubShortConvolution
    fla.ops = fla_ops
    fla.modules = fla_modules
    for name, mod in (
        ("fla", fla), ("fla.ops", fla_ops), ("fla.ops.kda", fla_ops_kda),
        ("fla.ops.simple_gla", fla_ops_simple_gla),
        ("fla.ops.simple_gla.fused_recurrent", fla_ops_sg_fr),
        ("fla.ops.simple_gla.chunk", fla_ops_sg_ch),
        ("fla.ops.utils", fla_ops_utils),
        ("fla.ops.utils.index", fla_ops_utils_index),
        ("fla.utils", fla_utils), ("fla.modules", fla_modules),
    ):
        sys.modules[name] = mod



# ── remote module classes ─────────────────────────────────────────────────

def apply_shims() -> None:
    """Three load-and-forward inert shims, recording-time measurements:
    - the removed is_torch_fx_available leaf
    - the ROPE_INIT_FUNCTIONS default entry, the rope_scaling restore runs
      right after the config load"""
    import transformers.utils.import_utils as import_utils
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False

    import transformers.modeling_rope_utils as rope_utils

    def _default_rope_init(config, device, seq_len=None, layer_type=None,
                           **kwargs):
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            config.rope_theta ** (
                torch.arange(0, dim, 2, dtype=torch.int64, device=DEVICE)
                .to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_init)


def load_remote_classes():
    """Returns the checkpoint config plus the three remote-code fixture
    classes:
    - rope_scaling restores to None (the 4.45 reality) before module
    - rope_scaling restores to None (the 4.45 reality) before module construction,
      the synthesized 5.x dict would crash the MLA init"""
    apply_shims()
    install_fla_stubs()
    from transformers import AutoConfig  # noqa: E402
    from transformers.dynamic_module_utils import (  # noqa: E402
        get_class_from_dynamic_module)

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    cfg.rope_scaling = None
    attn_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3.BailingMoeV3MultiLatentAttention", MODEL_DIR)
    kda_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3.BailingMoeV3KimiDeltaAttention", MODEL_DIR)
    rotary_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3.BailingMoeV3RotaryEmbedding", MODEL_DIR)
    return cfg, attn_cls, kda_cls, rotary_cls



def shard_of(key: str) -> str:
    """Shard filename holding one checkpoint tensor, read off the model.safetensors.index.json weight map."""
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise SystemExit(
            f"[gen_bf16_ling3_01_layer_internals_mla] key {key} missing from the "
            "weight map")
    return weight_map[key]


def load_weights(prefix: str, names) -> dict:
    """Memory-mapped loads of the named checkpoint tensors, one clone each."""
    weights = {}
    shards = {name: shard_of(prefix + name) for name in names}
    for name, shard in sorted(shards.items()):
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            weights[name] = f.get_tensor(prefix + name).clone()
    return weights


MLA_ATTRS = {
    "q_a_proj.weight": "q_a_proj.weight",
    "q_b_proj.weight": "q_b_proj.weight",
    "q_a_layernorm.weight": "q_a_layernorm.weight",
    "kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "kv_a_layernorm.weight",
    "kv_b_proj.weight": "kv_b_proj.weight",
    "dense.weight": "dense.weight",
    "g_proj.weight": "g_proj.weight",
}
KDA_ATTRS = {
    "q_proj.weight": "q_proj.weight",
    "k_proj.weight": "k_proj.weight",
    "v_proj.weight": "v_proj.weight",
    "q_conv1d.weight": "q_conv1d.weight",
    "k_conv1d.weight": "k_conv1d.weight",
    "v_conv1d.weight": "v_conv1d.weight",
    "A_log": "A_log",
    "f_proj.weight": "f_proj.weight",
    "dt_bias": "dt_bias",
    "b_proj.weight": "b_proj.weight",
    "g_proj.weight": "g_proj.weight",
    "o_norm.weight": "o_norm.weight",
    "o_proj.weight": "o_proj.weight",
}


def set_module_tensor(module: nn.Module, dotted: str, tensor) -> None:
    """Assigns one checkpoint tensor into a module parameter."""
    parts = dotted.split(".")
    leaf = module
    for part in parts[:-1]:
        leaf = getattr(leaf, part)
    target = getattr(leaf, parts[-1])
    with torch.no_grad():
        target.data = tensor


def save_fixture(directory: str, layer_name: str, case_num: int,
                 metadata: dict, tensors: dict) -> str:
    """Save one fixture safetensor plus its zstd metadata sidecar and the 004 stats sidecar."""
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(directory, filename)
    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().contiguous().clone())
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
    return filepath



COMMON_META = {
    "model": MODEL_NAME,
    "recorded_from": "m4max-cpu",
    "num_heads": NUM_HEADS,
    "q_lora_rank": Q_LORA_RANK,
    "kv_lora_rank": KV_LORA_RANK,
    "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
    "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
    "qk_head_dim": QK_HEAD_DIM,
    "v_head_dim": V_HEAD_DIM,
    "hidden_size": HIDDEN,
    "rope_theta": ROPE_THETA,
    "num_threads": NUM_THREADS,
    "dtype": "bfloat16",
    "python": ".".join(map(str, sys.version_info[:3])),
    "torch": torch.__version__,
    "reference_commit": branch_head(),
}



ROPE_WIDTH_META = {
    "rope_plane_width": QK_ROPE_HEAD_DIM,
    "rope_width_source": "qk_rope_head_dim, the module deepcopy sets "
        "head_dim = qk_rope_head_dim and partial_rotary_factor 1.0 before "
        "the rope_init_fn call",
    "rope_width_refused": "qk_head_dim * partial_rotary_factor = 96, the "
        "config field combination that must never reach the tables",
    "partial_rotary_factor_config": 0.5,
    "rope_interleave": True,
    "inv_freq_numel": QK_ROPE_HEAD_DIM // 2,
}



# ── reference rotation, copied from the remote modeling file ──────────────

def rotate_half(x):
    """Remote rotate_half, the cat of the negated second half and the first."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    """Remote apply_rotary_pos_emb_interleave, spelled op for op:
    - the pair view transpose seats the data in the half-split layout
    - the rotation runs the bf16 cos/sin tables, the output keeps the half-split layout"""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def precast_table_rows(rotary, positions: list) -> tuple:
    """cos/sin rows (len(positions), 32) f32 pre-cast from the reference rotary
    on an f32 carrier, attention_scaling applied (1.0 on this checkpoint):
    - the first half of the cat(freqs, freqs) table holds the unique
      per-pair angles the interleaved rotation consumes"""
    x = torch.zeros(1, len(positions), HIDDEN, dtype=torch.float32, device=DEVICE)
    pos = torch.tensor(positions, dtype=torch.long, device=DEVICE).reshape(1, len(positions))
    cos, sin = rotary(x, pos)
    return cos[0, :, :QK_ROPE_HEAD_DIM // 2].contiguous(), \
        sin[0, :, :QK_ROPE_HEAD_DIM // 2].contiguous()


def bf16_cos_sin(rotary, positions: list) -> tuple:
    """cos/sin (1, seq, 64) at the hidden dtype, exactly the position embeddings
the reference attention consumes."""
    x = torch.zeros(1, len(positions), HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    pos = torch.tensor(positions, dtype=torch.long, device=DEVICE).reshape(1, len(positions))
    cos, sin = rotary(x, pos)
    return cos, sin


def interleave_pairs(half_split: torch.Tensor) -> torch.Tensor:
    """Reference rotation output (cat of the even and odd pair values, the half-split layout) permuted back into the interleaved
    pair layout, pure data movement:
    - pair i lands at channels 2i and 2i+1"""
    b, h, s, d = half_split.shape
    even = half_split[..., :d // 2]
    odd = half_split[..., d // 2:]
    return torch.stack([even, odd], dim=-1).reshape(b, h, s, d)


def eager_attention(module, query, key, value):
    """Reference eager_attention_forward spelling on the MLA geometry:
    - repeat_kv2 with one group is the identity, the softmax runs f32,
      casts back to the query dtype, the value matmul runs at the query dtype"""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous()


def mla_forward_capture(attn, hidden_states, position_embeddings, cache):
    """Replay of the reference
    BailingMoeV3MultiLatentAttention.forward with intermediate capture:
    - decompress-before-cache, the reference expands the normed latent
      through kv_b_proj and caches the expanded key and value, the recorded
      cache tensors are the expanded slabs, the key slab nope 128 plus rope
      64 broadcast, the value slab 128
    - gate placement, the eager output scales by the head-wise sigmoid gate
      before the dense projection"""
    b, s = hidden_states.shape[:-1]
    q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states)))
    q_states = q_states.view(b, s, -1, attn.qk_head_dim).transpose(1, 2)
    q_pass, q_rot_raw = torch.split(
        q_states, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

    compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot_raw = torch.split(
        compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    k_pass = attn.kv_b_proj(attn.kv_a_layernorm(k_pass)).view(
        b, s, -1,
        attn.qk_nope_head_dim + attn.v_head_dim).transpose(1, 2)
    k_nope, value_states = torch.split(
        k_pass, [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1)

    k_rot = k_rot_raw.view(b, 1, s, attn.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot_raw, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_nope, k_rot), dim=-1)
    if cache is not None:
        key_states, value_states = cache.update(
            key_states, value_states, attn.layer_idx)

    attn_output = eager_attention(attn, query_states, key_states, value_states)
    gate = attn.g_proj(hidden_states)
    gate = F.sigmoid(gate.float()).type_as(hidden_states)
    gated = attn_output * gate[:, :, :, None]
    output = attn.dense(gated.reshape(b, s, -1).contiguous())

    sdpa_twin = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=None, dropout_p=0.0,
        is_causal=(query_states.shape[2] == key_states.shape[2]),
        scale=attn.scaling)
    # SDPA emits the sdpa-native layout (b, h, s, v), the eager spelling
    # above already transposed its output into the plane layout (b, s, h, v):
    # - the counterpart transposes to the plane layout too, the drift diff
    #   and the fixture rows compare plane to plane
    sdpa_twin = sdpa_twin.transpose(1, 2)
    return {
        "q_pass": q_pass, "q_rot_raw": q_rot_raw, "q_rot": q_rot,
        "k_nope": k_nope, "value_states": value_states,
        "key_states": key_states, "query_states": query_states,
        "attn_output": attn_output, "gate": gate, "gated": gated,
        "output": output, "sdpa_twin": sdpa_twin,
    }


def generate_tables_fixtures(rotary) -> None:
    """Model-level rotary construction fixture, inv_freq plus the cos/sin
    rows for positions 0..15, recorded f32 pre-cast:
    - the metadata carries the rope-width proof rows"""
    torch.manual_seed(SEED_TABLES)
    inv_freq = rotary.inv_freq.detach().clone().float()
    assert inv_freq.numel() == QK_ROPE_HEAD_DIM // 2, (
        f"inv_freq numel {inv_freq.numel()} is not the 64-channel plane half")
    assert float(rotary.attention_scaling) == 1.0
    cos_rows, sin_rows = precast_table_rows(rotary, list(range(16)))
    save_fixture(MLA_FIXTURE_DIR, "tables", 0, {
        **COMMON_META,
        **ROPE_WIDTH_META,
        "layer": MLA_PREFIX,
        "case": "plain_theta_table_rows_0_15",
        "seed": SEED_TABLES,
        "attention_scaling": float(rotary.attention_scaling),
        "softmax_scaling": SOFTMAX_SCALING,
        "max_seq_len_recorded": 16,
        "rope_reference_cast": "cos/sin recorded f32 pre-cast, the reference "
            "attention consumes the bf16 cast of these rows",
    }, {
        "inv_freq": inv_freq,
        "cos_rows": cos_rows,
        "sin_rows": sin_rows,
        "positions": torch.arange(16, dtype=torch.long, device=DEVICE),
    })
    print("Generated tables fixtures")



def generate_norm_fixtures(attn) -> None:
    """kv_a_layernorm fixtures using the real weight, the module default eps 1e-6
as the recorded spelling."""
    torch.manual_seed(SEED_NORM)
    cases = [
        (0, "prefill_latent", torch.randn(1, 8, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
        (1, "decode_latent", torch.randn(1, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
        (2, "zeros_input", torch.zeros(1, 4, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
    ]
    for case_num, case, x in cases:
        output = attn.kv_a_layernorm(x)
        save_fixture(MLA_FIXTURE_DIR, "norm", case_num, {
            **COMMON_META,
            "layer": MLA_PREFIX,
            "case": case,
            "eps": attn.kv_a_layernorm.variance_epsilon,
            "eps_note": "module default eps 1e-6, the reference attention "
                "constructs the latent norm without an eps argument",
        }, {
            "input": x, "output": output,
            "weight": attn.kv_a_layernorm.weight.data,
        })
    print("Generated norm fixtures")


def generate_qchain_fixtures(attn) -> None:
    """Compressed-Q chain per-op fixtures, q_a_proj, q_a_layernorm at the module-default eps, q_b_proj, real weights,
    the 256-wide bottleneck is the numerics class of this checkpoint's compressed-Q spelling."""
    torch.manual_seed(SEED_QCHAIN)
    cases = [
        (0, "prefill_seq8", torch.randn(1, 8, HIDDEN, dtype=torch.bfloat16, device=DEVICE)),
        (1, "decode_single_token", torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE)),
        (2, "zeros_input", torch.zeros(1, 4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)),
    ]
    for case_num, case, x in cases:
        q_a_out = attn.q_a_proj(x)
        q_normed = attn.q_a_layernorm(q_a_out)
        q_out = attn.q_b_proj(q_normed)
        save_fixture(MLA_FIXTURE_DIR, "qchain", case_num, {
            **COMMON_META,
            "layer": MLA_PREFIX,
            "case": case,
            "seed": SEED_QCHAIN,
            "eps": attn.q_a_layernorm.variance_epsilon,
            "eps_note": "module default eps 1e-6, the reference attention "
                "constructs q_a_layernorm without an eps argument",
            "q_lora_class": "q_a_proj GEMM, f32 RMS norm at module-default "
                "eps 1e-6 over the 256-wide bottleneck, one bf16 round, "
                "q_b_proj GEMM",
        }, {
            "input": x, "q_a_out": q_a_out, "q_normed": q_normed,
            "q_out": q_out, "weight": attn.q_a_layernorm.weight.data,
        })
    print("Generated qchain fixtures")



def generate_rope_fixtures(rotary) -> None:
    """Interleaved rope per-op fixtures on the 64-wide plane, q_pe and the single-head k_pe rotate against the recorded f32
    pre-cast tables, the reference rotation is the bf16 spelling."""
    torch.manual_seed(SEED_ROPE)

    def run_case(case_num, case, positions):
        seq_len = len(positions)
        cos, sin = precast_table_rows(rotary, positions)
        cos_b, sin_b = bf16_cos_sin(rotary, positions)
        q_pe = torch.randn(1, NUM_HEADS, seq_len, QK_ROPE_HEAD_DIM,
                           dtype=torch.bfloat16, device=DEVICE)
        k_pe = torch.randn(1, 1, seq_len, QK_ROPE_HEAD_DIM,
                           dtype=torch.bfloat16, device=DEVICE)
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos_b, sin_b)
        save_fixture(MLA_FIXTURE_DIR, "rope", case_num, {
            **COMMON_META,
            **ROPE_WIDTH_META,
            "layer": MLA_PREFIX,
            "case": case,
            "seed": SEED_ROPE,
            "positions": positions,
            "rope_bf16_reference": True,
            "rope_reference_cast": "cos/sin recorded f32 pre-cast, the "
                "reference rotation consumed the bf16 cast of these rows "
                "and ran the multiply-add chain in bf16",
            "rope_output_layout": "the reference interleave function emits "
                "cat(even pair values, odd pair values), the half-split "
                "layout the downstream attention consumes, q_rot/k_rot are "
                "that raw module output, q_rot_interleaved/"
                "k_rot_interleaved are the pure data-movement twins in the "
                "interleaved pair layout the nim rotation produces",
        }, {
            "q_pe": q_pe, "k_pe": k_pe,
            "cos": cos, "sin": sin,
            "q_rot": q_rot, "k_rot": k_rot,
            "q_rot_interleaved": interleave_pairs(q_rot),
            "k_rot_interleaved": interleave_pairs(k_rot),
        })

    run_case(0, "prefill_seq8", list(range(8)))
    run_case(1, "decode_single_token_pos5", [5])
    run_case(2, "scattered_positions", [3, 17, 255, 4095])
    print("Generated rope fixtures")



def generate_attn_fixtures(attn, rotary) -> None:
    """Gated MLA layer fixtures with real weights (layer 3), latent-cache path, eager reference spelling plus the SDPA counterpart:
    - the replay asserts equality with the module's own forward on a fresh preloaded cache"""
    torch.manual_seed(SEED_ATTN)

    def run_pass(x, positions, cache_layers):
        cos_b, sin_b = bf16_cos_sin(rotary, positions)

        def fresh_cache():
            cache = DynamicCache()
            for key_states, value_states in cache_layers:
                cache.update(key_states, value_states, MLA_LAYER_IDX)
            return cache

        cap = mla_forward_capture(attn, x, (cos_b, sin_b), fresh_cache())
        output_real, _, _ = attn(
            hidden_states=x,
            position_embeddings=(cos_b, sin_b),
            attention_mask=None,
            past_key_values=fresh_cache(),
        )
        assert torch.equal(output_real, cap["output"]), (
            "replay diverged from the real module forward")
        cos_rows, sin_rows = precast_table_rows(rotary, positions)
        cap["cos"] = cos_rows
        cap["sin"] = sin_rows
        cap["eager_sdpa_drift"] = (
            (cap["attn_output"].float() - cap["sdpa_twin"].float())
            .abs().max().item())
        return cap

    # Case 00 prefill seq 4, positions 0..3.
    torch.manual_seed(SEED_ATTN)
    x = torch.randn(1, 4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    cap = run_pass(x, list(range(4)), [])
    save_fixture(MLA_FIXTURE_DIR, "attn", 0, {
        **COMMON_META,
        **ROPE_WIDTH_META,
        "layer": MLA_PREFIX,
        "case": "prefill_seq4",
        "seed": SEED_ATTN,
        "positions": list(range(4)),
        "softmax_scaling": SOFTMAX_SCALING,
        "attention_spelling": "eager_attention_forward, softmax f32 cast "
            "back to the query dtype, value matmul at the query dtype",
        "gate_note": "head_wise gate: sigmoid(g_proj(x).float()).type_as(x), "
            "multiplied before the dense projection",
        "cache_note": "decompress-before-cache: the expanded key (nope 128 "
            "+ rope 64 broadcast) and value 128 are the cached slabs",
        "sdpa_twin_layout": "plane (b, s, h, v), the sdpa-native "
            "(b, h, s, v) output transposed for row comparability with "
            "attn_output",
        "eager_vs_sdpa_maxdiff": cap["eager_sdpa_drift"],
    }, {
        "hidden_states": x,
        "q_rot_raw": cap["q_rot_raw"].transpose(1, 2),
        "query_states": cap["query_states"],
        "key_states": cap["key_states"],
        "value_states": cap["value_states"],
        "attn_output": cap["attn_output"],
        "gate": cap["gate"],
        "gated": cap["gated"],
        "output": cap["output"],
        "sdpa_twin": cap["sdpa_twin"],
        "cos": cap["cos"], "sin": cap["sin"],
        "positions": torch.arange(4, dtype=torch.long, device=DEVICE),
    })

    # Case 01 prefill 2 then one decode step at position 2.
    torch.manual_seed(SEED_ATTN + 1)
    x = torch.randn(1, 2, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    x_step = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    cap = run_pass(x, [0, 1], [])
    cache_layers = [(cap["key_states"], cap["value_states"])]
    cap_step = run_pass(x_step, [2], cache_layers)
    save_fixture(MLA_FIXTURE_DIR, "attn", 1, {
        **COMMON_META,
        **ROPE_WIDTH_META,
        "layer": MLA_PREFIX,
        "case": "prefill2_decode1_pos2",
        "seed": SEED_ATTN + 1,
        "positions": [0, 1],
        "positions_step": [2],
        "softmax_scaling": SOFTMAX_SCALING,
        "attention_spelling": "eager_attention_forward, softmax f32 cast "
            "back to the query dtype, value matmul at the query dtype",
        "gate_note": "head_wise gate: sigmoid(g_proj(x).float()).type_as(x), "
            "multiplied before the dense projection",
        "cache_note": "decompress-before-cache: the expanded key and value "
            "are the cached slabs, the decode update returns the full slab",
        "sdpa_twin_layout": "plane (b, s, h, v), the sdpa-native "
            "(b, h, s, v) output transposed for row comparability with "
            "attn_output",
        "eager_vs_sdpa_maxdiff": max(cap["eager_sdpa_drift"],
                                     cap_step["eager_sdpa_drift"]),
    }, {
        "hidden_states": x, "x_step": x_step,
        "query_states": cap["query_states"],
        "query_states_step": cap_step["query_states"],
        "key_states": cap["key_states"],
        "value_states": cap["value_states"],
        "key_states_step": cap_step["key_states"],
        "value_states_step": cap_step["value_states"],
        "gate": cap["gate"],
        "gate_step": cap_step["gate"],
        "output": cap["output"],
        "output_step": cap_step["output"],
        "cos": cap["cos"], "sin": cap["sin"],
        "cos_step": cap_step["cos"], "sin_step": cap_step["sin"],
        "positions": torch.tensor([0, 1], dtype=torch.long, device=DEVICE),
    })

    # Case 02 prefill 3 then a 3-step decode sequence, positions 3, 4, 5.
    torch.manual_seed(SEED_ATTN + 2)
    x = torch.randn(1, 3, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    x_steps = [torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
               for _ in range(3)]
    cap = run_pass(x, [0, 1, 2], [])
    cache_layers = [(cap["key_states"], cap["value_states"])]
    tensors = {
        "hidden_states": x,
        "query_states": cap["query_states"],
        "key_states": cap["key_states"],
        "value_states": cap["value_states"],
        "gate": cap["gate"],
        "output": cap["output"],
        "cos": cap["cos"], "sin": cap["sin"],
        "positions": torch.tensor([0, 1, 2], dtype=torch.long, device=DEVICE),
    }
    step_positions = []
    drift = cap["eager_sdpa_drift"]
    for i, x_step in enumerate(x_steps):
        position = 3 + i
        step_positions.append(position)
        cap_step = run_pass(x_step, [position], cache_layers)
        cache_layers = [(cap_step["key_states"], cap_step["value_states"])]
        drift = max(drift, cap_step["eager_sdpa_drift"])
        tensors[f"x_step{i}"] = x_step
        tensors[f"cos_step{i}"] = cap_step["cos"]
        tensors[f"sin_step{i}"] = cap_step["sin"]
        tensors[f"query_states_step{i}"] = cap_step["query_states"]
        tensors[f"key_states_step{i}"] = cap_step["key_states"]
        tensors[f"value_states_step{i}"] = cap_step["value_states"]
        tensors[f"gate_step{i}"] = cap_step["gate"]
        tensors[f"output_step{i}"] = cap_step["output"]
    tensors["key_states_final"] = cap_step["key_states"]
    tensors["value_states_final"] = cap_step["value_states"]
    tensors["positions_steps"] = torch.tensor(step_positions, dtype=torch.long, device=DEVICE)
    save_fixture(MLA_FIXTURE_DIR, "attn", 2, {
        **COMMON_META,
        **ROPE_WIDTH_META,
        "layer": MLA_PREFIX,
        "case": "prefill3_decode3_steps",
        "seed": SEED_ATTN + 2,
        "positions": [0, 1, 2],
        "positions_steps": step_positions,
        "softmax_scaling": SOFTMAX_SCALING,
        "attention_spelling": "eager_attention_forward, softmax f32 cast "
            "back to the query dtype, value matmul at the query dtype",
        "sdpa_twin_layout": "plane (b, s, h, v), the sdpa-native "
            "(b, h, s, v) output transposed for row comparability with "
            "attn_output",
        "eager_vs_sdpa_maxdiff": drift,
    }, tensors)
    print("Generated attn fixtures")



def main() -> None:
    """Records the layer-3 MLA fixtures of the checkpoint."""
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
    cfg, attn_cls, _kda_cls, rotary_cls = load_remote_classes()

    mla_weights = load_weights(MLA_PREFIX, MLA_TENSORS)
    attn = attn_cls(cfg, layer_idx=MLA_LAYER_IDX)
    with torch.no_grad():
        for name, dotted in MLA_ATTRS.items():
            set_module_tensor(attn, dotted, mla_weights[name])
    attn = attn.to(DEVICE)
    attn.eval()
    assert attn.scaling == SOFTMAX_SCALING
    assert attn.q_a_layernorm.variance_epsilon == 1e-6
    assert attn.kv_a_layernorm.variance_epsilon == 1e-6
    assert tuple(attn.g_proj.weight.shape) == (NUM_HEADS, HIDDEN)
    assert cfg.gated_attention_proj_granularity_type == "head_wise"
    assert cfg.q_lora_rank == Q_LORA_RANK
    assert cfg.v_head_dim == V_HEAD_DIM

    rotary = rotary_cls(cfg)
    rotary = rotary.to(DEVICE)
    assert rotary.inv_freq.numel() == QK_ROPE_HEAD_DIM // 2
    assert float(rotary.attention_scaling) == 1.0

    os.makedirs(MLA_FIXTURE_DIR, exist_ok=True)
    generate_tables_fixtures(rotary)
    generate_norm_fixtures(attn)
    generate_qchain_fixtures(attn)
    generate_rope_fixtures(rotary)
    generate_attn_fixtures(attn, rotary)

    import transformers
    print(f"[gen_bf16_ling3_01_layer_internals_mla] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_ling3_01_layer_internals_mla] wrote {MLA_FIXTURE_DIR}")


if __name__ == "__main__":
    main()
