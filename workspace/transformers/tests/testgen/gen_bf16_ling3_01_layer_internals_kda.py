#!/usr/bin/env python3
"""Ling-3.0-tiny fixture generator over the real checkpoint shards, resolved through the gitignored tests/hf_models symlink.

KDA layer-1 per-op records (comp-*, onorm-*) on torch bf16:
BailingMoeV3KimiDeltaAttention modeling over the fla-stub kernel dispatch:
- fixture dir tests/fixtures/bf16-01-layer-internals/Ling-3.0-tiny-layer-1/
- consumer tests/q_bf16/t_bf16_ling3_01_layer_internals_kda.nim

- <name>-Ling-3.0-tiny-<case>.safetensor, the recorded boundary slices of the case
- .metadata.json.zst, the case identity, the recorded geometry and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors

Recorded content:
- prefill 70 (chunked), prefill 8 (fused_recurrent), prefill 8 plus 3 decode steps (state continuity)
- g_safe derives in the stub with the verified fla formula, recorded
  beside the composition rows
- fused o_norm one-shot rows, the direct reference for the FusedRmsNormGatedSigmoid type

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root, PYTHONPATH pointing at the kimi-linear reference worktree src with the torch fallback kernels:
  PYTHONPATH=<kimi-linear-reference-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_ling3_01_layer_internals_kda.py
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
from transformers.cache_utils import DynamicCache, DynamicLayer  # noqa: E402

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
        "[gen_bf16_ling3_01_layer_internals_kda] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit(
        "[gen_bf16_ling3_01_layer_internals_kda] vm_stat gave no 'Pages free' line")


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
            f"[gen_bf16_ling3_01_layer_internals_kda] free memory "
            f"{free / 1024 ** 3:.1f} GiB below the "
            f"{MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry "
            "when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            "[gen_bf16_ling3_01_layer_internals_kda] other python/torch "
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
            f"[gen_bf16_ling3_01_layer_internals_kda] key {key} missing from the "
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



# ── KDA: composition ───────────────────────────────────────────────────────

def split_heads(t, head_dim):
    """rearrange('... (h d) -> ... h d') spelling on a (b, t, h*d) tensor, pure
reshape on the contiguous layout."""
    return t.reshape(t.shape[0], t.shape[1], -1, head_dim)


def make_kda_cache(rec_state, conv_states, layer_idx=KDA_LAYER_IDX):
    """DynamicCache with the KDA state slotted in, the recurrent state under keys, the conv triple under values, matching
    the reference state read-back."""
    cache = DynamicCache()
    while len(cache.layers) <= layer_idx:
        cache.layers.append(DynamicLayer())
    if rec_state is not None:
        cache.layers[layer_idx].keys = rec_state
    if conv_states is not None:
        cache.layers[layer_idx].values = conv_states
    return cache


def kda_forward_capture(kda, x, conv_states, rec_state):
    """Replay of the reference BailingMoeV3KimiDeltaAttention.forward with intermediate capture:
    - mode dispatch follows the reference, the recurrent kernel iff
      q_len <= 64, the chunked kernel otherwise
    - g_safe derives inside the stub and lands in KDA_CAPTURE
    - the module's own forward over a fresh DynamicCache carrying the same
      state must equal the replay exactly"""
    b, q_len, _ = x.shape
    mode = "fused_recurrent" if q_len <= 64 else "chunk"
    KDA_CAPTURE.clear()

    q_proj_out = kda.q_proj(x)
    k_proj_out = kda.k_proj(x)
    v_proj_out = kda.v_proj(x)
    q, conv_state_q = kda.q_conv1d(
        x=q_proj_out, cache=conv_states[0] if conv_states else None,
        output_final_state=True)
    k, conv_state_k = kda.k_conv1d(
        x=k_proj_out, cache=conv_states[1] if conv_states else None,
        output_final_state=True)
    v, conv_state_v = kda.v_conv1d(
        x=v_proj_out, cache=conv_states[2] if conv_states else None,
        output_final_state=True)

    g_raw = kda.f_proj(x)
    beta = kda.b_proj(x).float().sigmoid()

    q = split_heads(q, kda.head_k_dim)
    k = split_heads(k, kda.head_k_dim)
    v = split_heads(v, kda.head_dim)
    g = split_heads(g_raw, kda.head_dim)

    if mode == "chunk":
        o, rec_new = stub_chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=kda.A_log, dt_bias=kda.dt_bias,
            initial_state=rec_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            safe_gate=kda.safe_gate, lower_bound=kda.lower_bound)
    else:
        o, rec_new = stub_fused_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=kda.A_log, dt_bias=kda.dt_bias,
            initial_state=rec_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            lower_bound=kda.lower_bound)
    g_safe = KDA_CAPTURE["g_safe"]

    g_out = split_heads(kda.g_proj(x), kda.head_dim)
    o_normed = kda.o_norm(o, g_out)
    o_flat = o_normed.reshape(b, q_len, -1)
    output = kda.o_proj(o_flat)

    # Module-forward equality:
    #   the module's own forward against a fresh
    # DynamicCache carrying the same state.
    cache = make_kda_cache(rec_state, conv_states)
    output_real, _, _ = kda(
        hidden_states=x, attention_mask=None, past_key_value=cache)
    assert torch.equal(output_real, output), (
        "replay diverged from the real module forward")

    return {
        "mode": mode,
        "q_proj_out": q_proj_out, "k_proj_out": k_proj_out,
        "v_proj_out": v_proj_out,
        "conv_q": q, "conv_k": k, "conv_v": v,
        "g_raw": g_raw, "beta": beta, "g_safe": g_safe,
        "o": o, "g_out": g_out, "o_normed": o_normed, "output": output,
        "rec_new": rec_new,
        "conv_states_new": (conv_state_q, conv_state_k, conv_state_v),
    }


def generate_comp_fixtures(kda) -> None:
    """KDA layer composition fixtures:
    - nil-state prefill 70 (chunked, the ragged second chunk covers 70 mod 64 = 6)
    - nil-state prefill 8 (fused_recurrent)
    - prefill 8 plus a 3-step decode sequence, the conv and recurrent state
      carry across the passes"""
    comp_meta = lambda case, seed, mode, extra=None: {
        **COMMON_META,
        "layer": KDA_PREFIX,
        "case": case,
        "seed": seed,
        "mode": mode,
        "kda_head_dim": KDA_HEAD_DIM,
        "kda_lower_bound": KDA_LOWER_BOUND,
        "kda_eps": KDA_EPS,
        "conv_kernel_size": CONV_KERNEL,
        "gate_formula": "safe gate derived in the stub with the verified "
            "fla formula (fla/ops/kda/gate.py naive_kda_lowerbound_gate), "
            "f32: lower_bound * sigmoid(exp(A_log).view(H,1) * (g + dt_bias))",
        "kernel_provenance": "kimi-linear branch torch fallback kernels, "
            "beta passed post-sigmoid",
        "conv_note": "parameter-bearing conv stub, causal_conv1d_fn torch "
            "fallback spelling, silu after the causal window, decode "
            "window exactly [history, x], no extra padding on the "
            "history-prefixed input",
        "o_norm_formula": "fused single rounding: f32 x*rstd*w*sigmoid(g), "
            "rstd spelled 1/sqrt as a division, one round at the end",
        **(extra or {}),
    }

    comp_tensors = lambda cap: {
        "q_proj_out": cap["q_proj_out"], "k_proj_out": cap["k_proj_out"],
        "v_proj_out": cap["v_proj_out"],
        "conv_q": cap["conv_q"], "conv_k": cap["conv_k"],
        "conv_v": cap["conv_v"],
        "g_raw": cap["g_raw"], "beta": cap["beta"],
        "g_safe": cap["g_safe"],
        "o": cap["o"], "g_out": cap["g_out"],
        "o_normed": cap["o_normed"], "output": cap["output"],
    }

    torch.manual_seed(SEED_KDA_70)
    x70 = torch.randn(1, 70, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    cap70 = kda_forward_capture(kda, x70, None, None)
    save_fixture(KDA_FIXTURE_DIR, "comp", 0,
                 comp_meta("prefill70_chunked_nil_state", SEED_KDA_70,
                           cap70["mode"]),
                 {"x": x70, **comp_tensors(cap70)})
    print("Generated comp-00")

    torch.manual_seed(SEED_KDA_8)
    x8 = torch.randn(1, 8, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    cap8 = kda_forward_capture(kda, x8, None, None)
    save_fixture(KDA_FIXTURE_DIR, "comp", 1,
                 comp_meta("prefill8_recurrent_nil_state", SEED_KDA_8,
                           cap8["mode"]),
                 {"x": x8, **comp_tensors(cap8)})
    print("Generated comp-01")

    torch.manual_seed(SEED_KDA_DECODE)
    xd = torch.randn(1, 8, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    capd = kda_forward_capture(kda, xd, None, None)
    tensors = {
        "x": xd,
        "beta": capd["beta"], "g_safe": capd["g_safe"],
        "output": capd["output"],
        "rec_state": capd["rec_new"],
        "conv_state_q": capd["conv_states_new"][0],
        "conv_state_k": capd["conv_states_new"][1],
        "conv_state_v": capd["conv_states_new"][2],
    }
    rec_state = capd["rec_new"]
    conv_states = capd["conv_states_new"]
    for i in range(3):
        x_step = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
        cap_step = kda_forward_capture(kda, x_step, conv_states, rec_state)
        rec_state = cap_step["rec_new"]
        conv_states = cap_step["conv_states_new"]
        tensors[f"x_step{i}"] = x_step
        tensors[f"beta_step{i}"] = cap_step["beta"]
        tensors[f"g_safe_step{i}"] = cap_step["g_safe"]
        tensors[f"conv_q_step{i}"] = cap_step["conv_q"]
        tensors[f"o_step{i}"] = cap_step["o"]
        tensors[f"o_normed_step{i}"] = cap_step["o_normed"]
        tensors[f"output_step{i}"] = cap_step["output"]
    tensors["rec_state_final"] = rec_state
    save_fixture(KDA_FIXTURE_DIR, "comp", 2,
                 comp_meta("prefill8_decode3_state_continuity",
                           SEED_KDA_DECODE, capd["mode"],
                           {"positions_steps": [8, 9, 10],
                            "continuity": "conv and recurrent states carry "
                                          "across the passes"}),
                 tensors)
    print("Generated comp-02")



def generate_onorm_fixtures(kda) -> None:
    """Fused o_norm one-shot rows, random bf16 o and sigmoid gate rows through the module stub, the FusedRmsNormGatedSigmoid reference."""
    torch.manual_seed(SEED_ONORM)
    cases = [
        (0, "prefill", torch.randn(1, 8, NUM_HEADS, KDA_HEAD_DIM,
                                   dtype=torch.bfloat16, device=DEVICE)),
        (1, "decode", torch.randn(1, 1, NUM_HEADS, KDA_HEAD_DIM,
                                  dtype=torch.bfloat16, device=DEVICE)),
        (2, "zeros_input", torch.zeros(1, 4, NUM_HEADS, KDA_HEAD_DIM,
                                       dtype=torch.bfloat16, device=DEVICE)),
    ]
    for case_num, case, o in cases:
        g = torch.randn(1, o.shape[1], NUM_HEADS, KDA_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
        out = kda.o_norm(o, g)
        save_fixture(KDA_FIXTURE_DIR, "onorm", case_num, {
            **COMMON_META,
            "layer": KDA_PREFIX,
            "case": case,
            "seed": SEED_ONORM,
            "eps": kda.o_norm.eps,
            "o_norm_formula": "fused single rounding: f32 x*rstd*w*sigmoid(g), "
                "rstd spelled 1/sqrt as a division, one round at the end",
        }, {
            "o": o, "g": g, "output": out,
            "weight": kda.o_norm.weight.data,
        })
    print("Generated onorm fixtures")



def main() -> None:
    """Records the layer-1 KDA per-op fixtures of the checkpoint."""
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
    cfg, _attn_cls, kda_cls, _rotary_cls = load_remote_classes()

    kda_weights = load_weights(KDA_PREFIX, KDA_TENSORS)
    kda = kda_cls(cfg, layer_idx=KDA_LAYER_IDX)
    with torch.no_grad():
        for name, dotted in KDA_ATTRS.items():
            set_module_tensor(kda, dotted, kda_weights[name])
    kda = kda.to(DEVICE)
    kda.eval()
    assert kda.A_log.dtype == torch.float32
    assert kda.dt_bias.dtype == torch.float32
    assert tuple(kda.o_norm.weight.shape) == (KDA_HEAD_DIM,)
    assert kda.o_norm.eps == KDA_EPS
    assert kda.lower_bound == KDA_LOWER_BOUND
    assert kda.safe_gate is True and kda.no_kda_lora is True
    assert kda.mode == "chunk"

    os.makedirs(KDA_FIXTURE_DIR, exist_ok=True)
    generate_comp_fixtures(kda)
    generate_onorm_fixtures(kda)

    import transformers
    print(f"[gen_bf16_ling3_01_layer_internals_kda] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_ling3_01_layer_internals_kda] wrote {KDA_FIXTURE_DIR}")


if __name__ == "__main__":
    main()
