#!/usr/bin/env python3
"""Tier-01 layer-0 fixture generator, Kimi-Linear-48B-A3B-Instruct,
CPU torch bf16 from safetensors over the kimi-linear branch worktree
reference kernels, consumer tests/q_bf16/t_bf16_kimilinear_01_layer_internals.nim.

- single-file grammar, the payload carries the suite-read driving tensor kda.input, every recorded intermediate a stats fingerprint
- one KDA mixer scenario, the kernel-boundary prefill T=5, head 0 over real layer-0 weights, seed 421
- out_recurrent is the bitwise reference for the Nim block, out_chunked is the chunked form, the drift lands in the metadata

- the f32 log decay g and beta derive externally through the low-rank projections, the kernels feed directly
- recorded_from defaults to m4max-cpu (TTT_RECORD_FROM overrides), the --device argument fills the metadata device rows

Regenerate from the worktree root, twice for byte determinism, PYTHONPATH
carrying the branch worktree src:

  PYTHONPATH=<kimi-linear-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_kimilinear_01_layer_internals.py
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

from collections import OrderedDict
import os
import subprocess
import sys


import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open
from safetensors import torch as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import


# Reference kernels, the kimi-linear branch worktree, never an installed
# package. PYTHONPATH decides the resolution, the assertions below lock it.
import transformers.models.kimi_linear.modular_kimi_linear as kda_ref  # noqa: E402

BRANCH_MARKER = "kimi-linear-branch"
REFERENCE_COMMIT = "eea35a8513"
REFERENCE_BRANCH = "origin/kimi-linear"
REFERENCE_NOTE = "branch worktree, torch fallback kernels"

if BRANCH_MARKER not in kda_ref.__file__:
    raise SystemExit(
        "[gen_bf16_kimilinear_01_layer_internals] the imported kimi_linear module resolves to "
        f"{kda_ref.__file__}, not inside the {BRANCH_MARKER} worktree. Set "
        "PYTHONPATH to the branch worktree src, never to site-packages")


def branch_head() -> str:
    """HEAD of the reference worktree the module resolved from, asserted onto the recorded reference commit."""
    worktree = kda_ref.__file__.split("/src/")[0]
    out = subprocess.run(
        ["git", "-C", worktree, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    head = out.stdout.strip()
    if not head.startswith(REFERENCE_COMMIT):
        raise SystemExit(
            f"[gen_bf16_kimilinear_01_layer_internals] branch worktree HEAD {head} does not "
            f"match the recorded reference commit {REFERENCE_COMMIT}")
    return head


# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GRANDPARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "Kimi-Linear-48B-A3B-Instruct"
FIXTURE_ROOT = os.path.join(GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals")
FIXTURE_DIR = os.path.join(FIXTURE_ROOT, f"{MODEL_NAME}-layer-0")
FIXTURE_STEM = f"layer0-{MODEL_NAME}-00"
MODEL_DIR = os.path.join(os.path.dirname(GRANDPARENT_DIR), "tests", "hf_models", MODEL_NAME)

NUM_THREADS = 1

DEVICE = "cpu"
# Recording device, the --device argument overrides the default, the metadata device rows record the value the run used.
MIN_FREE_BYTES = 8 * 1024 ** 3

# Generation shape of the prefill case.
PREFILL_SEQ = 5
SEED_KDA_PREFILL = 421

# Layer-0 geometry of the checkpoint, the kernel-boundary contract of the port.
KDA_PREFIX = "model.layers.0.self_attn."
HIDDEN = 2304
KEY_SPAN = 4096
VALUE_SPAN = 4096
K_HEADS = 32
V_HEADS = 32
K_DIM = 128
V_DIM = 128
HEAD_LO, HEAD_HI = 0, 1

# Recurrent-vs-chunked drift bands, absolute measured floors:
#
# - the two reference evaluation orders diverge on this checkpoint family,
#   the extreme per-channel decay is data-dependent, layer-0 A_log exp
#   reaches 201 with g to -1.8e3 per step, far beyond the few-ulp state bound
# - reference floor over three seeds at T in {3, 64, 65}, state drift
#   1.6e-7 to 6.6e-5, output drift 7.5e-9 to 1.5e-5, the bands sit above
#   every measured floor
# - the measured margins land in the metadata rows, the consuming suite
#   budget rows are measured against them
DRIFT_BAND_OUTPUT = 1.0e-4
DRIFT_BAND_STATE = 1.0e-4


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_kimilinear_01_layer_internals] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_bf16_kimilinear_01_layer_internals] vm_stat gave no 'Pages free' line")


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
            f"[gen_bf16_kimilinear_01_layer_internals] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_kimilinear_01_layer_internals] other python/torch processes hold RAM: {stray}. "
            "Stop and retry when idle")


def shard_of(model_dir: str, key: str) -> str:
    """Shard filename holding one checkpoint tensor, read off the model.safetensors.index.json weight map."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise SystemExit(f"[gen_bf16_kimilinear_01_layer_internals] key {key} missing from the weight map")
    return weight_map[key]


def load_checkpoint_tensors(model_dir: str, keys: list) -> dict:
    """Memory-mapped loads of the named checkpoint tensors, one clone each."""
    weights = {}
    shards = {}
    for key in keys:
        shards[key] = shard_of(model_dir, key)
    for key, shard in sorted(shards.items()):
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            weights[key] = f.get_tensor(key).clone()
    return weights


def save_fixture(filepath: str, metadata: dict, input_tensor: torch.Tensor,
                 segments: OrderedDict) -> None:
    """Save one fixture safetensor carrying the bare driving tensor.

    Sidecars:

    - one zstd metadata frame with the mixture rows
    - one 004 stats frame with mixture-namespaced keys, the driving tensor
      and the op-surface fingerprints"""
    payload = {"kda.input": input_tensor.detach().cpu().contiguous()}
    serialized = st.save(payload, metadata=None)
    with open(filepath, "wb") as f:
        f.write(serialized)
    write_json_zst(filepath + ".metadata.json.zst", metadata)
    stats_entries = [("kda.input", payload["kda.input"])]
    stats_entries += [
        (f"kda.{name}", tensor.detach().cpu().contiguous())
        for name, tensor in segments.items()
    ]
    write_stats_file(filepath + ".stats.json.zst", os.path.basename(filepath),
                     stats_entries)


def ulp_fp32(m: float) -> float:
    """One fp32 ulp at magnitude m, the ulp at m in [2**e, 2**(e+1)) equals 2**(e-23), zero input maps to the zero ulp."""
    if m <= 0:
        return 0.0
    e = int(torch.floor(torch.log2(torch.tensor(m, device=DEVICE))).item())
    return 2.0 ** (e - 23)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute element difference of two tensors, compared in f32."""
    return (a.float() - b.float()).abs().max().item()


def reference_pair(q, k, v, g, beta):
    """Both reference kernels on one input set, final states requested, the l2norm flag matching the kernel-boundary contract of the port."""
    o_rec, s_rec = kda_ref.torch_recurrent_kda(
        q, k, v, g, beta, initial_state=None,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_chk, s_chk = kda_ref.torch_chunk_kda(
        q, k, v, g, beta, initial_state=None,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    return o_rec, s_rec, o_chk, s_chk


def selfcheck_inputs(g: torch.Tensor, beta: torch.Tensor, label: str) -> None:
    """Input-contract checks, a log decay never runs positive and beta stays
    inside (0, 1), a positive decay compounds exp(g) past 1."""
    gmax = g.max().item()
    if gmax > 0.0:
        raise SystemExit(
            f"[gen_bf16_kimilinear_01_layer_internals] {label}: log decay max {gmax} is positive, "
            "the input contract is violated")
    bmin = beta.min().item()
    bmax = beta.max().item()
    if not (0.0 < bmin and bmax < 1.0):
        raise SystemExit(
            f"[gen_bf16_kimilinear_01_layer_internals] {label}: beta range ({bmin}, {bmax}) "
            "leaves the open unit interval")


def selfcheck_pair(o_rec, s_rec, o_chk, s_chk, label: str) -> dict:
    """Reference recurrent-vs-chunked drift, asserted under the band guards and returned for the metadata rows."""
    o_diff = max_abs_diff(o_rec, o_chk)
    s_diff = max_abs_diff(s_rec, s_chk)
    s_max = s_rec.abs().max().item()
    if s_diff > DRIFT_BAND_STATE:
        raise SystemExit(
            f"[gen_bf16_kimilinear_01_layer_internals] {label}: recurrent-vs-chunked state drift "
            f"{s_diff:.3e} passes the {DRIFT_BAND_STATE:.1e} band")
    if o_diff > DRIFT_BAND_OUTPUT:
        raise SystemExit(
            f"[gen_bf16_kimilinear_01_layer_internals] {label}: recurrent-vs-chunked output drift "
            f"{o_diff:.3e} passes the {DRIFT_BAND_OUTPUT:.1e} band")
    return {
        "output_drift": o_diff,
        "state_drift": s_diff,
        "state_drift_fp32_ulps_at_max": s_diff / ulp_fp32(s_max) if s_max > 0 else 0.0,
    }


def reference_conv(qkv_proj, conv_weight, seq_len):
    """Reference causal short conv over the packed qkv projections, the causal_conv1d_fn torch fallback spelling, left padding then silu."""
    dim = qkv_proj.shape[1]
    out = F.conv1d(
        qkv_proj.to(conv_weight.dtype),
        weight=conv_weight.unsqueeze(1),
        padding=conv_weight.shape[-1] - 1,
        groups=dim,
    )[:, :, :seq_len]
    return F.silu(out).to(qkv_proj.dtype)


def split_heads(mixed, key_span, value_span, k_heads, v_heads, k_dim, v_dim):
    """Split the conv output into the head views (b, T, heads, dim)."""
    q, k, v = torch.split(mixed, [key_span, key_span, value_span], dim=-1)
    return (
        q.reshape(q.shape[0], q.shape[1], k_heads, k_dim),
        k.reshape(k.shape[0], k.shape[1], k_heads, k_dim),
        v.reshape(v.shape[0], v.shape[1], v_heads, v_dim),
    )


def head_slice_time(t: torch.Tensor):
    """Head slice of a time-major tensor, (b, T, heads, dim) kernel inputs and outputs, (b, T, heads) beta."""
    return t[:, :, HEAD_LO:HEAD_HI].contiguous()


def head_slice_state(t: torch.Tensor):
    """Head slice of a state tensor (b, heads, dk, dv), the head axis at dim 1."""
    return t[:, HEAD_LO:HEAD_HI].contiguous()


def generate_kda_prefill_fixture() -> None:
    """KDA kernel-boundary prefill T=5, head 0, both reference kernels on the real layer-0 weights, g and beta derived externally."""
    keys = [KDA_PREFIX + name for name in (
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
        "f_a_proj.weight", "f_b_proj.weight", "b_proj.weight", "A_log", "dt_bias")]
    weights = load_checkpoint_tensors(MODEL_DIR, keys)
    weights = {key: tensor.to(DEVICE) for key, tensor in weights.items()}

    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED_KDA_PREFILL)
    x = torch.randn(1, PREFILL_SEQ, HIDDEN, generator=gen,
                    dtype=torch.bfloat16)
    x = x.to(DEVICE)

    with torch.no_grad():
        q_proj = F.linear(x, weights[KDA_PREFIX + "q_proj.weight"])
        k_proj = F.linear(x, weights[KDA_PREFIX + "k_proj.weight"])
        v_proj = F.linear(x, weights[KDA_PREFIX + "v_proj.weight"])
        packed_conv = torch.cat(
            [weights[KDA_PREFIX + "q_conv1d.weight"],
             weights[KDA_PREFIX + "k_conv1d.weight"],
             weights[KDA_PREFIX + "v_conv1d.weight"]], dim=0)
        conv_out = reference_conv(
            torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2),
            packed_conv.squeeze(1), x.shape[1])
        q, k, v = split_heads(
            conv_out.transpose(1, 2), KEY_SPAN, VALUE_SPAN,
            K_HEADS, V_HEADS, K_DIM, V_DIM)
        # The low-rank gate weights derive g externally per the recorded contract,
        # the sigmoid path matches the reference module's forward.
        f_mid = F.linear(
            F.linear(x, weights[KDA_PREFIX + "f_a_proj.weight"]),
            weights[KDA_PREFIX + "f_b_proj.weight"])
        a_log = weights[KDA_PREFIX + "A_log"].float().reshape(V_HEADS, 1)
        dt_bias = weights[KDA_PREFIX + "dt_bias"].float().reshape(
            V_HEADS, V_DIM)
        g = -a_log.exp() * F.softplus(
            f_mid.reshape(x.shape[0], x.shape[1], V_HEADS,
                          V_DIM).float() + dt_bias)
        beta = torch.sigmoid(
            F.linear(x, weights[KDA_PREFIX + "b_proj.weight"]).float())

    label = f"{MODEL_NAME}-layer-0"
    selfcheck_inputs(g, beta, label)
    print(f"  {label}: g range [{g.min().item():.3e}, {g.max().item():.3e}], "
          f"beta range [{beta.min().item():.4f}, {beta.max().item():.4f}]")

    o_rec, s_rec, o_chk, s_chk = reference_pair(q, k, v, g, beta)
    margins = selfcheck_pair(o_rec, s_rec, o_chk, s_chk, f"{label} T={PREFILL_SEQ}")
    print(f"  {label} T={PREFILL_SEQ}: rec-vs-chunk o {margins['output_drift']:.3e},"
          f" s {margins['state_drift']:.3e}"
          f" ({margins['state_drift_fp32_ulps_at_max']:.2f} ulp at max)")

    segments = OrderedDict([
        ("query", head_slice_time(q)),
        ("key", head_slice_time(k)),
        ("value", head_slice_time(v)),
        ("log_decay", head_slice_time(g)),
        ("beta", head_slice_time(beta)),
        ("out_recurrent", head_slice_time(o_rec)),
        ("state_recurrent", head_slice_state(s_rec)),
        ("out_chunked", head_slice_time(o_chk)),
        ("state_chunked", head_slice_state(s_chk)),
    ])
    metadata = {
        "model": MODEL_NAME,
        "file": FIXTURE_STEM + ".safetensor",
        "dtype": "bfloat16",
        "layer": KDA_PREFIX.rstrip("."),
        "hidden_size": HIDDEN,
        "num_threads": NUM_THREADS,
        "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
        "device": DEVICE,
        "reference": f"{REFERENCE_BRANCH} {REFERENCE_COMMIT} ({REFERENCE_NOTE})",
        "reference_commit": branch_head(),
        "use_qk_l2norm_in_kernel": True,
        "kernel_boundary_contract":
            "q/k/v post-conv, g and beta derived externally and fed "
            "to the kernels, head slice of the full-head computation",
        "mixtures": {
            "kda": {
                "gate": "lowrank",
                "case": f"heads{HEAD_LO}_seq{PREFILL_SEQ}",
                "head_slice": [HEAD_LO, HEAD_HI],
                "num_heads_total": V_HEADS,
                "seq_len": PREFILL_SEQ,
                "head_k_dim": K_DIM,
                "head_v_dim": V_DIM,
                "seed": SEED_KDA_PREFILL,
                "qkv_dtype": "bfloat16",
                "g_dtype": "float32",
                "beta_dtype": "float32",
                "state_dtype": "float32",
                **{f"rec_vs_chunk_{k}": v for k, v in margins.items()},
            },
        },
    }
    filepath = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")
    save_fixture(filepath, metadata, x, segments)
    print(f"  wrote {filepath}")


def main() -> None:
    """Records the layer-0 KDA prefill fixture of the checkpoint."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    DEVICE = args.device
    check_ram()
    branch_head()
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_kda_prefill_fixture()
    print(f"[gen_bf16_kimilinear_01_layer_internals] torch {torch.__version__}")
    print(f"[gen_bf16_kimilinear_01_layer_internals] reference {REFERENCE_BRANCH} "
          f"{REFERENCE_COMMIT} ({REFERENCE_NOTE})")
    print(f"[gen_bf16_kimilinear_01_layer_internals] wrote {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
