#!/usr/bin/env python3
"""Kimi-Linear-48B-A3B-Instruct fixture generator over the real checkpoint shards, resolved through the gitignored tests/hf_models symlink.

KDA layer-0 per-op records (kda-*) through the reference kernels
torch_recurrent_kda and torch_chunk_kda (torch fallback spellings) on torch bf16:
- fixture dir tests/fixtures/bf16-01-layer-internals/Kimi-Linear-48B-A3B-Instruct-layer-0/
- consumer tests/q_bf16/t_bf16_kimi_01_layer_internals_kda.nim

- kda-Kimi-Linear-48B-A3B-Instruct-<head><tlen>.safetensor, the head slices 0 and 1 at T in {1, 64, 65}
- .metadata.json.zst, the case identity, the recorded geometry and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors

Recorded content:
- kernel-boundary inputs q/k/v post-conv, g (f32, externally derived) and beta
- both kernels' outputs and final states, margin rows for the recurrent-vs-chunked floor per length

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root, PYTHONPATH pointing at the kimi-linear reference
worktree src carrying the torch fallback kernels:
  PYTHONPATH=<kimi-linear-reference-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_kimi_01_layer_internals_kda.py
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
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open
from safetensors import torch as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import



# Reference kernels, the kimi-linear reference worktree, never an installed
# package. PYTHONPATH decides the resolution, the assertions below lock it.
import transformers.models.kimi_linear.modular_kimi_linear as kda_ref  # noqa: E402

BRANCH_MARKER = "kimi-linear-branch"
REFERENCE_COMMIT = "eea35a8513"
REFERENCE_BRANCH = "origin/kimi-linear"
REFERENCE_NOTE = "branch worktree, torch fallback kernels"

if BRANCH_MARKER not in kda_ref.__file__:
    raise SystemExit(
        "[gen_bf16_kimi_01_layer_internals_kda] the imported kimi_linear module resolves to "
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
            f"[gen_bf16_kimi_01_layer_internals_kda] branch worktree HEAD {head} does not "
            f"match the recorded reference commit {REFERENCE_COMMIT}")
    return head


# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

GRANDPARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_ROOT = os.path.join(GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals")
MODEL_NAME = "Kimi-Linear-48B-A3B-Instruct"
MODEL_DIR = os.path.join(os.path.dirname(GRANDPARENT_DIR), "tests", "hf_models", MODEL_NAME)

NUM_THREADS = 1

DEVICE = "cpu"
    # Recording device, the --device argument overrides the default, the metadata and env device rows record the value the run used.
MIN_FREE_BYTES = 8 * 1024 ** 3

# Recurrent-vs-chunked self-check bands, absolute measured floors:
# - the two reference evaluation orders diverge on this checkpoint
#   family under the extreme per-channel decay, the layer-0 A_log exp
#   reaches 201 and g reaches -1.8e3 per step, the gap is data-dependent
#   and far beyond the few-ulp GDN bound at the state max
# - reference floor over three seeds at T in {3, 64, 65}, state drift
#   1.6e-7 to 6.6e-5, output drift 7.5e-9 to 1.5e-5
# - the guard sits above every measured floor, worst 6.6e-5, 1.5x below
#   the band, far below any structural fault
# - the per-T measured margins land in the metadata rows, the consuming
#   suite budget rows are measured against them
DRIFT_BAND_OUTPUT = 1.0e-4
DRIFT_BAND_STATE = 1.0e-4

# Sequence lengths:
#   1 the decode step, 3 the mid-boundary loop, 64 and 65
# the mode boundary from both sides, 70 the ragged second chunk.
REAL_SEQ_LENS = (1, 64, 65)
SYNTH_SEQ_LENS = (1, 3, 64, 65, 70)
CONT_SEQ_LENS = (1, 65)


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_kimi_01_layer_internals_kda] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_bf16_kimi_01_layer_internals_kda] vm_stat gave no 'Pages free' line")


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
            f"[gen_bf16_kimi_01_layer_internals_kda] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_kimi_01_layer_internals_kda] other python/torch processes hold RAM: {stray}. "
            "Stop and retry when idle")


def shard_of(model_dir: str, key: str) -> str:
    """Shard filename holding one checkpoint tensor, read off the model.safetensors.index.json weight map."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise SystemExit(f"[gen_bf16_kimi_01_layer_internals_kda] key {key} missing from the weight map")
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


def save_fixture(filepath: str, metadata: dict, tensors: dict) -> None:
    """Save one fixture safetensor plus its zstd metadata sidecar and the 004 stats sidecar."""
    # Clone forces fresh storage:
    # - batch-1 time slices of a contiguous base already count as contiguous
    #   (the size-1 leading dim stride is ignored)
    # - safetensors refuses shared storage
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


def ulp_fp32(m: float) -> float:
    """One fp32 ulp at magnitude m, the ulp at m in [2**e, 2**(e+1)) equals 2**(e-23), zero input maps to the zero ulp."""
    if m <= 0:
        return 0.0
    e = int(torch.floor(torch.log2(torch.tensor(m, device=DEVICE))).item())
    return 2.0 ** (e - 23)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute element difference of two tensors, compared in f32."""
    return (a.float() - b.float()).abs().max().item()


def reference_pair(q, k, v, g, beta, initial_state=None):
    """Both reference kernels on one input set, final states requested, the l2norm flag matching the kernel-boundary contract of the port."""
    o_rec, s_rec = kda_ref.torch_recurrent_kda(
        q, k, v, g, beta, initial_state=initial_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_chk, s_chk = kda_ref.torch_chunk_kda(
        q, k, v, g, beta, initial_state=initial_state,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    return o_rec, s_rec, o_chk, s_chk


def selfcheck_inputs(g: torch.Tensor, beta: torch.Tensor, label: str) -> None:
    """Input-contract checks, a log decay never runs positive and beta stays
    inside (0, 1), a positive decay compounds exp(g) past 1."""
    gmax = g.max().item()
    if gmax > 0.0:
        raise SystemExit(
            f"[gen_bf16_kimi_01_layer_internals_kda] {label}: log decay max {gmax} is positive, "
            "the input contract is violated")
    bmin = beta.min().item()
    bmax = beta.max().item()
    if not (0.0 < bmin and bmax < 1.0):
        raise SystemExit(
            f"[gen_bf16_kimi_01_layer_internals_kda] {label}: beta range ({bmin}, {bmax}) "
            "leaves the open unit interval")


def selfcheck_pair(o_rec, s_rec, o_chk, s_chk, label: str) -> dict:
    """Reference recurrent-vs-chunked drift, asserted under the band guards and returned for the metadata rows."""
    o_diff = max_abs_diff(o_rec, o_chk)
    s_diff = max_abs_diff(s_rec, s_chk)
    s_max = s_rec.abs().max().item()
    if s_diff > DRIFT_BAND_STATE:
        raise SystemExit(
            f"[gen_bf16_kimi_01_layer_internals_kda] {label}: recurrent-vs-chunked state drift "
            f"{s_diff:.3e} passes the {DRIFT_BAND_STATE:.1e} band")
    if o_diff > DRIFT_BAND_OUTPUT:
        raise SystemExit(
            f"[gen_bf16_kimi_01_layer_internals_kda] {label}: recurrent-vs-chunked output drift "
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


def seq_len_tag(t: int) -> str:
    """Fixture name tag of one recorded sequence length, `t70` for 70."""
    return f"t{t}"


def head_slice_time(t: torch.Tensor, head_lo: int, head_hi: int):
    """Head slice of a time-major tensor, (b, T, heads, dim) kernels inputs and outputs, (b, T, heads) beta."""
    return t[:, :, head_lo:head_hi].contiguous()


def head_slice_state(t: torch.Tensor, head_lo: int, head_hi: int):
    """Head slice of a state tensor (b, heads, dk, dv), the head axis at dim 1."""
    return t[:, head_lo:head_hi].contiguous()


def real_variant(variant: dict) -> None:
    """One real checkpoint variant, real weights, head-sliced per-op records, g and beta derived externally and fed to the kernels."""
    model_dir = MODEL_DIR
    prefix = variant["prefix"]
    keys = [prefix + name for name in (
        "q_proj.weight", "k_proj.weight", "v_proj.weight",
        "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
        variant["f_lo"], variant["f_hi"], "b_proj.weight", "A_log", "dt_bias")]
    weights = load_checkpoint_tensors(model_dir, keys)
    weights = {key: tensor.to(DEVICE) for key, tensor in weights.items()}

    seed = variant["seed"]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hidden = variant["hidden"]
    x = torch.randn(1, max(REAL_SEQ_LENS), hidden, generator=gen,
                    dtype=torch.bfloat16)
    x = x.to(DEVICE)

    with torch.no_grad():
        q_proj = F.linear(x, weights[prefix + "q_proj.weight"])
        k_proj = F.linear(x, weights[prefix + "k_proj.weight"])
        v_proj = F.linear(x, weights[prefix + "v_proj.weight"])
        packed_conv = torch.cat(
            [weights[prefix + "q_conv1d.weight"],
             weights[prefix + "k_conv1d.weight"],
             weights[prefix + "v_conv1d.weight"]], dim=0)
        conv_out = reference_conv(
            torch.cat([q_proj, k_proj, v_proj], dim=-1).transpose(1, 2),
            packed_conv.squeeze(1), x.shape[1])
        q, k, v = split_heads(
            conv_out.transpose(1, 2), variant["key_span"],
            variant["value_span"], variant["k_heads"], variant["v_heads"],
            variant["k_dim"], variant["v_dim"])
        # The gate g and beta weights derive externally per the recorded
        # contract and the reference module itself derives the same values.
        if variant["gate"] == "lowrank":
            f_mid = F.linear(
                F.linear(x, weights[prefix + variant["f_lo"]]),
                weights[prefix + variant["f_hi"]])
            a_log = weights[prefix + "A_log"].float().reshape(
                variant["v_heads"], 1)
            dt_bias = weights[prefix + "dt_bias"].float().reshape(
                variant["v_heads"], variant["v_dim"])
            g = -a_log.exp() * F.softplus(
                f_mid.reshape(x.shape[0], x.shape[1], variant["v_heads"],
                              variant["v_dim"]).float() + dt_bias)
        else:
            f = F.linear(x, weights[prefix + variant["f_lo"]])
            a_log = weights[prefix + "A_log"].float().reshape(
                variant["v_heads"], 1)
            dt_bias = weights[prefix + "dt_bias"].float().reshape(
                variant["v_heads"], variant["v_dim"])
            arg = f.reshape(x.shape[0], x.shape[1], variant["v_heads"],
                            variant["v_dim"]).float() + dt_bias
            g = variant["lower_bound"] * torch.sigmoid(a_log.exp() * arg)
        beta = torch.sigmoid(
            F.linear(x, weights[prefix + "b_proj.weight"]).float())

    label = variant["dir"]
    selfcheck_inputs(g, beta, label)
    print(f"  {label}: g range [{g.min().item():.3e}, {g.max().item():.3e}], "
          f"beta range [{beta.min().item():.4f}, {beta.max().item():.4f}]")

    margins = {}
    for seq_len in REAL_SEQ_LENS:
        q_t, k_t, v_t = q[:, :seq_len], k[:, :seq_len], v[:, :seq_len]
        g_t, beta_t = g[:, :seq_len], beta[:, :seq_len]
        o_rec, s_rec, o_chk, s_chk = reference_pair(q_t, k_t, v_t, g_t, beta_t)
        margins[seq_len_tag(seq_len)] = selfcheck_pair(
            o_rec, s_rec, o_chk, s_chk, f"{label} T={seq_len}")
        print(f"  {label} T={seq_len}: rec-vs-chunk o {margins[seq_len_tag(seq_len)]['output_drift']:.3e},"
              f" s {margins[seq_len_tag(seq_len)]['state_drift']:.3e}"
              f" ({margins[seq_len_tag(seq_len)]['state_drift_fp32_ulps_at_max']:.2f} ulp at max)")

    fixture_dir = os.path.join(FIXTURE_ROOT, variant["dir"])
    os.makedirs(fixture_dir, exist_ok=True)
    for seq_len in REAL_SEQ_LENS:
        q_t, k_t, v_t = q[:, :seq_len], k[:, :seq_len], v[:, :seq_len]
        g_t, beta_t = g[:, :seq_len], beta[:, :seq_len]
        o_rec, s_rec, o_chk, s_chk = reference_pair(q_t, k_t, v_t, g_t, beta_t)
        for head in variant["head_slices"]:
            tensors = {
                "query": head_slice_time(q_t, head, head + 1),
                "key": head_slice_time(k_t, head, head + 1),
                "value": head_slice_time(v_t, head, head + 1),
                "log_decay": head_slice_time(g_t, head, head + 1),
                "beta": head_slice_time(beta_t, head, head + 1),
                "out_recurrent": head_slice_time(o_rec, head, head + 1),
                "state_recurrent": head_slice_state(s_rec, head, head + 1),
                "out_chunked": head_slice_time(o_chk, head, head + 1),
                "state_chunked": head_slice_state(s_chk, head, head + 1),
            }
            metadata = {
                "model": variant["model"],
                "layer": variant["prefix"].rstrip("."),
                "gate": variant["gate"],
                "case": f"heads{head}_t{seq_len}",
                "head_slice": [head, head + 1],
                "num_heads_total": variant["v_heads"],
                "seq_len": seq_len,
                "head_k_dim": variant["k_dim"],
                "head_v_dim": variant["v_dim"],
                "hidden_size": variant["hidden"],
                "seed": seed,
                "num_threads": NUM_THREADS,
                "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
                "device": DEVICE,
                "qkv_dtype": "bfloat16",
                "g_dtype": "float32",
                "beta_dtype": "float32",
                "state_dtype": "float32",
                "reference": f"{REFERENCE_BRANCH} {REFERENCE_COMMIT} ({REFERENCE_NOTE})",
                "reference_commit": branch_head(),
                "use_qk_l2norm_in_kernel": True,
                "kernel_boundary_contract":
                    "q/k/v post-conv, g and beta derived externally and fed "
                    "to the kernels, head slice of the full-head computation",
                **{f"rec_vs_chunk_{k}": v for k, v in margins[seq_len_tag(seq_len)].items()},
            }
            filepath = os.path.join(
                fixture_dir,
                f"kda-{variant['stem']}-{head:02d}{seq_len_tag(seq_len)}.safetensor")
            save_fixture(filepath, metadata, tensors)
            print(f"  wrote {filepath}")


def variant_definition() -> list:
    """One fixture variant, checkpoint path, weight vocabulary, geometry, gate derivation and seed."""
    return [
        {
            "model": MODEL_NAME,
            "dir": f"{MODEL_NAME}-layer-0",
            "stem": MODEL_NAME,
            "prefix": "model.layers.0.self_attn.",
            "gate": "lowrank",
            "f_lo": "f_a_proj.weight",
            "f_hi": "f_b_proj.weight",
            "hidden": 2304,
            "key_span": 4096,
            "value_span": 4096,
            "k_heads": 32,
            "v_heads": 32,
            "k_dim": 128,
            "v_dim": 128,
            "head_slices": (0, 1),
            "seed": 421,
        },
    ]


def main() -> None:
    """Records the layer-0 KDA per-op fixtures of the checkpoint."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    check_ram()
    branch_head()
    for variant in variant_definition():
        real_variant(variant)
    print(f"[gen_bf16_kimi_01_layer_internals_kda] torch {torch.__version__}")
    print(f"[gen_bf16_kimi_01_layer_internals_kda] reference {REFERENCE_BRANCH} "
          f"{REFERENCE_COMMIT} ({REFERENCE_NOTE})")
    print(f"[gen_bf16_kimi_01_layer_internals_kda] wrote {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
