#!/usr/bin/env python3
"""GLM-4.7-Flash fixture generator over the real checkpoint shards, the gitignored
tests/hf_models symlink resolves the checkpoint.

MLA layer-0 records (tables-*, norm-*, qchain-*, rope-*, attn-*) through the installed
reference modeling, Glm4MoeLiteAttention with the sdpa interface, on torch bf16:
- fixture dir tests/fixtures/bf16-01-layer-internals/GLM-4.7-Flash-layer-0/
- consumer tests/q_bf16/t_bf16_glm47flash_01_layer_internals_mla.nim

- <name>-GLM-4.7-Flash-<case>.safetensor, the recorded boundary slices of the case
- .metadata.json.zst, the case identity, the recorded geometry and margins
- .stats.json.zst, the ttt-tf-004-uniform-stats frame over the floating-point payload tensors

Recorded content:
- rotary table rows f32 pre-cast, the reference consumes the bf16 cast
- latent norm and compressed-Q chain at the module-default eps 1e-6
- bf16-spelled interleaved rope with the interleaved-layout reorders
- the latent-cache attention replay (prefill and decode cases), asserted equal to the module's own forward
- no Qwen3 analog exists, the Qwen3 and Qwen3.5 families are plain multi-head attention, no MLA rung exists

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root
  uv run python workspace/transformers/tests/testgen/gen_bf16_glm47flash_01_layer_internals_mla.py
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


import transformers  # noqa: E402
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
    Glm4MoeLiteConfig,
)
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteRotaryEmbedding,
    apply_rotary_pos_emb_interleave,
)
from transformers.cache_utils import DynamicCache  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import


# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The checkpoint resolves through the gitignored tests/hf_models symlink.
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = "GLM-4.7-Flash"
LAYER_IDX = 0
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals",
    f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests", "hf_models", MODEL_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

with open(CONFIG_PATH) as f:
    _CKPT_CONFIG = json.load(f)

NUM_THREADS = 1

DEVICE = "cpu"
# Recording device, the --device argument overrides the default, both the metadata and the env device rows record the value the run used.

# Per-generator seeds, independent and order-agnostic.
SEED_TABLES = 231
SEED_NORM = 232
SEED_QCHAIN = 233
SEED_ROPE = 234
SEED_ATTN = 235

# MLA geometry of the checkpoint, config is king, the parsed values equal the 20/768/512/192/64/256/2048 tuple and theta 1000000
# the checkpoint carries.
NUM_HEADS = _CKPT_CONFIG["num_attention_heads"]
Q_LORA_RANK = _CKPT_CONFIG["q_lora_rank"]
KV_LORA_RANK = _CKPT_CONFIG["kv_lora_rank"]
QK_NOPE_HEAD_DIM = _CKPT_CONFIG["qk_nope_head_dim"]
QK_ROPE_HEAD_DIM = _CKPT_CONFIG["qk_rope_head_dim"]
V_HEAD_DIM = _CKPT_CONFIG["v_head_dim"]
HIDDEN = _CKPT_CONFIG["hidden_size"]
ROPE_THETA = _CKPT_CONFIG["rope_theta"]
SOFTMAX_SCALING = 0.0625
MAX_SEQ = 8192

PREFIX = f"model.layers.{LAYER_IDX}.self_attn."
ATTN_TENSORS = (
    "q_a_proj.weight",
    "q_b_proj.weight",
    "q_a_layernorm.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
)
MIN_FREE_BYTES = 8 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_glm47flash_01_layer_internals_mla] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit(
        "[gen_bf16_glm47flash_01_layer_internals_mla] vm_stat gave no 'Pages free' line")


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
    """Refuses to load weights under low memory or a stray python/torch process, the pgrep match excludes this process chain."""
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"[gen_bf16_glm47flash_01_layer_internals_mla] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_glm47flash_01_layer_internals_mla] other python/torch processes hold RAM: {stray}. "
            "Stop and retry when idle")


def shard_of(key: str) -> str:
    """Shard filename holding one checkpoint tensor, read off the model.safetensors.index.json weight map, main-stack layer 0 tensors only."""
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise SystemExit(
            f"[gen_bf16_glm47flash_01_layer_internals_mla] key {key} missing from the weight map")
    return weight_map[key]


def load_config() -> Glm4MoeLiteConfig:
    """Load the checkpoint config.json (flat glm4_moe_lite layout). The rope_parameters default lands on the plain theta spelling:
      rope_type default, theta 1000000, rope_interleave true, partial_rotary_factor 1.0."""
    with open(CONFIG_PATH) as f:
        cfg = Glm4MoeLiteConfig.from_dict(json.load(f))
    cfg._attn_implementation = "sdpa"
    return cfg


def load_layer0_weights() -> dict:
    """Load the seven layer-0 self_attn tensors from the shards that hold them, via safe_open (memory-mapped, only these tensors are copied)."""
    weights = {}
    shards = {name: shard_of(PREFIX + name) for name in ATTN_TENSORS}
    for name, shard in shards.items():
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            weights[name] = f.get_tensor(PREFIX + name).clone()
    return weights


def build_attention(weights: dict, cfg: Glm4MoeLiteConfig) -> Glm4MoeLiteAttention:
    """Glm4MoeLiteAttention with real layer-0 weights loaded."""
    attn = Glm4MoeLiteAttention(cfg, layer_idx=LAYER_IDX)
    with torch.no_grad():
        for name in ATTN_TENSORS:
            module = {
                "q_a_proj.weight": attn.q_a_proj,
                "q_b_proj.weight": attn.q_b_proj,
                "q_a_layernorm.weight": attn.q_a_layernorm,
                "kv_a_proj_with_mqa.weight": attn.kv_a_proj_with_mqa,
                "kv_a_layernorm.weight": attn.kv_a_layernorm,
                "kv_b_proj.weight": attn.kv_b_proj,
                "o_proj.weight": attn.o_proj,
            }[name]
            module.weight.data = weights[name]
    attn.eval()
    return attn


def precast_table_rows(rotary: Glm4MoeLiteRotaryEmbedding,
                       positions: list[int]) -> tuple:
    """cos/sin rows (len(positions), 32) f32 pre-cast from the reference rotary, attention_scaling already applied."""
    x = torch.zeros(1, len(positions), HIDDEN, dtype=torch.float32, device=DEVICE)
    pos = torch.tensor(positions, dtype=torch.long, device=DEVICE).reshape(1, len(positions))
    cos, sin = rotary(x, pos)
    return cos[0, :, :QK_ROPE_HEAD_DIM // 2].contiguous(), \
        sin[0, :, :QK_ROPE_HEAD_DIM // 2].contiguous()


def interleave_pairs(half_split: torch.Tensor) -> torch.Tensor:
    """Pure data movement reorder, the reference half-split rotation output permuted into the interleaved pair layout."""
    b, h, s, d = half_split.shape
    even = half_split[..., :d // 2]
    odd = half_split[..., d // 2:]
    return torch.stack([even, odd], dim=-1).reshape(b, h, s, d)


def bf16_cos_sin(rotary: Glm4MoeLiteRotaryEmbedding,
                 positions: list[int]) -> tuple:
    """cos/sin (1, seq, 64) at the hidden dtype, exactly the position embeddings the reference attention consumes."""
    x = torch.zeros(1, len(positions), HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    pos = torch.tensor(positions, dtype=torch.long, device=DEVICE).reshape(1, len(positions))
    cos, sin = rotary(x, pos)
    return cos, sin


def save_fixture(layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
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
    "layer": PREFIX,
    "num_heads": NUM_HEADS,
    "q_lora_rank": Q_LORA_RANK,
    "kv_lora_rank": KV_LORA_RANK,
    "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
    "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
    "v_head_dim": V_HEAD_DIM,
    "hidden_size": HIDDEN,
    "rope_theta": ROPE_THETA,
    "rope_type": "default",
    "rope_interleave": True,
    "num_threads": NUM_THREADS,
    "dtype": "bfloat16",
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
}


# ── Rotary construction ────────────────────────────────────────────────────

def generate_tables_fixtures(rotary: Glm4MoeLiteRotaryEmbedding) -> None:
    """Rotary construction fixture, plain-theta inv_freq plus the cos/sin rows for positions 0..15, recorded f32 pre-cast."""
    torch.manual_seed(SEED_TABLES)
    inv_freq = rotary.inv_freq.detach().clone().float()  # (32,) f32
    assert inv_freq.numel() == 32
    cos_rows, sin_rows = precast_table_rows(rotary, list(range(16)))
    save_fixture(
        "tables", 0,
        {
            **COMMON_META,
            "case": "plain_theta_table_rows_0_15",
            "attention_scaling": float(rotary.attention_scaling),
            "softmax_scaling": SOFTMAX_SCALING,
            "max_seq_len_recorded": 16,
            "rope_reference_cast": "cos/sin recorded f32 pre-cast, the reference "
                "attention consumes the bf16 cast of these rows",
        },
        {
            "inv_freq": inv_freq,
            "cos_rows": cos_rows,
            "sin_rows": sin_rows,
            "positions": torch.arange(16, dtype=torch.long, device=DEVICE),
        },
    )
    print("Generated tables fixtures")


# ── RMSNorm ────────────────────────────────────────────────────────────────

def generate_norm_fixtures(kv_a_layernorm) -> None:
    """kv_a_layernorm fixtures using the real weight, the module default eps 1e-6 as the recorded spelling."""
    torch.manual_seed(SEED_NORM)
    layer_name = "norm"
    cases = [
        (0, "prefill_latent", torch.randn(1, 8, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
        (1, "decode_latent", torch.randn(1, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
        (2, "zeros_input", torch.zeros(1, 4, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)),
    ]
    for case_num, case, x in cases:
        output = kv_a_layernorm(x)
        save_fixture(
            layer_name, case_num,
            {
                **COMMON_META,
                "case": case,
                "eps": kv_a_layernorm.variance_epsilon,
                "eps_note": "module default eps 1e-6, the reference attention "
                    "constructs the latent norm without an eps argument",
            },
            {"input": x, "output": output, "weight": kv_a_layernorm.weight.data},
        )
    print(f"Generated {layer_name} fixtures")


# ── Compressed-Q chain per-op ─────────────────────────────────────────────

def generate_qchain_fixtures(attn: Glm4MoeLiteAttention) -> None:
    """q_a_proj -> q_a_layernorm -> q_b_proj chain fixtures on real weights, one f32 norm over the 768-wide bottleneck between the two GEMMs."""
    torch.manual_seed(SEED_QCHAIN)
    layer_name = "qchain"
    cases = [
        (0, "prefill_seq8", torch.randn(1, 8, HIDDEN, dtype=torch.bfloat16, device=DEVICE)),
        (1, "decode_single_token", torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE)),
        (2, "zeros_input", torch.zeros(1, 4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)),
    ]
    for case_num, case, x in cases:
        q_a_out = attn.q_a_proj(x)
        q_normed = attn.q_a_layernorm(q_a_out)
        q_out = attn.q_b_proj(q_normed)
        save_fixture(
            layer_name, case_num,
            {
                **COMMON_META,
                "case": case,
                "eps": attn.q_a_layernorm.variance_epsilon,
                "eps_note": "module default eps 1e-6, the reference attention "
                    "constructs q_a_layernorm without an eps argument",
                "q_lora_class": "q_a_proj GEMM, f32 RMS norm at eps 1e-6 over "
                    "the 768-wide bottleneck, one bf16 round, q_b_proj GEMM",
            },
            {
                "input": x,
                "q_a_out": q_a_out,
                "q_normed": q_normed,
                "q_out": q_out,
                "weight": attn.q_a_layernorm.weight.data,
            },
        )
    print(f"Generated {layer_name} fixtures")


# ── apply_rotary_pos_emb_interleave per-op ────────────────────────────────

def generate_rope_fixtures(rotary: Glm4MoeLiteRotaryEmbedding) -> None:
    """Interleaved rope per-op fixtures, q_pe and the single-head k_pe rotate against the recorded f32 pre-cast tables."""
    torch.manual_seed(SEED_ROPE)
    layer_name = "rope"

    def run_case(case_num, case, positions):
        seq_len = len(positions)
        cos, sin = precast_table_rows(rotary, positions)
        cos_b, sin_b = bf16_cos_sin(rotary, positions)
        q_pe = torch.randn(1, NUM_HEADS, seq_len, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        k_pe = torch.randn(1, 1, seq_len, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos_b, sin_b)
        q_rot_il = interleave_pairs(q_rot)
        k_rot_il = interleave_pairs(k_rot)
        save_fixture(
            layer_name, case_num,
            {
                **COMMON_META,
                "case": case,
                "positions": positions,
                "rope_bf16_reference": True,
                "rope_reference_cast": "cos/sin recorded f32 pre-cast, the reference "
                    "rotation consumed the bf16 cast of these rows and ran the "
                    "multiply-add chain in bf16",
                "rope_output_layout": "the reference interleave function emits "
                    "cat(even pair values, odd pair values), the half-split "
                    "layout the downstream attention consumes, q_rot/k_rot are "
                    "that raw module output, q_rot_interleaved/k_rot_interleaved "
                    "are the pure data-movement twins in the interleaved pair "
                    "layout the nim rotation produces",
            },
            {
                "q_pe": q_pe, "k_pe": k_pe,
                "cos": cos, "sin": sin,
                "q_rot": q_rot, "k_rot": k_rot,
                "q_rot_interleaved": q_rot_il,
                "k_rot_interleaved": k_rot_il,
            },
        )

    # Case 00 prefill, positions 0..7.
    run_case(0, "prefill_seq8", list(range(8)))
    # Case 01 decode, single token at nonzero position.
    run_case(1, "decode_single_token_pos5", [5])
    # Case 02 scattered positions through the plain-theta range.
    run_case(2, "scattered_positions", [3, 17, 255, 4095])
    print(f"Generated {layer_name} fixtures")


# ── MLA layer forward with capture ────────────────────────────────────────

def attention_forward_capture(attn, hidden_states, cos, sin, cache):
    """Replay of the reference Glm4MoeLiteAttention.forward with intermediate capture, asserted equal to the module's own forward before saving."""
    batch_size, seq_length = hidden_states.shape[:-1]
    q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states)))
    q_states = q_states.view(
        batch_size, seq_length, -1, attn.qk_head_dim).transpose(1, 2)
    q_pass, q_pe_raw = torch.split(
        q_states, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

    compressed_kv = attn.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_rot_raw = torch.split(
        compressed_kv, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    latent_normed = attn.kv_a_layernorm(kv_nope).view(
        batch_size, 1, seq_length, attn.kv_lora_rank)
    k_rot = k_rot_raw.view(batch_size, 1, seq_length, attn.qk_rope_head_dim)

    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_pe_raw, k_rot, cos, sin)

    # Cache update happens after the rotation in the reference forward,
    # so the reference caches the rotated k_rot.
    if cache is not None:
        latent_cached, kpe_cached = cache.update(latent_normed, k_rot, attn.layer_idx)
    else:
        latent_cached, kpe_cached = latent_normed, k_rot

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states, value_states = attn.expand_kv(latent_cached, kpe_cached)

    is_causal = seq_length > 1
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states,
        attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=attn.scaling,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    output = attn.o_proj(attn_output.reshape(
        batch_size, seq_length, -1).contiguous())
    return {
        "query_states": query_states,
        "q_pe": q_pe_raw,
        "latent_normed": latent_normed,
        "latent_cached": latent_cached, "kpe_cached": kpe_cached,
        "key_states": key_states, "value_states": value_states,
        "attn_output": attn_output, "output": output,
    }


def generate_attn_fixtures(attn: Glm4MoeLiteAttention,
                           rotary: Glm4MoeLiteRotaryEmbedding) -> None:
    """MLA layer fixtures with real weights (layer 0), latent-cache path, prefill payloads kept short to respect the fixture file cap."""
    torch.manual_seed(SEED_ATTN)
    layer_name = "attn"

    def fresh_cache(cache_layers):
        cache = DynamicCache()
        for i, (lat, kpe) in enumerate(cache_layers):
            cache.update(lat, kpe, i)
        return cache

    def run_pass(attn, x, positions, cache_layers):
        # Replay and module forward each get a fresh cache preloaded
        # with the prior passes' cached tensors, so neither run advances
        # the state the other reads.
        cos, sin = bf16_cos_sin(rotary, positions)
        cap = attention_forward_capture(attn, x, cos, sin,
            fresh_cache(cache_layers))
        output_real, _ = attn(
            hidden_states=x,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=fresh_cache(cache_layers),
        )
        assert torch.equal(output_real, cap["output"]), (
            "replay diverged from real forward")
        cos_rows, sin_rows = precast_table_rows(rotary, positions)
        cap["cos"] = cos_rows
        cap["sin"] = sin_rows
        return cap

    # Case 00 prefill seq 4, positions 0..3, cache write of the full pass.
    torch.manual_seed(SEED_ATTN)
    x = torch.randn(1, 4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    cap = run_pass(attn, x, list(range(4)), [])
    save_fixture(
        layer_name, 0,
        {
            **COMMON_META,
            "case": "prefill_seq4",
            "positions": list(range(4)),
            "seed": SEED_ATTN,
            "softmax_scaling": SOFTMAX_SCALING,
            "rope_reference_cast": "cos/sin recorded f32 pre-cast, the reference "
                "rotation consumed the bf16 cast of these rows and ran the "
                "multiply-add chain in bf16",
            "rope_output_layout": "the rotated q channels of query_states sit "
                "in the reference half-split layout (cat of even and odd pair "
                "values), the layout the recorded sdpa consumed",
        },
        {
            "hidden_states": x,
            "q_pe": cap["q_pe"],
            "query_states": cap["query_states"],
            "latent_normed": cap["latent_normed"],
            "latent_cached": cap["latent_cached"],
            "kpe_cached": cap["kpe_cached"],
            "kpe_cached_interleaved": interleave_pairs(cap["kpe_cached"]),
            "key_states": cap["key_states"],
            "value_states": cap["value_states"],
            "attn_output": cap["attn_output"],
            "cos": cap["cos"], "sin": cap["sin"],
            "positions": torch.arange(4, dtype=torch.long, device=DEVICE),
        },
    )

    # Case 01 prefill 2 then one decode step at position 2.
    torch.manual_seed(SEED_ATTN + 1)
    x = torch.randn(1, 2, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    x_step = torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    cap = run_pass(attn, x, [0, 1], [])
    cache_layers = [(cap["latent_cached"], cap["kpe_cached"])]
    cap_step = run_pass(attn, x_step, [2], cache_layers)
    save_fixture(
        layer_name, 1,
        {
            **COMMON_META,
            "case": "prefill2_decode1_pos2",
            "positions": [0, 1],
            "positions_step": [2],
            "seed": SEED_ATTN + 1,
            "softmax_scaling": SOFTMAX_SCALING,
            "rope_reference_cast": "cos/sin recorded f32 pre-cast, the reference "
                "rotation consumed the bf16 cast of these rows and ran the "
                "multiply-add chain in bf16",
        },
        {
            "hidden_states": x, "x_step": x_step,
            "q_pe": cap["q_pe"],
            "query_states": cap["query_states"],
            "query_states_step": cap_step["query_states"],
            "latent_normed": cap["latent_normed"],
            "latent_normed_step": cap_step["latent_normed"],
            "key_states": cap["key_states"],
            "value_states": cap["value_states"],
            "key_states_step": cap_step["key_states"],
            "value_states_step": cap_step["value_states"],
            "latent_cached": cap_step["latent_cached"],
            "kpe_cached": cap_step["kpe_cached"],
            "kpe_cached_interleaved": interleave_pairs(cap_step["kpe_cached"]),
            "attn_output": cap["attn_output"],
            "output": cap["output"],
            "output_step": cap_step["output"],
            "cos": cap["cos"], "sin": cap["sin"],
            "cos_step": cap_step["cos"], "sin_step": cap_step["sin"],
            "positions": torch.tensor([0, 1], dtype=torch.long, device=DEVICE),
        },
    )

    # Case 02 prefill 3 then a 3-step decode sequence, positions 3, 4, 5.
    torch.manual_seed(SEED_ATTN + 2)
    x = torch.randn(1, 3, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    x_steps = [torch.randn(1, 1, HIDDEN, dtype=torch.bfloat16, device=DEVICE) for _ in range(3)]
    cap = run_pass(attn, x, [0, 1, 2], [])
    cache_layers = [(cap["latent_cached"], cap["kpe_cached"])]
    tensors = {
        "hidden_states": x,
        "latent_normed": cap["latent_normed"],
        "latent_cached": cap["latent_cached"],
        "kpe_cached": cap["kpe_cached"],
        "cos": cap["cos"], "sin": cap["sin"],
        "output": cap["output"],
        "positions": torch.tensor([0, 1, 2], dtype=torch.long, device=DEVICE),
    }
    step_meta_positions = []
    for i, x_step in enumerate(x_steps):
        position = 3 + i
        step_meta_positions.append(position)
        cap_step = run_pass(attn, x_step, [position], cache_layers)
        cache_layers = [(cap_step["latent_cached"], cap_step["kpe_cached"])]
        tensors[f"x_step{i}"] = x_step
        tensors[f"cos_step{i}"] = cap_step["cos"]
        tensors[f"sin_step{i}"] = cap_step["sin"]
        tensors[f"latent_normed_step{i}"] = cap_step["latent_normed"]
        tensors[f"key_states_step{i}"] = cap_step["key_states"]
        tensors[f"output_step{i}"] = cap_step["output"]
    tensors["latent_cached"] = cap_step["latent_cached"]
    tensors["kpe_cached"] = cap_step["kpe_cached"]
    tensors["kpe_cached_interleaved"] = interleave_pairs(cap_step["kpe_cached"])
    tensors["positions_steps"] = torch.tensor(step_meta_positions, dtype=torch.long, device=DEVICE)
    save_fixture(
        layer_name, 2,
        {
            **COMMON_META,
            "case": "prefill3_decode3_steps",
            "positions": [0, 1, 2],
            "positions_steps": step_meta_positions,
            "seed": SEED_ATTN + 2,
            "softmax_scaling": SOFTMAX_SCALING,
            "rope_reference_cast": "cos/sin recorded f32 pre-cast, the reference "
                "rotation consumed the bf16 cast of these rows and ran the "
                "multiply-add chain in bf16",
        },
        tensors,
    )
    print(f"Generated {layer_name} fixtures")


def main() -> None:
    """Records the layer-0 MLA fixtures of the checkpoint."""
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

    cfg = load_config()
    weights = load_layer0_weights()
    attn = build_attention(weights, cfg)
    rotary = Glm4MoeLiteRotaryEmbedding(cfg)
    attn = attn.to(DEVICE)
    rotary = rotary.to(DEVICE)
    assert cfg.rope_interleave, "the recorded rotation is the interleaved spelling"
    assert rotary.attention_scaling == 1.0
    assert attn.q_a_layernorm.variance_epsilon == 1e-6, (
        "the reference query bottleneck norm runs the module default eps 1e-6, "
        "config.rms_norm_eps never reaches it")
    assert attn.kv_a_layernorm.variance_epsilon == 1e-6, (
        "the reference latent norm runs the module default eps 1e-6, "
        "config.rms_norm_eps never reaches it")
    assert attn.scaling == SOFTMAX_SCALING
    assert cfg.q_lora_rank == Q_LORA_RANK
    assert cfg.v_head_dim == V_HEAD_DIM
    assert cfg.partial_rotary_factor == 1.0

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_tables_fixtures(rotary)
    generate_norm_fixtures(attn.kv_a_layernorm)
    generate_qchain_fixtures(attn)
    generate_rope_fixtures(rotary)
    generate_attn_fixtures(attn, rotary)



    print(f"[gen_bf16_glm47flash_01_layer_internals_mla] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_glm47flash_01_layer_internals_mla] wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
