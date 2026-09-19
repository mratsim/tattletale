#!/usr/bin/env python3
"""Layer-0 fixture file of the Moonlight-16B-A3B checkpoint, recorded on CPU
torch bf16 with the installed reference modeling, from safetensors.

Single-file grammar with one fixture file per family layer, one bare
bf16 driving tensor per mixture in the payload, every recorded
intermediate and output a stats fingerprint.

The layer-0 file carries three mixture groups under first_k_dense_replace:

- attn, the attention mixer op surface on the prefill path
- layer, the full dense decoder layer 0 chain over one recorded hidden input
- moe, the routed block surface of the first routed layer

- one bare bf16 driving tensor per mixture, attn.input, layer.layer_input
  and moe.h, the only tensors the file stores
- the rope rows and the recorded sdpa inputs stay on the stats frame,
  fingerprints only, no raw op-surface tensor ships in the file

- the checkpoint gate weight stays on the stats frame as a loader
  cross-check, the bf16 router rows compared row for row

No Qwen3 analog exists, the Qwen3 and Qwen3.5 families run plain multi-head
attention with neither a compressed-Q bottleneck nor a noaux_tc router.

Mixture payloads and stats entries:

| mixture | payload           | stats entries over the recorded surface                                                                                               |
| ------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| attn    | attn.input        | cos, sin, query_states, key_states, value_states, q_pe, latent_normed, latent_cached, kpe_cached, kpe_cached_interleaved, attn_output |
| layer   | layer.layer_input | input_layernorm_output, attn_output, post_attention_layernorm_output, mlp_output, layer_output                                        |
| moe     | moe.h             | router_logits, topk_weights, moe_output, bias, gate_weight                                                                            |

Stats keys carry the mixture-level `attn.` / `layer.` / `moe.` prefixes,
recorded expert ids live in the metadata (integer ids carry no stats record).

Mixture seeds and scenarios:

| mixture | scenario                                                       | seed         |
| ------- | -------------------------------------------------------------- | ------------ |
| attn    | latent-cache attention prefill at seq 8, positions 0 through 7 | 224          |
| layer   | dense layer-0 chain over one recorded hidden input, seq 4      | 229          |
| moe     | routed block of layer 1 on a margin-clean input, seq 6         | 227 + search |
- the routed-block seed search advances one seed at a time until the top-k
  boundary margin clears the 1e-4 sigmoid-plus-bias choice floor, n_group 1
  leaves no group boundary to check
- the rope rows are recorded f32 pre-cast, the reference rotation consumed
  the bf16 cast of these rows and ran the multiply-add chain in bf16
- the rotated q channels of query_states sit in the reference half-split
  layout (cat of even and odd pair values), matching the recorded sdpa input

Recording environment:

- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment
  variable overrides it
- recording runs on cpu, the replay device stays the consuming suite's call

Run from the worktree root, twice, cmp proves byte determinism:

  uv run python workspace/transformers/tests/testgen/gen_bf16_moonlight_01_layer_internals.py

RAM guards:

- the expert-stack build clones the layer-1 routed weights, the script
  refuses to run when the free+inactive+speculative pool sits below the floor
- another python/torch process holding RAM also blocks the run
- the process chain of this script stays excluded from that check"""

from collections import OrderedDict
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors import torch as st  # noqa: E402

import transformers  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (  # noqa: E402
    DeepseekV3Config,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: E402
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3MoE,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb_interleave,
)

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Checkpoint, fixture and config paths.
MODEL_NAME = "Moonlight-16B-A3B"
LAYER_IDX = 0
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals", f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
FIXTURE_STEM = f"layer{LAYER_IDX}-{MODEL_NAME}-00"
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


# Checkpoint geometry, parsed once from the config (config is king).
# The mixture generators read these parsed rows without hardcoded
# copies of any checkpoint width, head count or routing constant.
with open(CONFIG_PATH) as _f:
    _CKPT_CONFIG = json.load(_f)


class _Geometry:
    """Namespace over the parsed checkpoint config rows the mixtures read."""


GEOMETRY = _Geometry()
GEOMETRY.hidden_size = _CKPT_CONFIG["hidden_size"]
GEOMETRY.num_attention_heads = _CKPT_CONFIG["num_attention_heads"]
GEOMETRY.kv_lora_rank = _CKPT_CONFIG["kv_lora_rank"]
GEOMETRY.qk_nope_head_dim = _CKPT_CONFIG["qk_nope_head_dim"]
GEOMETRY.qk_rope_head_dim = _CKPT_CONFIG["qk_rope_head_dim"]
GEOMETRY.v_head_dim = _CKPT_CONFIG["v_head_dim"]
GEOMETRY.rope_theta = _CKPT_CONFIG["rope_theta"]
GEOMETRY.n_routed_experts = _CKPT_CONFIG["n_routed_experts"]
GEOMETRY.num_experts_per_tok = _CKPT_CONFIG["num_experts_per_tok"]
GEOMETRY.n_group = _CKPT_CONFIG["n_group"]
GEOMETRY.topk_group = _CKPT_CONFIG["topk_group"]
GEOMETRY.routed_scaling_factor = _CKPT_CONFIG["routed_scaling_factor"]
GEOMETRY.norm_topk_prob = _CKPT_CONFIG["norm_topk_prob"]
GEOMETRY.n_shared_experts = _CKPT_CONFIG["n_shared_experts"]
GEOMETRY.first_k_dense_replace = _CKPT_CONFIG["first_k_dense_replace"]

NUM_THREADS = 1

# Per-mixture seeds, independent and order-agnostic. The routed-block seed
# search advances from SEED_MOE until the boundary margin clears the floor.
SEED_ATTN = 224
SEED_LAYER = 229
SEED_MOE = 227

# Top-k boundary floor of the routed-block seed search, the recorded floor
# of the exact expert-id comparisons in the consuming suite.
MARGIN_FLOOR = 1e-4
MAX_SEED_TRIES = 200

MIN_POOL_BYTES = 32 * 1024 ** 3

PREFIX = f"model.layers.{LAYER_IDX}.self_attn."
ATTN_TENSORS = (
    "q_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
)
LAYER0_TENSORS = [PREFIX + name for name in ATTN_TENSORS] + [
    f"model.layers.{LAYER_IDX}.input_layernorm.weight",
    f"model.layers.{LAYER_IDX}.post_attention_layernorm.weight",
    f"model.layers.{LAYER_IDX}.mlp.gate_proj.weight",
    f"model.layers.{LAYER_IDX}.mlp.up_proj.weight",
    f"model.layers.{LAYER_IDX}.mlp.down_proj.weight",
]
GATE_KEY = f"model.layers.{GEOMETRY.first_k_dense_replace}.mlp.gate.weight"
BIAS_KEY = f"model.layers.{GEOMETRY.first_k_dense_replace}.mlp.gate.e_score_correction_bias"


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_moonlight_01_layer_internals] vm_stat gave no page size line")


def pool_bytes() -> int:
    """Free+inactive+speculative physical memory in bytes from `vm_stat`."""
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
            "[gen_bf16_moonlight_01_layer_internals] vm_stat gave no pool lines")
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
    """Refuse to load weights under low memory or a stray python/torch process, the pgrep match excludes this process chain."""
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_moonlight_01_layer_internals] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_moonlight_01_layer_internals] other python/torch processes hold RAM: {stray}. "
            "Stop and retry when idle")


def load_config() -> DeepseekV3Config:
    """Load the checkpoint config.json (flat DeepseekV3 layout). The rope_parameters default lands on the plain theta spelling:
      rope_type default, theta 50000, rope_interleave true."""
    with open(CONFIG_PATH) as f:
        cfg = DeepseekV3Config.from_dict(json.load(f))
    cfg._attn_implementation = "sdpa"
    return cfg


def shard_of(key: str) -> str:
    """Shard filename holding one checkpoint tensor, read off the model.safetensors.index.json weight map."""
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    if key not in weight_map:
        raise SystemExit(
            f"[gen_bf16_moonlight_01_layer_internals] key {key} missing from the weight map")
    return weight_map[key]


def load_checkpoint_tensors(keys: list) -> dict:
    """Load the named checkpoint tensors from the shards that hold them, via safe_open (memory-mapped, only these tensors are copied)."""
    weights = {}
    shards = {key: shard_of(key) for key in keys}
    for key, shard in shards.items():
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            weights[key] = f.get_tensor(key).clone()
    return weights


def build_decoder_layer(weights: dict, cfg: DeepseekV3Config) -> DeepseekV3DecoderLayer:
    """DeepseekV3DecoderLayer with the real dense layer-0 weights loaded.

    Args:
    - weights, cfg, the checkpoint tensors and the parsed config

    Returns:
    - the decoder layer in eval mode, its attention feeds the attn mixture,
      the full chain feeds the layer mixture, the dense layer-0 SwiGLU block
      sits under first_k_dense_replace
    """
    layer = DeepseekV3DecoderLayer(cfg, layer_idx=LAYER_IDX)
    attn = layer.self_attn
    with torch.no_grad():
        for name in ATTN_TENSORS:
            module = {
                "q_proj.weight": attn.q_proj,
                "kv_a_proj_with_mqa.weight": attn.kv_a_proj_with_mqa,
                "kv_a_layernorm.weight": attn.kv_a_layernorm,
                "kv_b_proj.weight": attn.kv_b_proj,
                "o_proj.weight": attn.o_proj,
            }[name]
            module.weight.data = weights[PREFIX + name]
        layer.input_layernorm.weight.data = \
            weights[f"model.layers.{LAYER_IDX}.input_layernorm.weight"]
        layer.post_attention_layernorm.weight.data = \
            weights[f"model.layers.{LAYER_IDX}.post_attention_layernorm.weight"]
        layer.mlp.gate_proj.weight.data = \
            weights[f"model.layers.{LAYER_IDX}.mlp.gate_proj.weight"]
        layer.mlp.up_proj.weight.data = \
            weights[f"model.layers.{LAYER_IDX}.mlp.up_proj.weight"]
        layer.mlp.down_proj.weight.data = \
            weights[f"model.layers.{LAYER_IDX}.mlp.down_proj.weight"]
    layer.eval()
    return layer


def build_routed_block(weights: dict, cfg: DeepseekV3Config) -> DeepseekV3MoE:
    """DeepseekV3MoE with the real first-routed-layer weights loaded.

    Args:
    - weights, cfg, the checkpoint tensors (layer 1 mlp rows) and the config

    Returns:
    - the routed block in eval mode, the experts stacked from the per-expert
      checkpoint rows in the chunked module layout
    - the router module op-identical to the checkpoint's own noaux_tc router
    """
    moe_layer_idx = GEOMETRY.first_k_dense_replace
    prefix = f"model.layers.{moe_layer_idx}.mlp."
    moe = DeepseekV3MoE(cfg)
    moe.gate = build_router(weights)
    with torch.no_grad():
        moe.experts.gate_up_proj.data = torch.stack([
            torch.cat([weights[f"{prefix}experts.{e}.gate_proj.weight"],
                       weights[f"{prefix}experts.{e}.up_proj.weight"]], dim=0)
            for e in range(GEOMETRY.n_routed_experts)])
        moe.experts.down_proj.data = torch.stack([
            weights[f"{prefix}experts.{e}.down_proj.weight"]
            for e in range(GEOMETRY.n_routed_experts)])
        moe.shared_experts.gate_proj.weight.data = \
            weights[prefix + "shared_experts.gate_proj.weight"]
        moe.shared_experts.up_proj.weight.data = \
            weights[prefix + "shared_experts.up_proj.weight"]
        moe.shared_experts.down_proj.weight.data = \
            weights[prefix + "shared_experts.down_proj.weight"]
    moe.eval()
    return moe


def build_router(weights: dict):
    """Reference noaux_tc router module, the real layer-1 router weight
    plus bias buffer assigned, eval mode.

    - DeepseekV3TopkRouter stays the noaux_tc authority
    - op-identical to the checkpoint's own router module
    - selection-only bias, the f32 renorm keeps the 1e-20 denominator"""
    cfg = DeepseekV3Config(
        num_local_experts=GEOMETRY.n_routed_experts,
        num_experts_per_tok=GEOMETRY.num_experts_per_tok,
        hidden_size=GEOMETRY.hidden_size,
        routed_scaling_factor=GEOMETRY.routed_scaling_factor,
        n_group=GEOMETRY.n_group,
        topk_group=GEOMETRY.topk_group,
        norm_topk_prob=GEOMETRY.norm_topk_prob,
    )
    router = DeepseekV3TopkRouter(cfg)
    router.weight.data = weights[GATE_KEY]
    router.e_score_correction_bias.data = (
        weights[BIAS_KEY].to(torch.float32)
    )
    router.eval()
    return router


def precast_table_rows(rotary: DeepseekV3RotaryEmbedding,
                       positions: list[int]) -> tuple:
    """cos/sin rows (len(positions), qk_rope_head_dim/2) f32 pre-cast from the reference rotary, attention_scaling already applied."""
    x = torch.zeros(1, len(positions), GEOMETRY.hidden_size, dtype=torch.float32)
    pos = torch.tensor(positions, dtype=torch.long).reshape(1, len(positions))
    cos, sin = rotary(x, pos)
    half = GEOMETRY.qk_rope_head_dim // 2
    return cos[0, :, :half].contiguous(), sin[0, :, :half].contiguous()


def bf16_cos_sin(rotary: DeepseekV3RotaryEmbedding,
                 positions: list[int]) -> tuple:
    """cos/sin (1, seq, qk_rope_head_dim) at the hidden dtype, exactly the position embeddings the reference attention consumes."""
    x = torch.zeros(1, len(positions), GEOMETRY.hidden_size, dtype=torch.bfloat16)
    pos = torch.tensor(positions, dtype=torch.long).reshape(1, len(positions))
    cos, sin = rotary(x, pos)
    return cos, sin


def interleave_pairs(half_split: torch.Tensor) -> torch.Tensor:
    """Pure data movement reorder, the reference half-split rotation output permuted into the interleaved pair layout."""
    b, h, s, d = half_split.shape
    even = half_split[..., :d // 2]
    odd = half_split[..., d // 2:]
    return torch.stack([even, odd], dim=-1).reshape(b, h, s, d)


def attention_forward_capture(attn, hidden_states, cos, sin, cache):
    """Replay of the reference DeepseekV3Attention.forward with intermediate capture, asserted equal to the module's own forward before saving."""
    batch_size, seq_length = hidden_states.shape[:-1]
    q = attn.q_proj(hidden_states)
    q = q.view(batch_size, seq_length, -1, attn.qk_head_dim).transpose(1, 2)
    q_pass, q_pe_raw = torch.split(
        q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

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


def fresh_cache(cache_layers):
    """DynamicCache preloaded with the (latent, kpe) pairs of the prior passes."""
    cache = DynamicCache()
    for i, (lat, kpe) in enumerate(cache_layers):
        cache.update(lat, kpe, i)
    return cache


def run_pass(attn, rotary, x, positions, cache_layers):
    """One attention pass over a fresh cache preloaded with the prior passes' cached tensors.

    Args:
    - attn, rotary, the weighted attention and the reference rotary
    - x, positions, the bf16 hidden states and their positions
    - cache_layers, the (latent, kpe) pairs each prior pass contributed

    Returns:
    - the capture dict of attention_forward_capture with the f32
      pre-cast cos/sin rows added, the replay output equals the module
      forward over one fresh cache (torch.equal)
    """
    # Each run gets a fresh cache preloaded with the prior passes' cached tensors,
    # so neither run advances the state the other reads.
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


def topk_boundary_margin(choice: torch.Tensor, top_k: int) -> float:
    """Smallest gap between the Kth and the K+1th sorted choice score, minimized over all rows, the exact-index condition of the top-k selection."""
    sorted_choice = choice.sort(dim=-1, descending=True).values
    gaps = sorted_choice[:, top_k - 1] - sorted_choice[:, top_k]
    return gaps.min().item()


def generate_attn_mixture(layer, rotary) -> tuple:
    """Latent-cache attention prefill at seq 8, positions 0..7, the cache write of the full pass recorded.

    Args:
    - layer, rotary, the weighted decoder layer (its attention replays)
      plus the reference rotary

    Returns:
    - meta, the mixture metadata
    - payload, the driving input, the rope rows and the recorded sdpa inputs
    - captured, the recorded per-op intermediates for the stats frame
    """
    attn = layer.self_attn
    torch.manual_seed(SEED_ATTN)
    x = torch.randn(1, 8, GEOMETRY.hidden_size, dtype=torch.bfloat16)
    cap = run_pass(attn, rotary, x, list(range(8)), [])
    meta = {
        "case": "prefill_seq8",
        "seq_len": 8,
        "positions": list(range(8)),
        "seed": SEED_ATTN,
        "softmax_scaling": float(attn.scaling),
        "rope_reference_cast": "cos/sin recorded f32 pre-cast, the reference "
            "rotation consumed the bf16 cast of these rows and ran the "
            "multiply-add chain in bf16",
        "rope_output_layout": "the rotated q channels of query_states sit "
            "in the reference half-split layout (cat of even and odd pair "
            "values), the layout the recorded sdpa consumed",
    }
    payload = OrderedDict([("attn.input", x)])
    captured = {
        "attn.cos": cap["cos"], "attn.sin": cap["sin"],
        "attn.query_states": cap["query_states"],
        "attn.key_states": cap["key_states"],
        "attn.value_states": cap["value_states"],
        "attn.q_pe": cap["q_pe"],
        "attn.latent_normed": cap["latent_normed"],
        "attn.latent_cached": cap["latent_cached"],
        "attn.kpe_cached": cap["kpe_cached"],
        "attn.kpe_cached_interleaved": interleave_pairs(cap["kpe_cached"]),
        "attn.attn_output": cap["attn_output"],
    }
    print("[gen_bf16_moonlight_01_layer_internals] attn mixture")
    return meta, payload, captured


def generate_layer_mixture(layer, rotary) -> tuple:
    """Full dense decoder layer 0 chain over one recorded hidden input,
    input layernorm through the residual add after the dense SwiGLU block.

    Args:
    - layer, rotary, the weighted decoder layer and the reference rotary

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its segment name
    - captured, the captured chain tensors for the stats frame

    The manual chain is the recording form, bitwise-equal to the installed
    decoder layer forward before saving.
    """
    attn = layer.self_attn
    seq_len = 4
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED_LAYER)
    x = torch.randn(1, seq_len, GEOMETRY.hidden_size, generator=gen,
                    dtype=torch.bfloat16)
    cos, sin = bf16_cos_sin(rotary, list(range(seq_len)))

    with torch.no_grad():
        h_norm = layer.input_layernorm(x)
        cap = attention_forward_capture(attn, h_norm, cos, sin,
                                        fresh_cache([]))
        attn_out = cap["output"]
        h1 = x + attn_out
        h2 = layer.post_attention_layernorm(h1)
        mlp_out = layer.mlp(h2)
        layer_out = h1 + mlp_out
        module_out = layer(
            x, attention_mask=None, position_ids=None, past_key_values=None,
            use_cache=False, position_embeddings=(cos, sin))
    assert torch.equal(module_out, layer_out), (
        "manual chain diverged from the decoder layer forward")

    meta = {
        "case": "prefill_seq4",
        "seq_len": seq_len,
        "seed": SEED_LAYER,
        "layer": f"model.layers.{LAYER_IDX}",
        "chain": "dense layer-0 block under first_k_dense_replace, "
            "the attention mixer plus the dense SwiGLU block",
    }
    payload = OrderedDict([("layer.layer_input", x)])
    captured = {
        "layer.input_layernorm_output": h_norm,
        "layer.attn_output": attn_out,
        "layer.post_attention_layernorm_output": h2,
        "layer.mlp_output": mlp_out,
        "layer.layer_output": layer_out,
    }
    print("[gen_bf16_moonlight_01_layer_internals] layer mixture "
          f"(seed {SEED_LAYER})")
    return meta, payload, captured


def generate_moe_mixture(moe, weights) -> tuple:
    """Routed block of the first routed layer on a margin-clean hidden input.

    Args:
    - moe, weights, the weighted layer-1 routed block and the checkpoint
      tensors (the gate weight and bias fingerprints)

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its segment name
    - captured, the recorded decisions, block output and the loader
      cross-check fingerprints for the stats frame

    The seed search advances one seed at a time until the top-k boundary
    margin clears the recorded floor protecting the exact expert-id
    comparisons in the consuming suite.
    """
    top_k = GEOMETRY.num_experts_per_tok
    bias = moe.gate.e_score_correction_bias.detach()
    seed = SEED_MOE
    margin = -1.0
    for _ in range(MAX_SEED_TRIES):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        h = torch.randn(1, 6, GEOMETRY.hidden_size, generator=gen,
                        dtype=torch.bfloat16)
        with torch.no_grad():
            router_logits, topk_weights, topk_indices = moe.gate(h)
        choice = router_logits.sigmoid() + bias.unsqueeze(0)
        margin = topk_boundary_margin(choice, top_k)
        if margin > MARGIN_FLOOR:
            break
        seed += 1
    else:
        raise SystemExit(
            "[gen_bf16_moonlight_01_layer_internals] no margin-clean seed "
            "found for the routed block")

    # Self-check before saving, the renormalized rows sum to the routed
    # scaling factor and the weights stay the unbiased gathered scores
    # (selection-only bias), each within the f32 renorm rounding slack.
    gathered = router_logits.sigmoid().gather(1, topk_indices)
    expected = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20) \
        * GEOMETRY.routed_scaling_factor
    row_sums = topk_weights.sum(dim=-1)
    drift = (row_sums - GEOMETRY.routed_scaling_factor).abs().max().item()
    assert drift < 1e-6, f"renorm sum drift {drift}"
    assert torch.allclose(topk_weights, expected, rtol=0.0, atol=1e-6), \
        "weights diverge from the unbiased gathered scores"

    with torch.no_grad():
        moe_output = moe(h)

    meta = {
        "case": "routed_block_seq6_margin_clean",
        "seq_len": 6,
        "seed": seed,
        "layer": f"model.layers.{GEOMETRY.first_k_dense_replace}.mlp",
        "router": "noaux_tc",
        "num_prefill_tokens": 6,
        "margin_floor": MARGIN_FLOOR,
        "topk_boundary_margin": margin,
        "gate_weight_key": GATE_KEY,
        "gate_weight_dtype": str(weights[GATE_KEY].dtype).replace("torch.", ""),
        "hidden_dtype": "bfloat16",
        "logits_dtype": "float32",
        "weights_dtype": "float32",
        "indices_dtype": "int64",
        "bias_key": BIAS_KEY,
        "bias_dtype": "float32",
        "bias_checkpoint_dtype": str(weights[BIAS_KEY].dtype).replace("torch.", ""),
        "topk_indices": topk_indices.tolist(),
    }
    payload = OrderedDict([("moe.h", h)])
    captured = {
        "moe.router_logits": router_logits,
        "moe.topk_weights": topk_weights,
        "moe.moe_output": moe_output,
        "moe.bias": weights[BIAS_KEY].to(torch.float32),
        # Loader cross-check fingerprint, the suite compares the checkpoint
        # gate weight rows against this record.
        "moe.gate_weight": weights[GATE_KEY],
    }
    print(f"[gen_bf16_moonlight_01_layer_internals] moe mixture (seed {seed}): "
          f"topk margin {margin:.3e}")
    return meta, payload, captured


def save_fixture(metadata: dict, mixtures: list) -> None:
    """Writes the single-file fixture set, one safetensors payload carrying
    the three bare named driving tensors, one metadata sidecar and one
    stats sidecar with keys namespaced by mixture.

    Args:
    - metadata, the merged metadata frame
    - mixtures, one (payload, captured) pair per mixture, the payload tensor
      carrying its file name and the captured tensors already carrying
      their namespaced stats keys
    """
    file_tensors = OrderedDict()
    stats_entries = []
    for payload, captured in mixtures:
        for name, tensor in payload.items():
            file_tensors[name] = tensor.detach().cpu().contiguous()
            stats_entries.append((name, file_tensors[name]))
        for name, tensor in captured.items():
            stats_entries.append((name, tensor.detach().cpu().contiguous()))

    serialized = st.save(file_tensors, metadata=None)
    with open(FIXTURE_PATH, "wb") as f:
        f.write(serialized)

    import compression.zstd

    zstd_options = {
        compression.zstd.CompressionParameter.compression_level: 19,
        compression.zstd.CompressionParameter.content_size_flag: 1,
        compression.zstd.CompressionParameter.checksum_flag: 1,
    }
    payload_json = json.dumps(
        metadata, sort_keys=True, indent=2, ensure_ascii=True
    ).encode("utf-8") + b"\n"
    with open(FIXTURE_PATH + ".metadata.json.zst", "wb") as f:
        f.write(compression.zstd.compress(payload_json, options=zstd_options))
    write_stats_file(FIXTURE_PATH + ".stats.json.zst", FIXTURE_STEM + ".safetensor",
                     stats_entries)


def main() -> None:
    """Record the layer-0 fixture file set after the RAM guard."""
    recorded_from = os.environ.get("TTT_RECORD_FROM", "m4max-cpu")
    check_ram()

    cfg = load_config()
    moe_layer_idx = GEOMETRY.first_k_dense_replace
    expert_keys = [f"model.layers.{moe_layer_idx}.mlp.experts.{e}.{part}"
                   for e in range(GEOMETRY.n_routed_experts)
                   for part in ["gate_proj.weight", "up_proj.weight",
                                "down_proj.weight"]]
    weights = load_checkpoint_tensors(
        LAYER0_TENSORS
        + [GATE_KEY, BIAS_KEY,
           f"model.layers.{moe_layer_idx}.mlp.shared_experts.gate_proj.weight",
           f"model.layers.{moe_layer_idx}.mlp.shared_experts.up_proj.weight",
           f"model.layers.{moe_layer_idx}.mlp.shared_experts.down_proj.weight"]
        + expert_keys)
    assert cfg.rope_interleave, "the recorded rotation is the interleaved spelling"

    rotary = DeepseekV3RotaryEmbedding(cfg)
    assert rotary.attention_scaling == 1.0
    layer = build_decoder_layer(weights, cfg)
    assert layer.self_attn.kv_a_layernorm.variance_epsilon == 1e-6, (
        "the reference latent norm runs the module default eps 1e-6, "
        "config.rms_norm_eps never reaches it")
    assert layer.self_attn.scaling == 0.07216878364870322
    moe = build_routed_block(weights, cfg)

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    attn_meta, attn_payload, attn_captured = generate_attn_mixture(layer, rotary)
    layer_meta, layer_payload, layer_captured = generate_layer_mixture(layer, rotary)
    moe_meta, moe_payload, moe_captured = generate_moe_mixture(moe, weights)

    metadata = {
        "model": MODEL_NAME,
        "file": FIXTURE_STEM + ".safetensor",
        "dtype": "bfloat16",
        "num_threads": NUM_THREADS,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "recorded_from": recorded_from,
        "device": "cpu",
        "layer": PREFIX,
        "num_heads": GEOMETRY.num_attention_heads,
        "kv_lora_rank": GEOMETRY.kv_lora_rank,
        "qk_nope_head_dim": GEOMETRY.qk_nope_head_dim,
        "qk_rope_head_dim": GEOMETRY.qk_rope_head_dim,
        "v_head_dim": GEOMETRY.v_head_dim,
        "hidden_size": GEOMETRY.hidden_size,
        "rope_theta": GEOMETRY.rope_theta,
        "rope_type": "default",
        "rope_interleave": True,
        "softmax_scaling": float(layer.self_attn.scaling),
        "moe_layer_index": GEOMETRY.first_k_dense_replace,
        "num_experts": GEOMETRY.n_routed_experts,
        "num_experts_per_tok": GEOMETRY.num_experts_per_tok,
        "n_group": GEOMETRY.n_group,
        "topk_group": GEOMETRY.topk_group,
        "routed_scaling_factor": GEOMETRY.routed_scaling_factor,
        "norm_topk_prob": GEOMETRY.norm_topk_prob,
        "shared_experts": GEOMETRY.n_shared_experts,
        "flip_budget_policy": "exact expert-id comparisons in the consuming "
            "suites stand on a recorded positive boundary margin with the "
            "1e-4 floor, a row below the floor would carry an explicit "
            "per-mixture recorded exception, never a silent widening",
        "margin_floor": 1e-4,
        "mixtures": {
            "attn": attn_meta,
            "layer": layer_meta,
            "moe": moe_meta,
        },
    }
    save_fixture(metadata, [
        (attn_payload, attn_captured),
        (layer_payload, layer_captured),
        (moe_payload, moe_captured),
    ])

    print(f"[gen_bf16_moonlight_01_layer_internals] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_moonlight_01_layer_internals] wrote {FIXTURE_PATH}")

if __name__ == "__main__":
    main()
