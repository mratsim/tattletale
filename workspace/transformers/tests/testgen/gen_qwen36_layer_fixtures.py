#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B full decoder-layer fixtures (layer 0 GDN +
MoE, layer 3 gated full attention + MoE) from the real checkpoint shards,
using the vendored reference modeling on CPU torch bf16.

What is generated (under tests/fixtures/qwen36-layer/):

  layer-Qwen3.6-35B-A3B-00.safetensor (+ .metadata.json)
    Full decoder layer 0 (linear_attention + routed MoE), prefill T=6:
    layer input, input layernorm output, the GDN block output under both
    core rules, post-attention layernorm output, the routed-block
    intermediates (router logits, top-k indices, routing weights, shared
    gate, routed-block output) and the layer outputs under both rules.
  layer-Qwen3.6-35B-A3B-03.safetensor (+ .metadata.json)
    Full decoder layer 3 (full_attention + routed MoE), prefill T=6 at
    positions 0..5: layer input, positions, cos/sin, input layernorm
    output, the gated attention output after o_proj, the post-attention
    layernorm output, the routed-block intermediates and the layer output.

The layer chain is the vendored Qwen3_5MoeDecoderLayer.forward: local
residuals, input layernorm, token mixer, residual, post-attention
layernorm, routed block, residual. The GDN chunked rule is the vendored
forward. The recurrent rule is the bitwise reference for the Nim block
(the Nim recurrence is bit-identical to torch_recurrent_gated_delta_rule).
Weights are routed through model.safetensors.index.json, opening only the
shards that hold the requested layer.

Run (twice; cmp proves byte determinism):
  cd <worktree root> && uv run --no-project --python 3.12 \
    --with "transformers @ file://<root>/_references_prod/transformers" \
    --with torch \
    workspace/transformers/tests/testgen/gen_qwen36_layer_fixtures.py

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
        f"[gen_qwen36_layer_fixtures] vendored modeling not found above {_here}. "
        "Set QWEN36_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, VENDORED_SRC)

import transformers  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    apply_rotary_pos_emb,
    repeat_kv,
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
FIXTURE_DIR = os.path.join(GRANDPARENT_DIR, "fixtures", "qwen36-layer")
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# The vendored tree sha this fixture was generated against.
VENDORED_SHA = "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701"
NUM_THREADS = 1

# Per-generator seeds, independent and order-agnostic. The seed search
# rejects a seed whose top-k margins tie.
SEED_LAYER0_BASE = 211
SEED_LAYER3_BASE = 307

# Geometry of the checkpoint.
HIDDEN = 2048
SEQ_LEN = 6
CHUNK_SIZE = 64
NUM_EXPERTS = 256
TOP_K = 8

PREFIX_FMT = "model.language_model.layers.{layer}."
MIN_FREE_BYTES = 8 * 1024 ** 3

# The GDN chunked-vs-recurrent block bar, the same bar the GDN suite
# asserts against the vendored chunked form.
BLOCK_BAR = 1e-3
# The gated attention block bar after o_proj, the same bar the attention
# suite asserts on the identical quantity (cross-version SDPA noise).
ATTN_BAR = 5e-3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_qwen36_layer_fixtures] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_qwen36_layer_fixtures] vm_stat gave no 'Pages free' line")


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
            f"[gen_qwen36_layer_fixtures] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_qwen36_layer_fixtures] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def check_vendored_sha() -> str:
    """Record the vendored tree sha and fail loudly on drift."""
    out = subprocess.run(
        ["git", "-C", VENDORED_SRC, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    sha = out.stdout.strip()
    if sha != VENDORED_SHA:
        raise SystemExit(
            f"[gen_qwen36_layer_fixtures] vendored tree sha {sha} differs from "
            f"{VENDORED_SHA}; the fixture truth moved")
    return sha


def load_text_config() -> Qwen3_5MoeTextConfig:
    """Load the nested text_config from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeTextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


def load_index_weight_map() -> dict:
    """Parse the checkpoint index and return its weight_map."""
    with open(INDEX_PATH) as f:
        index = json.load(f)
    return index["weight_map"]


def layer_shorts(cfg: Qwen3_5MoeTextConfig, layer_idx: int) -> list:
    """Tensor name suffixes of one decoder layer: the token mixer keys of
    its kind plus the routed block and the two layernorms."""
    kind = cfg.layer_types[layer_idx]
    if kind == "linear_attention":
        mixer = [
            "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
            "linear_attn.conv1d.weight", "linear_attn.A_log",
            "linear_attn.dt_bias", "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
        ]
    else:
        mixer = [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        ]
    return mixer + [
        "mlp.gate.weight", "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
    ]


def load_layer_weights(weight_map: dict, cfg: Qwen3_5MoeTextConfig,
                       layer_idx: int) -> dict:
    """Load one decoder layer's tensors through the checkpoint index,
    opening only the shards that hold them (safe_open, memory mapped,
    only these tensors are copied)."""
    base = PREFIX_FMT.format(layer=layer_idx)
    by_shard = OrderedDict()
    for short in layer_shorts(cfg, layer_idx):
        key = base + short
        by_shard.setdefault(weight_map[key], []).append(key)
    weights = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            for key in keys:
                weights[key[len(base):]] = f.get_tensor(key).clone()
    return weights


def build_moe(cfg: Qwen3_5MoeTextConfig, weights: dict) -> Qwen3_5MoeSparseMoeBlock:
    """Routed block with real weights: router, fused experts, shared expert."""
    moe = Qwen3_5MoeSparseMoeBlock(cfg)
    with torch.no_grad():
        moe.gate.weight.data = weights["mlp.gate.weight"]
        moe.experts.gate_up_proj.data = weights["mlp.experts.gate_up_proj"]
        moe.experts.down_proj.data = weights["mlp.experts.down_proj"]
        moe.shared_expert.gate_proj.weight.data = \
            weights["mlp.shared_expert.gate_proj.weight"]
        moe.shared_expert.up_proj.weight.data = \
            weights["mlp.shared_expert.up_proj.weight"]
        moe.shared_expert.down_proj.weight.data = \
            weights["mlp.shared_expert.down_proj.weight"]
        moe.shared_expert_gate.weight.data = weights["mlp.shared_expert_gate.weight"]
    moe.eval()
    return moe


def build_decoder_layer(cfg: Qwen3_5MoeTextConfig, weights: dict,
                        layer_idx: int) -> Qwen3_5MoeDecoderLayer:
    """Full decoder layer with real weights, both layernorms, the token
    mixer of the layer kind and the routed block."""
    layer = Qwen3_5MoeDecoderLayer(cfg, layer_idx)
    with torch.no_grad():
        if hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn
            gdn.in_proj_qkv.weight.data = weights["linear_attn.in_proj_qkv.weight"]
            gdn.in_proj_z.weight.data = weights["linear_attn.in_proj_z.weight"]
            gdn.out_proj.weight.data = weights["linear_attn.out_proj.weight"]
            gdn.A_log.data = weights["linear_attn.A_log"].to(torch.bfloat16)
            gdn.conv1d.weight.data = weights["linear_attn.conv1d.weight"]
            gdn.dt_bias.data = weights["linear_attn.dt_bias"]
            gdn.in_proj_a.weight.data = weights["linear_attn.in_proj_a.weight"]
            gdn.in_proj_b.weight.data = weights["linear_attn.in_proj_b.weight"]
            gdn.norm.weight.data = weights["linear_attn.norm.weight"].to(torch.bfloat16)
        else:
            attn = layer.self_attn
            attn.q_proj.weight.data = weights["self_attn.q_proj.weight"]
            attn.k_proj.weight.data = weights["self_attn.k_proj.weight"]
            attn.v_proj.weight.data = weights["self_attn.v_proj.weight"]
            attn.o_proj.weight.data = weights["self_attn.o_proj.weight"]
            attn.q_norm.weight.data = weights["self_attn.q_norm.weight"].to(torch.bfloat16)
            attn.k_norm.weight.data = weights["self_attn.k_norm.weight"].to(torch.bfloat16)
        moe = layer.mlp
        moe.gate.weight.data = weights["mlp.gate.weight"]
        moe.experts.gate_up_proj.data = weights["mlp.experts.gate_up_proj"]
        moe.experts.down_proj.data = weights["mlp.experts.down_proj"]
        moe.shared_expert.gate_proj.weight.data = \
            weights["mlp.shared_expert.gate_proj.weight"]
        moe.shared_expert.up_proj.weight.data = \
            weights["mlp.shared_expert.up_proj.weight"]
        moe.shared_expert.down_proj.weight.data = \
            weights["mlp.shared_expert.down_proj.weight"]
        moe.shared_expert_gate.weight.data = weights["mlp.shared_expert_gate.weight"]
        layer.input_layernorm.weight.data = weights["input_layernorm.weight"]
        layer.post_attention_layernorm.weight.data = \
            weights["post_attention_layernorm.weight"]
    layer.eval()
    return layer


def gdn_forward_replay(block, hidden_states: torch.Tensor,
                       use_recurrent: bool) -> dict:
    """Replay of the reference Qwen3_5MoeGatedDeltaNet.forward with a
    selectable core rule, capturing every intermediate.

    The chunked replay must be bit-identical to the module forward
    (asserted by the caller). The recurrent replay is the bitwise
    reference for the Nim implementation."""
    batch_size, seq_len, _ = hidden_states.shape
    with torch.no_grad():
        mixed_qkv = block.in_proj_qkv(hidden_states).transpose(1, 2)
        z = block.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, block.head_v_dim)
        b = block.in_proj_b(hidden_states)
        a = block.in_proj_a(hidden_states)
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
        query_core, key_core = query, key
        if block.num_v_heads // block.num_k_heads > 1:
            ratio = block.num_v_heads // block.num_k_heads
            query_core = query.repeat_interleave(ratio, dim=2)
            key_core = key.repeat_interleave(ratio, dim=2)
        rule = (torch_recurrent_gated_delta_rule if use_recurrent
                else torch_chunk_gated_delta_rule)
        core_attn_out, _ = rule(
            query_core, key_core, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        core_2d = core_attn_out.reshape(-1, block.head_v_dim)
        z_2d = z.reshape(-1, block.head_v_dim)
        normed = block.norm(core_2d, z_2d).reshape(batch_size, seq_len, -1)
        output = block.out_proj(normed)
    return {"output": output}


def moe_forward_capture(moe: Qwen3_5MoeSparseMoeBlock,
                        hidden_states: torch.Tensor) -> dict:
    """Replay of the reference Qwen3_5MoeSparseMoeBlock.forward, capturing
    every intermediate. Must be bit-identical to the module forward
    (asserted by the caller)."""
    batch, seq, hidden_dim = hidden_states.shape
    with torch.no_grad():
        flat = hidden_states.view(-1, hidden_dim)
        shared_out = moe.shared_expert(flat)
        router_logits, routing_weights, selected_experts = moe.gate(flat)
        expert_out = moe.experts(flat, selected_experts, routing_weights)
        shared_gate = torch.sigmoid(moe.shared_expert_gate(flat))
        moe_out = expert_out + shared_gate * shared_out
    return {
        "router_logits": router_logits,
        "routing_weights": routing_weights,
        "topk_indices": selected_experts,
        "shared_gate": shared_gate,
        "moe_output": moe_out.reshape(batch, seq, hidden_dim),
    }


def attn_forward_capture(attn: Qwen3_5MoeAttention,
                         hidden_states: torch.Tensor,
                         position_embeddings) -> dict:
    """Replay of the reference Qwen3_5MoeAttention.forward, capturing the
    gated output and the o_proj output. Must be bit-identical to the
    module forward (asserted by the caller)."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    with torch.no_grad():
        query_states, gate = torch.chunk(
            attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2),
            2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(
            attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states_r = repeat_kv(key_states, attn.num_key_value_groups)
        value_states_r = repeat_kv(value_states, attn.num_key_value_groups)
        is_causal = query_states.shape[2] > 1 and attn.is_causal
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states_r, value_states_r,
            attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=attn.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output_gated = attn_output * torch.sigmoid(gate)
        output = attn.o_proj(attn_output_gated)
    return {"output": output, "attn_output_gated": attn_output_gated}


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute element difference of two tensors, compared in f32."""
    return (a.float() - b.float()).abs().max().item()


def ulp_bf16(m: float) -> float:
    """One bf16 ulp at magnitude m: bf16 stores 7 significand bits, so for
    m in [2**e, 2**(e+1)) the ulp is 2**(e-7). Zero maps to 0."""
    if m <= 0:
        return 0.0
    e = int(torch.floor(torch.log2(torch.tensor(m))).item())
    return 2.0 ** (e - 7)


def router_margins(router_logits: torch.Tensor) -> tuple:
    """Smallest top-k margin and smallest adjacent gap of the sorted
    probabilities. Both must be positive: a tie would leave the top-k
    order ambiguous between a sort and a topk, and so make the fixture
    unusable for exact-index asserts."""
    probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    top_margin = (sorted_probs[:, TOP_K - 1] - sorted_probs[:, TOP_K]).min().item()
    inner_gap = (sorted_probs[:, :TOP_K - 1] - sorted_probs[:, 1:TOP_K]).min().item()
    return top_margin, inner_gap


def moe_bands(capture: dict) -> dict:
    """Routed-block bands from bf16 ulp arithmetic, first principles:
    one ulp at the max magnitude for the router logits, routing weights
    and shared gate, three ulps for the routed-block output (a GEMM
    boundary flip is tolerated up to that band)."""
    return {
        "router_logits_band": ulp_bf16(capture["router_logits"].abs().max().item()),
        "routing_weights_band": ulp_bf16(capture["routing_weights"].abs().max().item()),
        "shared_gate_band": ulp_bf16(capture["shared_gate"].abs().max().item()),
        "moe_output_band": 3.0 * ulp_bf16(capture["moe_output"].abs().max().item()),
    }


def save_fixture(case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata
    file."""
    filename = f"layer-{MODEL_NAME}-{case_num:02d}.safetensor"
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


def generate_layer0_fixture(cfg: Qwen3_5MoeTextConfig, weight_map: dict) -> None:
    """Full decoder layer 0 (GDN + MoE): the recurrent rule as the bitwise
    reference, the vendored chunked rule as the module forward, and the
    vendored decoder layer as the chain cross-check."""
    base = PREFIX_FMT.format(layer=0)
    weights = load_layer_weights(weight_map, cfg, 0)

    # Full decoder layer 0 from the vendored class, real weights.
    layer = Qwen3_5MoeDecoderLayer(cfg, 0)
    with torch.no_grad():
        if hasattr(layer, "linear_attn"):
            gdn = layer.linear_attn
            gdn.in_proj_qkv.weight.data = weights["linear_attn.in_proj_qkv.weight"]
            gdn.in_proj_z.weight.data = weights["linear_attn.in_proj_z.weight"]
            gdn.out_proj.weight.data = weights["linear_attn.out_proj.weight"]
            gdn.A_log.data = weights["linear_attn.A_log"].to(torch.bfloat16)
            gdn.conv1d.weight.data = weights["linear_attn.conv1d.weight"]
            gdn.dt_bias.data = weights["linear_attn.dt_bias"]
            gdn.in_proj_a.weight.data = weights["linear_attn.in_proj_a.weight"]
            gdn.in_proj_b.weight.data = weights["linear_attn.in_proj_b.weight"]
            gdn.norm.weight.data = weights["linear_attn.norm.weight"].to(torch.bfloat16)
        moe = layer.mlp
        moe.gate.weight.data = weights["mlp.gate.weight"]
        moe.experts.gate_up_proj.data = weights["mlp.experts.gate_up_proj"]
        moe.experts.down_proj.data = weights["mlp.experts.down_proj"]
        moe.shared_expert.gate_proj.weight.data = \
            weights["mlp.shared_expert.gate_proj.weight"]
        moe.shared_expert.up_proj.weight.data = \
            weights["mlp.shared_expert.up_proj.weight"]
        moe.shared_expert.down_proj.weight.data = \
            weights["mlp.shared_expert.down_proj.weight"]
        moe.shared_expert_gate.weight.data = weights["mlp.shared_expert_gate.weight"]
        layer.input_layernorm.weight.data = weights["input_layernorm.weight"]
        layer.post_attention_layernorm.weight.data = \
            weights["post_attention_layernorm.weight"]
        input_ln = layer.input_layernorm
        post_ln = layer.post_attention_layernorm
    layer.eval()

    # Seed search: a tie in the top-k probabilities would leave the order
    # ambiguous, so the fixture demands positive margins. Only the cheap
    # prefix of the chain runs per candidate.
    for offset in range(64):
        seed = SEED_LAYER0_BASE + offset
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        x = torch.randn(1, SEQ_LEN, HIDDEN, generator=gen, dtype=torch.bfloat16)
        with torch.no_grad():
            h_norm = input_ln(x)
            seq_replay = gdn_forward_replay(gdn, h_norm, use_recurrent=True)
            h1 = x + seq_replay["output"]
            h2 = post_ln(h1)
            flat = h2.view(-1, HIDDEN)
            router_logits, _, _ = moe.gate(flat)
        top_margin, inner_gap = router_margins(router_logits)
        if top_margin > 0 and inner_gap > 0:
            break
    else:
        raise SystemExit(
            "[gen_qwen36_layer_fixtures] no seed with unambiguous top-k "
            "margins in 64 tries for layer 0")

    with torch.no_grad():
        module_mixer = gdn(h_norm, cache_params=None, attention_mask=None)
        chunk_replay = gdn_forward_replay(gdn, h_norm, use_recurrent=False)
        assert torch.equal(module_mixer, chunk_replay["output"]), (
            "[gen_qwen36_layer_fixtures] chunked replay diverged from the "
            "module forward for layer 0")

        # The vendored decoder layer forward is the chain ground truth,
        # bitwise-equal to the manual chunked chain.
        real_layer_out = layer(
            x, position_embeddings=None, attention_mask=None, past_key_values=None)
        h1_chunked = x + chunk_replay["output"]
        h2_chunked = post_ln(h1_chunked)
        moe_chunked = moe_forward_capture(moe, h2_chunked)
        manual_layer_out = h1_chunked + moe_chunked["moe_output"]
        assert torch.equal(real_layer_out, manual_layer_out), (
            "[gen_qwen36_layer_fixtures] manual chain diverged from the "
            "vendored decoder layer forward for layer 0")

        # Recurrent chain for the Nim block with the same routed block
        # and residuals.
        moe_seq = moe_forward_capture(moe, h2)
        layer_out_seq = h1 + moe_seq["moe_output"]

        # The fixture metadata records the observed recurrent-vs-chunked
        # mixer divergence, documenting the observed floor.
        mixer_diff = max_abs_diff(seq_replay["output"], module_mixer)
        assert mixer_diff < BLOCK_BAR, (
            f"[gen_qwen36_layer_fixtures] recurrent-vs-chunked mixer diff "
            f"outside (0, {BLOCK_BAR}): {mixer_diff}")

        # Bands: the routed-block band plus one bf16 ulp for the residual
        # add, at the larger of the two magnitudes. The GDN block stays
        # bitwise against the recurrent replay, so the chain delta comes
        # from the routed-block band alone.
        bands = moe_bands(moe_seq)
        layer_band = bands["moe_output_band"] + ulp_bf16(
            max(layer_out_seq.abs().max().item(),
                moe_seq["moe_output"].abs().max().item()))

    save_fixture(0, {
        "model": MODEL_NAME,
        "layer": base + "linear_attn",
        "case": "prefill_seq6",
        "seq_len": SEQ_LEN,
        "chunk_size": CHUNK_SIZE,
        "hidden_size": HIDDEN,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "seed": seed,
        "num_threads": NUM_THREADS,
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "vendored_sha": VENDORED_SHA,
        "bands": {
            **bands,
            "layer_output_band": layer_band,
            "layer_output_chunked_bar": BLOCK_BAR,
        },
        "margins": {
            "topk_margin_min": top_margin,
            "topk_inner_gap_min": inner_gap,
        },
        "chunk_vs_recurrent_mixer_diff": mixer_diff,
    }, {
        "layer_input": x,
        "input_layernorm_output": h_norm,
        "gdn_block_output_seq": seq_replay["output"],
        "gdn_block_output_chunked": module_mixer,
        "post_attention_layernorm_output": h2,
        "router_logits": moe_seq["router_logits"],
        "topk_indices": moe_seq["topk_indices"],
        "routing_weights": moe_seq["routing_weights"],
        "shared_gate": moe_seq["shared_gate"],
        "moe_output": moe_seq["moe_output"],
        "layer_output_seq": layer_out_seq,
        "layer_output_chunked": real_layer_out,
    })
    print(f"Generated layer 0 fixture (seed {seed})")


def generate_layer3_fixture(cfg: Qwen3_5MoeTextConfig, weight_map: dict,
                            rotary) -> None:
    """Full decoder layer 3 (gated attention + MoE): the vendored decoder
    layer forward as the chain ground truth, prefill T=6 at positions 0..5."""
    base = PREFIX_FMT.format(layer=3)
    weights = load_layer_weights(weight_map, cfg, 3)
    layer = Qwen3_5MoeDecoderLayer(cfg, 3)
    with torch.no_grad():
        attn = layer.self_attn
        attn.q_proj.weight.data = weights["self_attn.q_proj.weight"]
        attn.k_proj.weight.data = weights["self_attn.k_proj.weight"]
        attn.v_proj.weight.data = weights["self_attn.v_proj.weight"]
        attn.o_proj.weight.data = weights["self_attn.o_proj.weight"]
        attn.q_norm.weight.data = weights["self_attn.q_norm.weight"].to(torch.bfloat16)
        attn.k_norm.weight.data = weights["self_attn.k_norm.weight"].to(torch.bfloat16)
        moe = layer.mlp
        moe.gate.weight.data = weights["mlp.gate.weight"]
        moe.experts.gate_up_proj.data = weights["mlp.experts.gate_up_proj"]
        moe.experts.down_proj.data = weights["mlp.experts.down_proj"]
        moe.shared_expert.gate_proj.weight.data = \
            weights["mlp.shared_expert.gate_proj.weight"]
        moe.shared_expert.up_proj.weight.data = \
            weights["mlp.shared_expert.up_proj.weight"]
        moe.shared_expert.down_proj.weight.data = \
            weights["mlp.shared_expert.down_proj.weight"]
        moe.shared_expert_gate.weight.data = weights["mlp.shared_expert_gate.weight"]
        layer.input_layernorm.weight.data = weights["input_layernorm.weight"]
        layer.post_attention_layernorm.weight.data = \
            weights["post_attention_layernorm.weight"]
        input_ln = layer.input_layernorm
        post_ln = layer.post_attention_layernorm
    layer.eval()

    for offset in range(64):
        seed = SEED_LAYER3_BASE + offset
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        x = torch.randn(1, SEQ_LEN, HIDDEN, generator=gen, dtype=torch.bfloat16)
        position_ids = torch.arange(SEQ_LEN, device="cpu").view(1, -1).contiguous()
        with torch.no_grad():
            cos, sin = rotary(x, position_ids)
            h_norm = input_ln(x)
            mixer, _ = attn(
                h_norm, position_embeddings=(cos, sin),
                attention_mask=None, past_key_values=None)
            h2 = post_ln(x + mixer)
            flat = h2.view(-1, HIDDEN)
            router_logits, _, _ = moe.gate(flat)
        top_margin, inner_gap = router_margins(router_logits)
        if top_margin > 0 and inner_gap > 0:
            break
    else:
        raise SystemExit(
            "[gen_qwen36_layer_fixtures] no seed with unambiguous top-k "
            "margins in 64 tries for layer 3")

    with torch.no_grad():
        cos, sin = rotary(x, position_ids)
        h_norm = input_ln(x)

        # The module attention forward, the mixer ground truth, equals
        # the manual capture bitwise.
        mixer_real, _ = attn(
            h_norm, position_embeddings=(cos, sin),
            attention_mask=None, past_key_values=None)
        mixer_cap = attn_forward_capture(attn, h_norm, (cos, sin))
        assert torch.equal(mixer_real, mixer_cap["output"]), (
            "[gen_qwen36_layer_fixtures] attention replay diverged from the "
            "module forward for layer 3")

        # The vendored decoder layer forward, the chain ground truth,
        # equals the manual chain bitwise.
        real_layer_out = layer(
            x, position_embeddings=(cos, sin),
            attention_mask=None, past_key_values=None)
        h1 = x + mixer_cap["output"]
        h2 = post_ln(h1)
        moe_cap = moe_forward_capture(moe, h2)
        manual_layer_out = h1 + moe_cap["moe_output"]
        assert torch.equal(real_layer_out, manual_layer_out), (
            "[gen_qwen36_layer_fixtures] manual chain diverged from the "
            "vendored decoder layer forward for layer 3")

        # Bands: cross-version SDPA CPU kernel noise. The Nim binary links
        # libtorch 2.11, this fixture records with torch 2.13. Identical
        # inputs and flags differ by one bf16 ulp at the differing
        # element, invariant to stride and to input provenance. The layer
        # band compounds the attention block bar with the routed-block
        # band and one bf16 ulp for the residual add.
        bands = moe_bands(moe_cap)
        layer_band = ATTN_BAR + bands["moe_output_band"] + ulp_bf16(
            max(real_layer_out.abs().max().item(),
                moe_cap["moe_output"].abs().max().item()))

    save_fixture(3, {
        "model": MODEL_NAME,
        "layer": base + "self_attn",
        "case": "prefill_seq6",
        "seq_len": SEQ_LEN,
        "hidden_size": HIDDEN,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "seed": seed,
        "num_threads": NUM_THREADS,
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "vendored_sha": VENDORED_SHA,
        "bands": {
            **bands,
            "attn_mixer_band": ATTN_BAR,
            "layer_output_band": layer_band,
        },
        "margins": {
            "topk_margin_min": top_margin,
            "topk_inner_gap_min": inner_gap,
        },
    }, {
        "layer_input": x,
        "position_ids": position_ids,
        "cos": cos, "sin": sin,
        "input_layernorm_output": h_norm,
        "attn_mixer_output": mixer_cap["output"],
        "post_attention_layernorm_output": h2,
        "router_logits": moe_cap["router_logits"],
        "topk_indices": moe_cap["topk_indices"],
        "routing_weights": moe_cap["routing_weights"],
        "shared_gate": moe_cap["shared_gate"],
        "moe_output": moe_cap["moe_output"],
        "layer_output": real_layer_out,
    })
    print(f"Generated layer 3 fixture (seed {seed})")


def main() -> None:
    check_ram()
    check_vendored_sha()

    cfg = load_text_config()
    weight_map = load_index_weight_map()

    # The text rotary of the checkpoint: theta 1e7, head 256, partial 0.25.
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextRotaryEmbedding,
    )
    rotary = Qwen3_5MoeTextRotaryEmbedding(cfg)

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_layer0_fixture(cfg, weight_map)
    generate_layer3_fixture(cfg, weight_map, rotary)


if __name__ == "__main__":
    main()
