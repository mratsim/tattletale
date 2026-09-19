#!/usr/bin/env python3
"""Layer-0 fixture file of the Qwen3.6-35B-A3B checkpoint, recorded on CPU
torch bf16 with the installed reference modeling, from safetensors.

Single-file grammar with one fixture file per family layer.

- every mixture sits in the one file as a named tensor group, one metadata
  sidecar and one stats sidecar serve the file
- the payload carries the suite-read driving tensors, the recorded
  intermediates and outputs stay on the stats frame as fingerprints

No Qwen3 analog exists to inherit, the GatedDeltaNet SSM mixer and the fused
rank-3 routed experts match the Qwen3.5 family, both absent from Qwen3.

Consumed by tests/q_bf16/t_bf16_qwen36moe_01_layer_internals.nim, one
assertion block per mixture.

Emitted under tests/fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-0/:

| file                                                   | contents                                                           |
| ------------------------------------------------------ | ------------------------------------------------------------------ |
| layer0-Qwen3.6-35B-A3B-00.safetensor                   | the three mixtures, gdn.input, layer.layer_input and moe.h         |
| layer0-Qwen3.6-35B-A3B-00.safetensor.metadata.json.zst | per-mixture metadata under the mixtures key                        |
| layer0-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst    | keys namespaced by mixture, one uniform record per recorded tensor |

Mixture seeds and cases:

- gdn.input, the GDN block prefill T=5 with seed 71
- layer.layer_input, the full decoder layer 0 (GDN mixer plus routed MoE)
  prefill T=6, unambiguous-top-k seed search from 211
- moe.h, the routed block on T=6 deterministic tokens, seed search from 71

Mixture payloads and stats entries:

- gdn, payload gdn.input, stats entries over the captured prefill
  intermediates (conv output, q/k/v post-split, g, beta), the recurrent-rule
  block output and the chunked module output
- layer, payload layer.layer_input, stats entries over the recorded chain:
  layernorm outputs, GDN block output under both core rules, router logits,
  routing weights, shared expert gate and layer outputs
- moe, payload moe.h, stats entries over the f32 router logits, top-k
  expert ids as an f32 view, pre-cast fp32 renormalized values, dtype-cast
  routing weights and the MoE output

Compute forms of the gated delta rule core:

- the chunked form is the installed forward,
  torch_chunk_gated_delta_rule at chunk_size 64
- the recurrent form is the bitwise reference for the Nim block, matching
  torch_recurrent_gated_delta_rule element for element
- the recurrent-to-chunked distance equals the reference floor between
  chunked and recurrent forms

Run from the worktree root, twice, cmp proves byte determinism:

  uv run python workspace/transformers/tests/testgen/gen_bf16_qwen36moe_01_layer_internals.py

RAM guards:

- the script refuses the weight load when free memory sits below the floor
- another python/torch process holding RAM also blocks the run
- the process chain of this script stays excluded from that check
"""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from fixture_stats import assert_path_equivalent, write_stats_file  # noqa: E402
import torch.nn.functional as F  # noqa: E402, the path insert precedes the import
from safetensors import safe_open  # noqa: E402
from safetensors import torch as st  # noqa: E402

import transformers  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeDecoderLayer,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

# Determinism, single intra-op thread, deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Checkpoint, fixture and config paths.
MODEL_NAME = "Qwen3.6-35B-A3B"
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
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

NUM_THREADS = 1

# Per-mixture seeds, independent and order-agnostic. The routed-block seed
# search and the full-layer seed search reject a seed whose top-k margins tie.
SEED_GDN = 71
SEED_LAYER_BASE = 211
SEED_MOE_BASE = 71

# Generation shapes, geometry lives in the parsed config.
GDN_SEQ = 5
LAYER_SEQ = 6
MOE_SEQ = 6
CHUNK_SIZE = 64

PREFIX_FMT = "model.language_model.layers.{layer}."
MIN_FREE_BYTES = 8 * 1024 ** 3

# Recurrent-vs-chunked divergence caps at 35B dims, two tiers:
#
# - BLOCK_BAR, the mixer-level bar
# - the SSM floor, about one fp32 ulp at the divergent element's magnitude,
#   sub-linear in seq_len, about four orders of magnitude under bf16 rounding
#
# The cap below allows four fp32 ulps at the generated state's max magnitude.
BLOCK_BAR = 1e-3
SSM_ULP_MARGIN = 8.0


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals] vm_stat gave no 'Pages free' line")


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
    process holds RAM.

    Guard:

    - free memory below the floor raises
    - the pgrep match excludes this process chain, its own command line
      spells the torch dependency of the run
    """
    free = free_bytes()
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_01_layer_internals] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_01_layer_internals] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def load_text_config() -> Qwen3_5MoeTextConfig:
    """Load the nested text_config from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeTextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


# Geometry reads from the parsed config (config is king), no hardcoded
# copies of the checkpoint's hidden size, expert count, or top-k.
GEOMETRY = load_text_config()
HIDDEN = GEOMETRY.hidden_size
NUM_EXPERTS = GEOMETRY.num_experts
TOP_K = GEOMETRY.num_experts_per_tok


def load_index_weight_map() -> dict:
    """Parse the checkpoint index and return its weight_map."""
    with open(INDEX_PATH) as f:
        index = json.load(f)
    return index["weight_map"]


def layer_shorts(cfg: Qwen3_5MoeTextConfig, layer_idx: int) -> list:
    """Tensor name suffixes of one decoder layer, the token mixer keys of its kind plus the routed block and the two layernorms."""
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
    """Loads one decoder layer's tensors through the checkpoint index.

    Returns the weight dict. Only the safetensors files holding the layer
    open (safe_open, memory mapped, only these tensors are copied)."""
    base = PREFIX_FMT.format(layer=layer_idx)
    by_file = OrderedDict()
    for short in layer_shorts(cfg, layer_idx):
        key = base + short
        by_file.setdefault(weight_map[key], []).append(key)
    weights = {}
    for file_name, keys in by_file.items():
        with safe_open(os.path.join(MODEL_DIR, file_name), framework="pt") as f:
            for key in keys:
                weights[key[len(base):]] = f.get_tensor(key).clone()
    return weights


def build_decoder_layer0(cfg: Qwen3_5MoeTextConfig, weights: dict) -> Qwen3_5MoeDecoderLayer:
    """Qwen3_5MoeDecoderLayer with real layer-0 weights loaded.

    Args:
    - cfg, weights, the parsed text config and the load_layer_weights dict

    Returns:
    - the layer-0 decoder layer in eval mode, its linear_attn and mlp
      plus both layernorms feed every mixture

    A_log and the norm weights are cast to bf16, matching a bf16 model load
    (the checkpoint already stores them bf16, so the cast is a no-op).
    """
    layer = Qwen3_5MoeDecoderLayer(cfg, LAYER_IDX)
    gdn = layer.linear_attn
    moe = layer.mlp
    with torch.no_grad():
        gdn.in_proj_qkv.weight.data = weights["linear_attn.in_proj_qkv.weight"]
        gdn.in_proj_z.weight.data = weights["linear_attn.in_proj_z.weight"]
        gdn.out_proj.weight.data = weights["linear_attn.out_proj.weight"]
        gdn.A_log.data = weights["linear_attn.A_log"].to(torch.bfloat16)
        gdn.conv1d.weight.data = weights["linear_attn.conv1d.weight"]
        gdn.dt_bias.data = weights["linear_attn.dt_bias"]
        gdn.in_proj_a.weight.data = weights["linear_attn.in_proj_a.weight"]
        gdn.in_proj_b.weight.data = weights["linear_attn.in_proj_b.weight"]
        gdn.norm.weight.data = weights["linear_attn.norm.weight"].to(torch.bfloat16)
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
    """Replays the reference Qwen3_5MoeGatedDeltaNet.forward with a selectable
    core rule, capturing every intermediate.

    Args:
    - block, hidden_states, the weighted layer-0 module and its bf16 input
    - use_recurrent, True selects torch_recurrent_gated_delta_rule, False
      the chunked rule

    Returns:
    - the capture dict, every intermediate tensor under its own name

    The caller asserts the chunked replay against the module's own forward
    through assert_path_equivalent. The recurrent replay is the bitwise
    reference for the Nim implementation.
    """
    batch_size, seq_len, _ = hidden_states.shape
    with torch.no_grad():
        mixed_qkv = block.in_proj_qkv(hidden_states).transpose(1, 2)
        z = block.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, block.head_v_dim)
        b = block.in_proj_b(hidden_states)
        a = block.in_proj_a(hidden_states)

        # Fresh prefill conv, matching the reference causal_conv1d_fn fallback:
        #
        # - the fallback's only padding source is the built-in padding
        #   (kernel - 1) of this same F.conv1d call
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
        # forward where the value heads share one key head per group, a no-op
        # when the two head counts are equal.
        query_core, key_core = query, key
        if block.num_v_heads // block.num_k_heads > 1:
            ratio = block.num_v_heads // block.num_k_heads
            query_core = query.repeat_interleave(ratio, dim=2)
            key_core = key.repeat_interleave(ratio, dim=2)

        rule = (torch_recurrent_gated_delta_rule if use_recurrent
                else torch_chunk_gated_delta_rule)
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
        "g": g, "beta": beta,
        "core_attn_out": core_attn_out,
        "ssm_state": ssm_state,
        "normed": normed,
        "output": output,
    }


def moe_forward_capture(moe, hidden_states: torch.Tensor) -> dict:
    """Replays the reference Qwen3_5MoeSparseMoeBlock.forward, capturing
    every intermediate. Returns the capture dict, the caller asserting
    it against the module forward with assert_path_equivalent."""
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


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute element difference of two tensors, compared in f32."""
    return (a.float() - b.float()).abs().max().item()


def ulp_fp32(m: float) -> float:
    """One fp32 ulp at magnitude m, fp32 has 23 significand bits, so for m
    in [2**e, 2**(e+1)) the ulp is 2**(e-23). Zero maps to 0."""
    if m <= 0:
        return 0.0
    e = int(torch.floor(torch.log2(torch.tensor(m))).item())
    return 2.0 ** (e - 23)


def ulp_bf16(m: float) -> float:
    """One bf16 ulp at magnitude m, bf16 stores 7 significand bits, so
    for m in [2**e, 2**(e+1)) the ulp is 2**(e-7), zero maps to 0."""
    if m <= 0:
        return 0.0
    e = int(torch.floor(torch.log2(torch.tensor(m))).item())
    return 2.0 ** (e - 7)


def ssm_cap(ssm_state: torch.Tensor) -> float:
    """Returns the recurrent-vs-chunked SSM cap, SSM_ULP_MARGIN fp32 ulps
    measured at the generated state's max magnitude:

    - the two rules diverge by a few ulps at the largest divergent element,
      whose magnitude is bounded by the state max
    """
    return SSM_ULP_MARGIN * ulp_fp32(ssm_state.abs().max().item())


def router_margins(router_logits: torch.Tensor) -> tuple:
    """Smallest top-k margin and smallest adjacent gap of the sorted probabilities.

    Returns the two margins. Both must be positive, a tie would leave
    the top-k order ambiguous between a sort and a topk, and so make the fixture
    unusable for exact-index asserts."""
    probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    top_margin = (sorted_probs[:, TOP_K - 1] - sorted_probs[:, TOP_K]).min().item()
    inner_gap = (sorted_probs[:, :TOP_K - 1] - sorted_probs[:, 1:TOP_K]).min().item()
    return top_margin, inner_gap


def moe_bands(capture: dict) -> dict:
    """Routed-block bands from bf16 ulp arithmetic, first principles.

    Returns the band dict. One ulp at the max magnitude for the router logits,
    routing weights, and `shared_gate`, three ulps for the routed-block
    output (a GEMM boundary flip is tolerated up to that band)."""
    return {
        "router_logits_band": ulp_bf16(capture["router_logits"].abs().max().item()),
        "routing_weights_band": ulp_bf16(capture["routing_weights"].abs().max().item()),
        "shared_gate_band": ulp_bf16(capture["shared_gate"].abs().max().item()),
        "moe_output_band": 3.0 * ulp_bf16(capture["moe_output"].abs().max().item()),
    }


def generate_gdn_mixture(gdn) -> tuple:
    """GDN block prefill T=5, the recurrent reference plus the chunked module output.

    Args:
    - gdn, the weighted layer-0 GatedDeltaNet module

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the captured prefill intermediates for the stats frame

    The chunked replay must match the module forward, the path-equivalence
    instrument guards the replay and the recurrent-vs-chunked divergence
    stays inside the documented floors.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED_GDN)
    x = torch.randn(1, GDN_SEQ, HIDDEN, generator=gen, dtype=torch.bfloat16)

    with torch.no_grad():
        module_output = gdn(x)  # reference forward, chunked rule
        chunk_replay = gdn_forward_replay(gdn, x, use_recurrent=False)
        seq_replay = gdn_forward_replay(gdn, x, use_recurrent=True)
    assert_path_equivalent(module_output, chunk_replay["output"],
        "[gen_bf16_qwen36moe_01_layer_internals] chunked replay vs the module forward")

    output_diff = max_abs_diff(seq_replay["output"], module_output)
    core_diff = max_abs_diff(seq_replay["core_attn_out"], chunk_replay["core_attn_out"])
    ssm_diff = max_abs_diff(
        seq_replay["ssm_state"][0], chunk_replay["ssm_state"][0])
    assert output_diff < BLOCK_BAR, (
        f"[gen_bf16_qwen36moe_01_layer_internals] recurrent-vs-chunked output diff outside "
        f"(0, {BLOCK_BAR}): {output_diff}")
    assert ssm_diff <= ssm_cap(seq_replay["ssm_state"][0]), (
        f"[gen_bf16_qwen36moe_01_layer_internals] recurrent-vs-chunked SSM diff outside the "
        f"documented floor: {ssm_diff}")

    meta = {
        "layer": "model.language_model.layers.0.linear_attn",
        "case": "prefill_seq5",
        "seq_len": GDN_SEQ,
        "chunk_size": CHUNK_SIZE,
        "head_k_dim": gdn.head_k_dim,
        "head_v_dim": gdn.head_v_dim,
        "num_k_heads": gdn.num_k_heads,
        "num_v_heads": gdn.num_v_heads,
        "seed": SEED_GDN,
        "chunk_vs_recurrent_output_diff": output_diff,
        "chunk_vs_recurrent_core_diff": core_diff,
        "chunk_vs_recurrent_ssm_diff": ssm_diff,
    }
    payload = {"gdn.input": x}
    captured = {
        "gdn.conv_output": chunk_replay["conv_output"],
        "gdn.q": chunk_replay["query"], "gdn.k": chunk_replay["key"],
        "gdn.v": chunk_replay["value"],
        "gdn.g": chunk_replay["g"], "gdn.beta": chunk_replay["beta"],
        "gdn.output_seq": seq_replay["output"],
        "gdn.output_chunked": module_output,
    }
    print(f"[gen_bf16_qwen36moe_01_layer_internals] gdn mixture T={GDN_SEQ}: "
          f"output diff {output_diff:.3e}, core diff {core_diff:.3e}, "
          f"ssm diff {ssm_diff:.3e}")
    return meta, payload, captured


def generate_layer_mixture(layer, gdn, moe, input_ln, post_ln) -> tuple:
    """Full decoder layer 0 (GDN mixer + routed MoE) fixture mixture.

    Args:
    - layer, gdn, moe, input_ln, post_ln, the weighted layer-0 modules

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the captured chain tensors for the stats frame

    Compute forms:

    - the recurrent rule is the bitwise reference for the Nim block
    - the chunked rule is the module forward, the installed decoder layer
      is the chain cross-check
    - the seed search rejects a top-k margin tie, only the cheap prefix
      of the chain runs per candidate
    """
    for offset in range(64):
        seed = SEED_LAYER_BASE + offset
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        x = torch.randn(1, LAYER_SEQ, HIDDEN, generator=gen, dtype=torch.bfloat16)
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
            "[gen_bf16_qwen36moe_01_layer_internals] no seed with unambiguous top-k "
            "margins in 64 tries for layer 0")

    with torch.no_grad():
        module_mixer = gdn(h_norm, cache_params=None, attention_mask=None)
        chunk_replay = gdn_forward_replay(gdn, h_norm, use_recurrent=False)
        assert_path_equivalent(module_mixer, chunk_replay["output"],
            "[gen_bf16_qwen36moe_01_layer_internals] chunked replay vs the "
            "module forward for layer 0")

        # The installed decoder layer forward is the chain ground truth,
        # bitwise-equal to the manual chunked chain.
        real_layer_out = layer(
            x, position_embeddings=None, attention_mask=None, past_key_values=None)
        h1_chunked = x + chunk_replay["output"]
        h2_chunked = post_ln(h1_chunked)
        moe_chunked = moe_forward_capture(moe, h2_chunked)
        manual_layer_out = h1_chunked + moe_chunked["moe_output"]
        assert_path_equivalent(real_layer_out, manual_layer_out,
            "[gen_bf16_qwen36moe_01_layer_internals] manual chain vs the "
            "decoder layer forward for layer 0")

        # recurrent chain for the Nim block with the same routed block and residuals
        moe_seq = moe_forward_capture(moe, h2)
        layer_out_seq = h1 + moe_seq["moe_output"]

        # The fixture metadata records the observed recurrent-vs-chunked
        # mixer divergence, documenting the observed floor.
        mixer_diff = max_abs_diff(seq_replay["output"], module_mixer)
        assert mixer_diff < BLOCK_BAR, (
            f"[gen_bf16_qwen36moe_01_layer_internals] recurrent-vs-chunked mixer diff "
            f"outside (0, {BLOCK_BAR}): {mixer_diff}")

        # the routed-block band plus one bf16 ulp for the residual add
        # at the larger of the two magnitudes
        #
        # - the GDN block stays bitwise against the recurrent replay,
        #   the chain delta comes from the routed-block band alone
        bands = moe_bands(moe_seq)
        layer_band = bands["moe_output_band"] + ulp_bf16(
            max(layer_out_seq.abs().max().item(),
                moe_seq["moe_output"].abs().max().item()))

    meta = {
        "layer": "model.language_model.layers.0",
        "case": "prefill_seq6",
        "seq_len": LAYER_SEQ,
        "chunk_size": CHUNK_SIZE,
        "seed": seed,
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
    }
    payload = {"layer.layer_input": x}
    captured = {
        "layer.input_layernorm_output": h_norm,
        "layer.gdn_block_output_seq": seq_replay["output"],
        "layer.gdn_block_output_chunked": module_mixer,
        "layer.post_attention_layernorm_output": h2,
        "layer.router_logits": moe_seq["router_logits"],
        "layer.routing_weights": moe_seq["routing_weights"],
        "layer.shared_gate": moe_seq["shared_gate"],
        "layer.moe_output": moe_seq["moe_output"],
        "layer.layer_output_seq": layer_out_seq,
        "layer.layer_output_chunked": real_layer_out,
    }
    print(f"[gen_bf16_qwen36moe_01_layer_internals] layer mixture (seed {seed})")
    return meta, payload, captured


def generate_moe_mixture(moe) -> tuple:
    """Routed-block fixture mixture on T=6 deterministic tokens.

    Args:
    - moe, the weighted layer-0 sparse MoE block

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the captured block tensors for the stats frame

    Router chain, computed standalone:

    - logits at the hidden-state dtype, softmax over the f32 expert logits,
      top-k, fp32 renorm, cast back last
    - the fp32 renormed values are kept pre-cast
    - the manual chain must reproduce the module router scores, guarded
      by the path-equivalence instrument, the indices checked by plain
      integer equality, the seed search rejecting top-k margin ties
    """
    top_k = TOP_K
    for offset in range(64):
        seed = SEED_MOE_BASE + offset
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        h = torch.randn(1, MOE_SEQ, HIDDEN, generator=gen, dtype=torch.bfloat16)
        with torch.no_grad():
            router_logits, _, _ = moe.gate(h)
            probs = torch.nn.functional.softmax(
                router_logits, dtype=torch.float32, dim=-1)
        sorted_probs = torch.sort(probs, dim=-1, descending=True).values
        top_margin = (sorted_probs[:, top_k - 1] - sorted_probs[:, top_k]).min().item()
        inner_gap = (sorted_probs[:, :top_k - 1] - sorted_probs[:, 1:top_k]).min().item()
        if top_margin > 0 and inner_gap > 0:
            break
    else:
        raise SystemExit(
            "[gen_bf16_qwen36moe_01_layer_internals] no seed with unambiguous top-k "
            "margins in 64 tries for the routed block")

    with torch.no_grad():
        router_logits, router_scores, router_indices = moe.gate(h)
        top_values_fp32, top_indices = torch.topk(probs, top_k, dim=-1)
        renorm_fp32 = top_values_fp32 / top_values_fp32.sum(dim=-1, keepdim=True)
        routing_weights = renorm_fp32.to(router_logits.dtype)

        assert_path_equivalent(routing_weights, router_scores,
            "[gen_bf16_qwen36moe_01_layer_internals] manual router chain vs the module router scores")
        assert bool((top_indices == router_indices).all()), \
            "[gen_bf16_qwen36moe_01_layer_internals] manual top-k indices != module top-k indices"

        shared_gate = torch.nn.functional.sigmoid(
            moe.shared_expert_gate(h))
        moe_output = moe(h)

    bands = {
        "router_logits_band": ulp_bf16(router_logits.float().abs().max().item()),
        "routing_weights_band": ulp_bf16(routing_weights.float().abs().max().item()),
        "shared_gate_band": ulp_bf16(shared_gate.abs().max().item()),
        "output_band": 3.0 * ulp_bf16(moe_output.abs().max().item()),
    }
    meta = {
        "layer": "model.language_model.layers.0.mlp",
        "case": "routed_block_seq6",
        "seq_len": MOE_SEQ,
        "seed": seed,
        "bands": bands,
        "margins": {
            "topk_margin_min": top_margin,
            "topk_inner_gap_min": inner_gap,
        },
    }
    payload = {"moe.h": h}
    captured = {
        "moe.router_logits": router_logits.to(torch.float32),
        "moe.topk_indices": top_indices,
        "moe.renorm_values": renorm_fp32.to(torch.float32),
        "moe.routing_weights": routing_weights,
        "moe.shared_gate": shared_gate[0, :, 0].to(torch.float32),
        "moe.moe_output": moe_output,
    }
    print(f"[gen_bf16_qwen36moe_01_layer_internals] moe mixture (seed {seed}): "
          f"topk margin {top_margin:.3e}, inner gap {inner_gap:.3e}")
    return meta, payload, captured


def save_fixture(metadata: dict, mixtures: list) -> None:
    """Writes the single-file fixture set, one safetensors payload carrying
    every mixture as a named tensor group, one metadata sidecar and one
    stats sidecar with keys namespaced by mixture.

    Args:
    - metadata, the merged metadata frame
    - mixtures, one (payload, captured) pair per mixture, the payload
      tensors carrying their file names, the captured tensors already
      carrying their namespaced stats keys
    """
    file_tensors = OrderedDict()
    stats_entries = []
    for payload, captured in mixtures:
        for name, tensor in payload.items():
            file_tensors[name] = tensor.detach().cpu().contiguous()
            stats_entries.append((name, file_tensors[name]))
        for name, tensor in captured.items():
            stats_entries.append((name, tensor.detach().cpu().contiguous()))

    # integer expert ids carry no stats record, the f32 view fingerprints them
    stats_entries = [
        (name, tensor.to(torch.float32) if name.endswith("topk_indices") else tensor)
        for name, tensor in stats_entries
    ]

    serialized = st.save(file_tensors, metadata=None)
    with open(FIXTURE_PATH, "wb") as f:
        f.write(serialized)

    write_json_zst(FIXTURE_PATH + ".metadata.json.zst", metadata)
    write_stats_file(FIXTURE_PATH + ".stats.json.zst", FIXTURE_STEM + ".safetensor",
                     stats_entries)


def main() -> None:
    """Record the layer-0 fixture file set after the RAM guard."""
    check_ram()

    cfg = load_text_config()
    weight_map = load_index_weight_map()
    weights = load_layer_weights(weight_map, cfg, LAYER_IDX)
    layer = build_decoder_layer0(cfg, weights)

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    gdn_meta, gdn_payload, gdn_captured = generate_gdn_mixture(layer.linear_attn)
    layer_meta, layer_payload, layer_captured = generate_layer_mixture(
        layer, layer.linear_attn, layer.mlp,
        layer.input_layernorm, layer.post_attention_layernorm)
    moe_meta, moe_payload, moe_captured = generate_moe_mixture(layer.mlp)

    metadata = {
        "model": MODEL_NAME,
        "file": FIXTURE_STEM + ".safetensor",
        "dtype": "bfloat16",
        "hidden_size": HIDDEN,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "num_threads": NUM_THREADS,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "mixtures": {
            "gdn": gdn_meta,
            "layer": layer_meta,
            "moe": moe_meta,
        },
    }
    save_fixture(metadata, [
        (gdn_payload, gdn_captured),
        (layer_payload, layer_captured),
        (moe_payload, moe_captured),
    ])

    print(f"[gen_bf16_qwen36moe_01_layer_internals] torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"[gen_bf16_qwen36moe_01_layer_internals] wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
