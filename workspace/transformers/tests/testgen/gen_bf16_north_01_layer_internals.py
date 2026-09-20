#!/usr/bin/env python3
"""Layer-internals fixture file of the North-Mini-Code-1.0 checkpoint, recorded
with torch bf16 on Metal (mps) under the installed reference modeling.

Single-file grammar, one fixture file per family layer group, one bare bf16
driving tensor per mixture, all recorded intermediates live on the stats
frame as fingerprints.

No Qwen3 analog exists for these tier-01 rows. Qwen3 runs plain multi-head
attention over a dense FFN. This checkpoint is the f-first 4:1 shape:

- full_attention layers sit at 0, 4, ... 48, the rest slide under window 4096
- layer 0 is the dense prefix (intermediate 3072), the other 48 layers route
- rope applies on the sliding layers plus layer 0, the full routed layers run unrotated

| mixture | row                                                                                      |
| ------- | ---------------------------------------------------------------------------------------- |
| layer0  | decoder layer 0, full attention with forced rope plus the dense prefix block             |
| layer1  | decoder layer 1, sliding attention with rope plus the routed block                       |
| layer4  | decoder layer 4, full attention without rope plus the routed block                       |
| moe     | the routed block surface of layer 1, the router decision rows plus the eager expert loop |

| file                                                           | contents                                        |
| -------------------------------------------------------------- | ----------------------------------------------- |
| layer0-1-4-North-Mini-Code-1.0-00.safetensor                   | layer0.input, layer1.input, layer4.input, moe.h |
| layer0-1-4-North-Mini-Code-1.0-00.safetensor.metadata.json.zst | per-mixture metadata under the mixtures key     |
| layer0-1-4-North-Mini-Code-1.0-00.safetensor.stats.json.zst    | one uniform record per recorded tensor          |

Stats keys carry the mixture-level `layer0.` / `layer1.` / `layer4.` / `moe.`
prefixes:

- recorded expert ids live in the metadata, integer ids carry no stats record
- every routed mixture stands on a margin-clean seed, the seed advances one
  step at a time until the top-k boundary margin clears the 1e-4 floor
  protecting the exact expert-id comparisons in the consuming suite

At seq 6 both mask kinds skip to the sdpa is_causal path (mask None)
and the 4096-token window does not constrain these rows, tier-04 carries
the window behavior.

Consumed by tests/q_bf16/t_bf16_north_01_layer_internals.nim, one
assertion block per mixture.

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_north_01_layer_internals.py

RAM guard:

- the script refuses the weight load when the free+inactive+speculative pool
  sits below 64 GiB
- another python/torch process holding RAM also blocks the run
"""

from collections import OrderedDict
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402, the path insert precedes the import
from safetensors import torch as st  # noqa: E402

import transformers  # noqa: E402
from transformers import Cohere2MoeForCausalLM  # noqa: E402
from transformers.masking_utils import (  # noqa: E402
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.cohere2_moe.modeling_cohere2_moe import (  # noqa: E402
    apply_rotary_pos_emb,
    repeat_kv,
)

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity constants of the checkpoint.
MODEL_NAME = "North-Mini-Code-1.0"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals", f"{MODEL_NAME}-layer-0-1-4"
)
FIXTURE_STEM = "layer0-1-4-" + MODEL_NAME + "-00"
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")

NUM_THREADS = 1

# Recorded rows of the f-first 4:1 shape, verified against the parsed
# layer_types at load time:
# - layer 0, the dense prefix, rope forced
# - layer 1, sliding attention with rope, routed mlp
# - layer 4, full attention without rope, routed mlp
#
# The moe mixture replays the layer-1 routed block surface over a separate
# margin-clean driving input.
DENSE_LAYER_IDX = 0
SLIDING_LAYER_IDX = 1
FULL_LAYER_IDX = 4
MOE_LAYER_IDX = SLIDING_LAYER_IDX

# Per-mixture seeds, independent and order-agnostic. The routed mixtures
# advance their seed one step at a time until the top-k boundary margin
# clears the floor.
SEED_LAYER0 = 311
SEED_LAYER1 = 313
SEED_LAYER4 = 317
SEED_MOE = 319

# Top-k boundary floor of the routed-block seed search, the recorded floor
# of the exact expert-id comparisons in the consuming suite.
MARGIN_FLOOR = 1e-4
MAX_SEED_TRIES = 200

SEQ = 6

MIN_POOL_BYTES = 64 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_north_01_layer_internals] vm_stat gave no page size line")


def pool_bytes() -> int:
    """Free+inactive+speculative physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    wanted = ("Pages free:", "Pages inactive:", "Pages speculative:")
    pool = 0
    for line in out.stdout.splitlines():
        for label in wanted:
            if line.startswith(label):
                pool += int(line.split()[2].rstrip(".")) * page
    if pool == 0:
        raise SystemExit(
            "[gen_bf16_north_01_layer_internals] vm_stat gave no pool lines")
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
    """Refuses the weight load under low memory or a stray python/torch
    process holding RAM.

    Precondition:

    - the free+inactive+speculative pool sits above the 64 GiB floor
    - no other python/torch process holds RAM, the pgrep match excludes
      this process chain (its own command line spells the torch dependency)
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_north_01_layer_internals] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_north_01_layer_internals] other python/torch processes hold RAM: {stray}, "
            "stop and retry when idle")


def load_model() -> Cohere2MoeForCausalLM:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps).

    Returns:
    - the model in eval mode, the recorded decoder layers 0, 1 and 4 plus
      the shared single-theta rotary feed every mixture consumes

    Identity asserts of the checkpoint config, f-first 4:1 shape:

    - full_attention at layers 0 and 4, sliding_attention at layer 1, window 4096
    - the dense prefix at layer 0, intermediate 3072, the routed block everywhere else
    - rope forced on layer 0, applied on the sliding layers, never on the full routed layers

    The sigmoid top-k router carries 128 experts with top 8, no renorm, no shared experts.
    """
    model = Cohere2MoeForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    cfg = model.config
    assert cfg.layer_types[DENSE_LAYER_IDX] == "full_attention", (
        "layer 0 must be the full_attention row of the f-first 4:1 pattern")
    assert cfg.layer_types[SLIDING_LAYER_IDX] == "sliding_attention", (
        "layer 1 must be a sliding_attention row of the f-first 4:1 pattern")
    assert cfg.layer_types[FULL_LAYER_IDX] == "full_attention", (
        "layer 4 must be the full_attention row of the f-first 4:1 pattern")
    assert cfg.mlp_layer_types[DENSE_LAYER_IDX] == "dense", (
        "layer 0 must be the dense prefix row")
    assert all(t == "sparse" for t in cfg.mlp_layer_types[1:]), (
        "every layer past the dense prefix must route")
    assert cfg.prefix_dense_intermediate_size == 3072, (
        "the dense prefix intermediate must stay the checkpoint 3072")
    assert cfg.sliding_window == 4096, (
        "the recorded rows assume the 4096 sliding window")
    assert cfg.rope_parameters["rope_theta"] == 50000, (
        "the rope theta must stay the single global 50000")
    assert cfg.expert_selection_fn == "sigmoid" and cfg.norm_topk_prob is False, (
        "the recorded router is the sigmoid top-k without renorm")
    assert cfg.num_experts == 128 and cfg.num_experts_per_tok == 8, (
        "the recorded rows assume 128 experts with top 8")
    assert cfg.num_shared_experts == 0, (
        "the checkpoint ships no shared experts")
    assert cfg.head_dim == 128 and cfg.num_attention_heads == 32 \
        and cfg.num_key_value_heads == 4, (
        "the recorded rows assume 32 q heads / 4 kv heads over head_dim 128")
    assert cfg.num_hidden_layers == 49 and cfg.hidden_size == 2048 \
        and cfg.vocab_size == 262144, (
        "the recorded rows assume the 49-layer, 2048-hidden, 262144-vocab checkpoint")
    assert model.model.layers[SLIDING_LAYER_IDX].self_attn.force_rope is False, (
        "the routed sliding layer must not force rope")
    assert model.model.layers[DENSE_LAYER_IDX].self_attn.force_rope is True, (
        "the dense prefix layer must force rope under sliding window pattern 1")
    return model


def build_mask(cfg, embeds: torch.Tensor, pos_ids: torch.Tensor,
               layer_type: str):
    """Returns the mask for one layer kind, built through the same masking
    entry points the model forward uses.

    Args:
    - cfg, the parsed config
    - embeds, pos_ids, the bf16 embeds, the matching position ids
    - layer_type, the recorded layer kind

    Returns None when the sdpa path skips to is_causal.
    """
    kwargs = {
        "config": cfg,
        "inputs_embeds": embeds,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": pos_ids,
    }
    if layer_type == "sliding_attention":
        return create_sliding_window_causal_mask(**kwargs)
    return create_causal_mask(**kwargs)


def attention_capture(attn, h_norm: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor, mask, apply_rope: bool) -> dict:
    """Replays the reference Cohere2MoeAttention.forward with intermediate capture,
    asserted equal to the module's own forward before saving.

    Args:
    - attn, the attention module whose forward the replay mirrors
    - h_norm, cos, sin, the post-input-layernorm input and position embeddings
    - mask, apply_rope, the mask row and the rope decision, the sliding
      layers plus the forced-rope dense prefix rotate, the full routed
      layers do not

    Returns:
    - the capture dict, q and k post-rope when rope applies, the expanded kv, the sdpa output and the o_proj output
    """
    seq_len = h_norm.shape[1]
    q = attn.q_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin) if apply_rope else (q, k)
    k_expanded = repeat_kv(k_rot, attn.num_key_value_groups)
    v_expanded = repeat_kv(v, attn.num_key_value_groups)
    sdpa_output = torch.nn.functional.scaled_dot_product_attention(
        q_rot, k_expanded, v_expanded, attn_mask=mask, dropout_p=0.0,
        is_causal=(mask is None and seq_len > 1), scale=attn.scaling)
    attn_output = sdpa_output.transpose(1, 2).contiguous()
    output = attn.o_proj(attn_output.reshape(1, seq_len, -1))
    with torch.no_grad():
        module_output, _ = attn(
            hidden_states=h_norm, position_embeddings=(cos, sin),
            attention_mask=mask, past_key_values=None)
    assert torch.equal(module_output, output), (
        "[gen_bf16_north_01_layer_internals] manual attention replay "
        "diverged from the module forward")
    return {
        "q_rot": q_rot,
        "k_rot": k_rot,
        "v": v,
        "k_expanded": k_expanded,
        "v_expanded": v_expanded,
        "sdpa_output": sdpa_output,
        "attn_output": output,
    }


def layer_chain_capture(layer, x: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, mask, pos_ids: torch.Tensor,
                        apply_rope: bool) -> dict:
    """Runs the full decoder-layer chain over one input, the manual replay
    asserted equal to the module forward before saving.

    The parallel block consumes ONE input_layernorm output, the attention
    mixer and the mlp block both read it, the layer output adds all three streams.

    Args:
    - layer, the decoder layer whose forward the replay mirrors
    - x, pos_ids, the bf16 input and its position ids
    - cos, sin, mask, apply_rope, the position embeddings, the layer-kind mask
      and the rope decision (sliding layers plus the forced-rope dense prefix)

    Returns:
    - the capture dict, the norm output, the attention and mlp outputs, the layer output, plus the attention-level capture
    """
    with torch.no_grad():
        h_norm = layer.input_layernorm(x)
        cap = attention_capture(layer.self_attn, h_norm, cos, sin, mask, apply_rope)
        mlp_out = layer.mlp(h_norm)
        layer_out = x + cap["attn_output"] + mlp_out
        module_out = layer(
            x, position_embeddings=(cos, sin), attention_mask=mask,
            position_ids=pos_ids, past_key_values=None, use_cache=False)
    assert torch.equal(module_out, layer_out), (
        "[gen_bf16_north_01_layer_internals] manual layer chain diverged "
        "from the module forward")
    cap.update({
        "layer.input_layernorm_output": h_norm,
        "layer.mlp_output": mlp_out,
        "layer.layer_output": layer_out,
    })
    return cap


def attention_stats_prefix(cap: dict, prefix: str, cos: torch.Tensor,
                           sin: torch.Tensor, apply_rope: bool) -> dict:
    """Namespaced stats entries of one attention capture, the rope rows
    included when the layer rotates, unrotated layers record no rope rows."""
    entries = {
        prefix + "q_rot": cap["q_rot"],
        prefix + "k_rot": cap["k_rot"],
        prefix + "v": cap["v"],
        prefix + "k_expanded": cap["k_expanded"],
        prefix + "v_expanded": cap["v_expanded"],
        prefix + "sdpa_output": cap["sdpa_output"],
        prefix + "attn_output": cap["attn_output"],
    }
    if apply_rope:
        entries[prefix + "cos"] = cos
        entries[prefix + "sin"] = sin
    return entries


def topk_boundary_margin(router_logits: torch.Tensor, top_k: int) -> float:
    """Smallest gap between the Kth and the K+1th sorted sigmoid scores,
    minimized over all rows, the exact-index condition of the top-k selection."""
    scores = router_logits.float().sigmoid()
    sorted_scores = scores.sort(dim=-1, descending=True).values
    gaps = sorted_scores[:, top_k - 1] - sorted_scores[:, top_k]
    return gaps.min().item()


def margin_clean_input(seed: int, hidden_size: int, routed_block) -> tuple:
    """Builds a margin-clean routed-block input, the seed advancing one step
    at a time until the top-k boundary margin clears the recorded floor.

    Args:
    - seed, hidden_size, the first seed tried and the checkpoint width
    - routed_block, the weighted routed block, its sigmoid router decides

    Returns:
    - the bf16 input, the advancing seed, the achieved margin
    """
    top_k = routed_block.gate.top_k
    margin = -1.0
    for _ in range(MAX_SEED_TRIES):
        gen = torch.Generator(device="mps")
        gen.manual_seed(seed)
        h = torch.randn(1, SEQ, hidden_size, generator=gen, dtype=torch.bfloat16,
                        device="mps")
        with torch.no_grad():
            router_logits, _, _ = routed_block.gate(h.view(-1, hidden_size))
        margin = topk_boundary_margin(router_logits, top_k)
        if margin > MARGIN_FLOOR:
            return h, seed, margin
        seed += 1
    raise SystemExit(
        "[gen_bf16_north_01_layer_internals] no margin-clean seed found "
        "for the routed block")


def routed_rows(routed_block, h: torch.Tensor, margin: float, seed: int) -> tuple:
    """Runs the routed block over one margin-clean input, the sigmoid
    weights self-checked against the selected logits before saving.

    Args:
    - routed_block, the weighted routed block, its sigmoid router plus eager expert loop
    - h, the margin-clean bf16 input
    - margin, seed, the achieved boundary margin and its seed

    Returns:
    - meta, the routed-decision metadata rows
    - captured, the stats-frame entries with the expert ids in the metadata
      (integer ids carry no stats record)
    """
    flat = h.view(-1, h.shape[-1])
    with torch.no_grad():
        router_logits, router_scores, selected_experts = routed_block.gate(flat)
        moe_output = routed_block(h)

    # Self-check before saving, the checkpoint renorm stays off so the recorded
    # weights stay the sigmoid of the selected logits, both sides on the bf16
    # grid the reference router computed.
    selected_logits = router_logits.gather(1, selected_experts)
    assert torch.equal(router_scores, selected_logits.sigmoid()), (
        "[gen_bf16_north_01_layer_internals] routing weights diverge from "
        "the sigmoid of the selected logits")
    meta = {
        "seed": seed,
        "margin_floor": MARGIN_FLOOR,
        "topk_boundary_margin": margin,
        "router": "sigmoid_topk",
        "num_experts": routed_block.gate.num_experts,
        "num_experts_per_tok": routed_block.gate.top_k,
        "norm_topk_prob": routed_block.gate.norm_topk_prob,
        "topk_indices": selected_experts.tolist(),
        "weights_dtype": str(router_scores.dtype).replace("torch.", ""),
        "logits_dtype": str(router_logits.dtype).replace("torch.", ""),
        "indices_dtype": str(selected_experts.dtype).replace("torch.", ""),
        "flip_budget_policy": "exact expert-id comparisons in the consuming "
            "suites stand on a recorded positive boundary margin with the "
            "1e-4 floor, a row below the floor would carry an explicit "
            "per-mixture recorded exception, never a silent widening",
    }
    captured = {
        "router_logits": router_logits,
        "topk_weights": router_scores,
        "moe_output": moe_output,
    }
    return meta, captured


def generate_layer0_mixture(model: Cohere2MoeForCausalLM, cfg) -> tuple:
    """Records the layer-0 row, full attention with forced rope plus the dense
    prefix block over one seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under layer0.
    """
    layer = model.model.layers[DENSE_LAYER_IDX]
    gen = torch.Generator(device="mps")
    gen.manual_seed(SEED_LAYER0)
    x = torch.randn(1, SEQ, cfg.hidden_size, generator=gen, dtype=torch.bfloat16,
                    device="mps")
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids)
    mask = build_mask(cfg, x, pos_ids, cfg.layer_types[DENSE_LAYER_IDX])
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids, apply_rope=True)
    meta = {
        "case": "prefill_seq6",
        "layer": f"model.layers.{DENSE_LAYER_IDX}",
        "layer_type": cfg.layer_types[DENSE_LAYER_IDX],
        "mlp_type": cfg.mlp_layer_types[DENSE_LAYER_IDX],
        "rope_applied": True,
        "seq_len": SEQ,
        "seed": SEED_LAYER0,
        "softmax_scaling": float(layer.self_attn.scaling),
        "rope_theta": cfg.rope_parameters["rope_theta"],
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "rope_output_layout": "the interleaved spelling (repeat_interleave "
            "freqs, even/odd rotate_half), the layout the recorded rotation "
            "consumed",
        "mask_note": "seq 6 stays inside the 4096 window, both mask kinds "
            "skip to the sdpa is_causal path (mask None), the window behavior "
            "is recorded at tier 04",
    }
    payload = OrderedDict([("layer0.input", x)])
    captured = attention_stats_prefix(cap, "layer0.", cos, sin, apply_rope=True)
    captured.update({
        "layer0.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "layer0.layer.mlp_output": cap["layer.mlp_output"],
        "layer0.layer.layer_output": cap["layer.layer_output"],
    })
    print("[gen_bf16_north_01_layer_internals] layer0 mixture "
          f"(seed {SEED_LAYER0})")
    return meta, payload, captured


def generate_layer1_mixture(model: Cohere2MoeForCausalLM, cfg) -> tuple:
    """Records the layer-1 row, sliding attention with rope plus the routed
    block over one margin-clean seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under layer1.
    """
    layer = model.model.layers[SLIDING_LAYER_IDX]
    x, seed, margin = margin_clean_input(
        SEED_LAYER1, cfg.hidden_size, layer.mlp)
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids)
    mask = build_mask(cfg, x, pos_ids, cfg.layer_types[SLIDING_LAYER_IDX])
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids, apply_rope=True)
    routed_meta, routed_captured = routed_rows(layer.mlp, x, margin, seed)
    meta = {
        "case": "prefill_seq6_margin_clean",
        "layer": f"model.layers.{SLIDING_LAYER_IDX}",
        "layer_type": cfg.layer_types[SLIDING_LAYER_IDX],
        "mlp_type": cfg.mlp_layer_types[SLIDING_LAYER_IDX],
        "rope_applied": True,
        "seq_len": SEQ,
        "seed": seed,
        "softmax_scaling": float(layer.self_attn.scaling),
        "rope_theta": cfg.rope_parameters["rope_theta"],
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "rope_output_layout": "the interleaved spelling (repeat_interleave "
            "freqs, even/odd rotate_half), the layout the recorded rotation "
            "consumed",
        "mask_note": "seq 6 stays inside the 4096 window, the sliding mask "
            "skips to the sdpa is_causal path (mask None), the window behavior "
            "is recorded at tier 04",
        "moe": routed_meta,
    }
    payload = OrderedDict([("layer1.input", x)])
    captured = attention_stats_prefix(cap, "layer1.", cos, sin, apply_rope=True)
    captured.update({
        "layer1.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "layer1.layer.mlp_output": cap["layer.mlp_output"],
        "layer1.layer.layer_output": cap["layer.layer_output"],
        "layer1.moe.router_logits": routed_captured["router_logits"],
        "layer1.moe.topk_weights": routed_captured["topk_weights"],
        "layer1.moe.moe_output": routed_captured["moe_output"],
    })
    print(f"[gen_bf16_north_01_layer_internals] layer1 mixture (seed {seed}): "
          f"topk margin {margin:.3e}")
    return meta, payload, captured


def generate_layer4_mixture(model: Cohere2MoeForCausalLM, cfg) -> tuple:
    """Records the layer-4 row, full attention without rope plus the routed
    block over one margin-clean seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under layer4.
    """
    layer = model.model.layers[FULL_LAYER_IDX]
    x, seed, margin = margin_clean_input(
        SEED_LAYER4, cfg.hidden_size, layer.mlp)
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids)
    mask = build_mask(cfg, x, pos_ids, cfg.layer_types[FULL_LAYER_IDX])
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids, apply_rope=False)
    routed_meta, routed_captured = routed_rows(layer.mlp, x, margin, seed)
    meta = {
        "case": "prefill_seq6_margin_clean",
        "layer": f"model.layers.{FULL_LAYER_IDX}",
        "layer_type": cfg.layer_types[FULL_LAYER_IDX],
        "mlp_type": cfg.mlp_layer_types[FULL_LAYER_IDX],
        "rope_applied": False,
        "seq_len": SEQ,
        "seed": seed,
        "softmax_scaling": float(layer.self_attn.scaling),
        "rope_theta": cfg.rope_parameters["rope_theta"],
        "rope_note": "the full routed layers run unrotated, the position "
            "embeddings exist but the module never consumes them, no rope "
            "rows are recorded",
        "mask_note": "seq 6 stays inside the 4096 window, both mask kinds "
            "skip to the sdpa is_causal path (mask None), the window behavior "
            "is recorded at tier 04",
        "moe": routed_meta,
    }
    payload = OrderedDict([("layer4.input", x)])
    captured = attention_stats_prefix(cap, "layer4.", cos, sin, apply_rope=False)
    captured.update({
        "layer4.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "layer4.layer.mlp_output": cap["layer.mlp_output"],
        "layer4.layer.layer_output": cap["layer.layer_output"],
        "layer4.moe.router_logits": routed_captured["router_logits"],
        "layer4.moe.topk_weights": routed_captured["topk_weights"],
        "layer4.moe.moe_output": routed_captured["moe_output"],
    })
    print(f"[gen_bf16_north_01_layer_internals] layer4 mixture (seed {seed}): "
          f"topk margin {margin:.3e}")
    return meta, payload, captured


def generate_moe_mixture(model: Cohere2MoeForCausalLM, cfg) -> tuple:
    """Records the routed-block surface of layer 1, the router decision rows
    plus the eager expert loop over one margin-clean input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under moe.
    """
    routed_block = model.model.layers[MOE_LAYER_IDX].mlp
    h, seed, margin = margin_clean_input(SEED_MOE, cfg.hidden_size, routed_block)
    routed_meta, routed_captured = routed_rows(routed_block, h, margin, seed)
    meta = {
        "case": "routed_block_seq6_margin_clean",
        "layer": f"model.layers.{MOE_LAYER_IDX}.mlp",
        "seq_len": SEQ,
        "hidden_dtype": "bfloat16",
        "expert_loop": "the grouped_mm per-pair spelling, expert-sorted "
            "pairs with a bf16 weight product per pair and the token sum "
            "in f32 with a single rounding",
        "flip_budget_policy": routed_meta["flip_budget_policy"],
        "margin_floor": MARGIN_FLOOR,
        "topk_boundary_margin": margin,
        "seed": seed,
        "router": "sigmoid_topk",
        "num_experts": routed_meta["num_experts"],
        "num_experts_per_tok": routed_meta["num_experts_per_tok"],
        "norm_topk_prob": routed_meta["norm_topk_prob"],
        "topk_indices": routed_meta["topk_indices"],
        "weights_dtype": routed_meta["weights_dtype"],
        "logits_dtype": routed_meta["logits_dtype"],
        "indices_dtype": routed_meta["indices_dtype"],
    }
    payload = OrderedDict([("moe.h", h)])
    captured = {
        "moe.router_logits": routed_captured["router_logits"],
        "moe.topk_weights": routed_captured["topk_weights"],
        "moe.moe_output": routed_captured["moe_output"],
    }
    print(f"[gen_bf16_north_01_layer_internals] moe mixture (seed {seed}): "
          f"topk margin {margin:.3e}")
    return meta, payload, captured


def write_metadata_zst(path: str, metadata: dict) -> None:
    """Writes one metadata sidecar, pretty JSON inside one zstd frame, level
    19 with content size and checksum recorded in the frame header."""
    import compression.zstd

    zstd_options = {
        compression.zstd.CompressionParameter.compression_level: 19,
        compression.zstd.CompressionParameter.content_size_flag: 1,
        compression.zstd.CompressionParameter.checksum_flag: 1,
    }
    payload_json = json.dumps(
        metadata, sort_keys=True, indent=2, ensure_ascii=True
    ).encode("utf-8") + b"\n"
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload_json, options=zstd_options))


def save_fixture(metadata: dict, mixtures: list) -> None:
    """Writes the single-file fixture set, one safetensors payload carrying
    the four bare named driving tensors, one metadata sidecar and one stats
    sidecar over every recorded tensor.

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

    write_stats_file(FIXTURE_PATH + ".stats.json.zst",
                     FIXTURE_STEM + ".safetensor", stats_entries)
    write_metadata_zst(FIXTURE_PATH + ".metadata.json.zst", metadata)


def main() -> None:
    """Records the layer-internals fixture file set after the RAM guard."""
    recorded_from = os.environ.get("TTT_RECORD_FROM", "m4max-metal")
    check_ram()

    model = load_model()
    cfg = model.config

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    layer0_meta, layer0_payload, layer0_captured = generate_layer0_mixture(model, cfg)
    layer1_meta, layer1_payload, layer1_captured = generate_layer1_mixture(model, cfg)
    layer4_meta, layer4_payload, layer4_captured = generate_layer4_mixture(model, cfg)
    moe_meta, moe_payload, moe_captured = generate_moe_mixture(model, cfg)

    metadata = {
        "model": MODEL_NAME,
        "file": FIXTURE_STEM + ".safetensor",
        "dtype": "bfloat16",
        "num_threads": NUM_THREADS,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "recorded_from": recorded_from,
        "device": "mps",
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "sliding_window": cfg.sliding_window,
        "attention_shape": "f-first 4:1, full_attention at layers 0, 4, "
            "... 48 and sliding_attention between, window 4096, the dense "
            "prefix at layer 0 routes nowhere, the other 48 layers route",
        "rope_parameters": cfg.rope_parameters,
        "layer_types": cfg.layer_types,
        "mlp_layer_types": cfg.mlp_layer_types,
        "prefix_dense_intermediate_size": cfg.prefix_dense_intermediate_size,
        "mixtures": {
            "layer0": layer0_meta,
            "layer1": layer1_meta,
            "layer4": layer4_meta,
            "moe": moe_meta,
        },
    }
    save_fixture(metadata, [
        (layer0_payload, layer0_captured),
        (layer1_payload, layer1_captured),
        (layer4_payload, layer4_captured),
        (moe_payload, moe_captured),
    ])

    print(f"[gen_bf16_north_01_layer_internals] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_north_01_layer_internals] wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
