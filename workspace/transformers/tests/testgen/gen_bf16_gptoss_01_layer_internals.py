#!/usr/bin/env python3
"""Layer-internals fixture file of the gpt-oss-20b checkpoint, recorded with torch bf16
on Metal (mps) under the installed reference modeling.

Fixture-only family, the Nim implementation stays parked. These rows carry
the reference surfaces a later port consumes, no consumer exists yet.

Single-file grammar, one fixture file per family layer group, one bare bf16
driving tensor per mixture, all recorded intermediates live on the stats
frame as fingerprints.

No Qwen3 analog exists for these tier-01 rows. Qwen3 runs plain softmax
attention without sinks and a dense FFN. This checkpoint is the 1:1
alternating shape with sink attention and a routed block on every layer:

- sliding_attention layers sit at 0, 2, ... 22, the full_attention layers
  sit between, window 128, 64 q heads over 8 kv heads, head_dim 64
- every attention head carries a sink row, the softmax runs over the keys
  plus the sink column and drops the sink before the value mix
- every layer routes, 32 experts with top 4, softmax over the top-4 logits
  under weights dequantized from MXFP4 to bf16 at load, the reference
  quantizer runs the dequantize (mps has no MXFP4 kernel)

| mixture | row                                                                                      |
| ------- | ---------------------------------------------------------------------------------------- |
| layer0  | decoder layer 0, sliding attention with the sink rows plus the routed block              |
| layer1  | decoder layer 1, full attention with the sink rows plus the routed block                 |
| moe     | the routed block surface of layer 0, the router decision rows plus the eager expert loop |

| file                                                 | contents                                    |
| ---------------------------------------------------- | ------------------------------------------- |
| layer0-1-gpt-oss-20b-00.safetensor                   | layer0.input, layer1.input, moe.h           |
| layer0-1-gpt-oss-20b-00.safetensor.metadata.json.zst | per-mixture metadata under the mixtures key |
| layer0-1-gpt-oss-20b-00.safetensor.stats.json.zst    | one uniform record per recorded tensor      |

Stats keys carry the mixture-level `layer0.` / `layer1.` / `moe.` prefixes.
Sink-aware rows per layer:

- sinks, the per-head sink row the softmax denominator includes
- attn_output_with_sink, the module output through the sink softmax
- attn_output_without_sink, the same path over a keys-only softmax showing
  what the sink column moves

The sink_log_delta row carries the log-space relative softmax-denominator
delta (sink - m) - logsumexp(attn_weights - m) per head and position,
computed f64 on cpu and recorded f32 (mps has no f64 storage).

A negative row means the sink raises the denominator and moves probability
mass off the keys.

- recorded expert ids live in the metadata, integer ids carry no stats record
- every routed mixture stands on a margin-clean seed, the seed advances one
  step at a time until the top-k boundary margin clears the 1e-4 floor
  protecting the exact expert-id comparisons in the consuming suite

At seq 6 both mask kinds stay inside the 128-token window and the sliding
mask carries no window cutoff, tier-04 carries the window behavior.

Consumed by tests/q_bf16/t_bf16_gptoss_01_layer_internals.nim, one assertion
block per mixture
(the consumer lands with the port, the implementation stays parked).

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_gptoss_01_layer_internals.py

RAM guard:

- the script refuses the weight load when the free+inactive+speculative pool
  sits below 64 GiB (the dequantized weights sit near 40 GiB)
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
from transformers import GptOssForCausalLM  # noqa: E402
from transformers.models.gpt_oss.modeling_gpt_oss import (  # noqa: E402
    apply_rotary_pos_emb,
    repeat_kv,
)

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity constants of the checkpoint.
MODEL_NAME = "gpt-oss-20b"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals", f"{MODEL_NAME}-layer-0-1"
)
FIXTURE_STEM = "layer0-1-" + MODEL_NAME + "-00"
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")

NUM_THREADS = 1

# Recorded rows of the 1:1 alternating shape, verified against the parsed
# layer_types at load time:
# - layer 0, sliding attention under window 128
# - layer 1, full attention
#
# The moe mixture replays the layer-0 routed block surface over a separate
# margin-clean driving input.
SLIDING_LAYER_IDX = 0
FULL_LAYER_IDX = 1
MOE_LAYER_IDX = SLIDING_LAYER_IDX

# Per-mixture seeds, independent and order-agnostic. The routed mixtures
# advance their seed one step at a time until the top-k boundary margin
# clears the floor.
SEED_LAYER0 = 351
SEED_LAYER1 = 353
SEED_MOE = 357

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
        "[gen_bf16_gptoss_01_layer_internals] vm_stat gave no page size line")


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
            "[gen_bf16_gptoss_01_layer_internals] vm_stat gave no pool lines")
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
      (the dequantized weights sit near 40 GiB)
    - no other python/torch process holds RAM, the pgrep match excludes
      this process chain (its own command line spells the torch dependency)
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_gptoss_01_layer_internals] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_gptoss_01_layer_internals] other python/torch processes "
            f"hold RAM: {stray}, stop and retry when idle")


def load_model() -> GptOssForCausalLM:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps), transformers-first over the MXFP4 checkpoint.

    Returns:
    - the model in eval mode, the recorded decoder layers 0 and 1 sharing
      the single-theta rotary feed every mixture consumes

    Identity asserts of the checkpoint config, 1:1 alternating shape:

    - sliding_attention at layer 0, full_attention at layer 1, window 128,
      64 q heads over 8 kv heads, head_dim 64, a sink row per q head
    - the eager attention path (the sink softmax has no sdpa spelling)
    - the routed block on every layer, 32 experts with top 4, softmax over
      the top-4 logits

    The MXFP4 checkpoint weights dequantize to bf16 through the reference
    quantizer at load, the recorded tensors carry the bf16 rows.
    """
    model = GptOssForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    cfg = model.config
    assert cfg.layer_types[SLIDING_LAYER_IDX] == "sliding_attention", (
        "layer 0 must be a sliding_attention row of the 1:1 pattern")
    assert cfg.layer_types[FULL_LAYER_IDX] == "full_attention", (
        "layer 1 must be the full_attention row of the 1:1 pattern")
    assert cfg.sliding_window == 128, (
        "the recorded rows assume the 128 sliding window")
    assert cfg.head_dim == 64 and cfg.num_attention_heads == 64 \
        and cfg.num_key_value_heads == 8, (
        "the recorded rows assume 64 q heads over 8 kv heads, head_dim 64")
    assert cfg._attn_implementation == "eager", (
        "the sink softmax runs the eager attention path")
    assert cfg.num_local_experts == 32 and cfg.num_experts_per_tok == 4, (
        "the recorded rows assume 32 experts with top 4")
    assert cfg.intermediate_size == 2880, (
        "the recorded rows assume the expert intermediate 2880")
    assert cfg.num_hidden_layers == 24 and cfg.hidden_size == 2880 \
        and cfg.vocab_size == 201088, (
        "the recorded rows assume the 24-layer, 2880-hidden, 201088-vocab checkpoint")
    assert model.model.layers[SLIDING_LAYER_IDX].self_attn.sinks.shape == (64,), (
        "every q head must carry one sink row")
    return model


def build_mask(cfg, embeds: torch.Tensor, pos_ids: torch.Tensor,
               layer_type: str):
    """Returns the mask for one layer kind, built through the same masking
    entry points the model forward uses.

    Args:
    - cfg, the parsed config
    - embeds, pos_ids, the bf16 embeds, the matching position ids
    - layer_type, the recorded layer kind

    Returns None when the mask builder skips to the bare attention scores.
    """
    kwargs = {
        "config": cfg,
        "inputs_embeds": embeds,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": pos_ids,
    }
    if layer_type == "sliding_attention":
        return create_sliding_window_causal_mask_local(cfg, embeds, pos_ids)
    return create_causal_mask_local(cfg, embeds, pos_ids)


def create_sliding_window_causal_mask_local(cfg, embeds, pos_ids):
    """Sliding-window causal mask through the shared masking entry point."""
    from transformers.masking_utils import create_sliding_window_causal_mask

    return create_sliding_window_causal_mask(
        config=cfg, inputs_embeds=embeds, attention_mask=None,
        past_key_values=None, position_ids=pos_ids)


def create_causal_mask_local(cfg, embeds, pos_ids):
    """Causal mask through the shared masking entry point."""
    from transformers.masking_utils import create_causal_mask

    return create_causal_mask(
        config=cfg, inputs_embeds=embeds, attention_mask=None,
        past_key_values=None, position_ids=pos_ids)


def sink_log_delta(aw: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """Log-space relative softmax-denominator delta of the sink column.

    Args:
    - aw, the pre-mask attention score rows (keys only)
    - sinks, the per-head sink row expanded over positions, both bf16

    Returns:
    - the (sink - m) - logsumexp(aw - m) rows per head and position, computed
      f64 on cpu (mps has no f64 storage) and returned f32, a row below 0
      means the sink raises the softmax denominator over the keys alone

    The raw exp ratio of the denominators overflows f32 on sink-dominated heads, the log
    space keeps the delta finite and comparable.

    The delta takes the pre-mask scores (keys only), masked-position deltas
    over the float-min mask entries would leave the f32 binade range.
    """
    combined = torch.cat([aw, sinks], dim=-1)
    m = combined.max(dim=-1, keepdim=True).values
    aw_cpu = aw.float().cpu().double() - m.float().cpu().double()
    sink_cpu = sinks.float().cpu().double() - m.float().cpu().double()
    delta = sink_cpu[..., -1] - torch.logsumexp(aw_cpu, dim=-1)
    return delta.to(torch.float32).to(aw.device)


def attention_capture(attn, h_norm: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor, mask) -> dict:
    """Replays the reference GptOssAttention.forward with intermediate capture, the with-sink
    output asserted equal to the module forward.

    Args:
    - attn, the attention module whose forward the replay mirrors
    - h_norm, cos, sin, mask, the post-input-layernorm input, the position
      embeddings and the causal mask
      (a tensor on the eager path, the scores add it before the softmax)

    Returns:
    - the capture dict, q and k post-rope, the expanded kv, the pre-mask
      score rows, the sink rows, the sink log delta and both outputs

    The recorded score rows carry the pre-mask scores, the module adds
    the causal mask before the softmax and the float-min masked entries
    sit outside the stats binade range.
    """
    seq_len = h_norm.shape[1]
    hidden_shape = (1, seq_len, -1, attn.head_dim)
    q = attn.q_proj(h_norm).view(hidden_shape).transpose(1, 2)
    k = attn.k_proj(h_norm).view(hidden_shape).transpose(1, 2)
    v = attn.v_proj(h_norm).view(hidden_shape).transpose(1, 2)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    k_expanded = repeat_kv(k_rot, attn.num_key_value_groups)
    v_expanded = repeat_kv(v, attn.num_key_value_groups)
    aw_raw = torch.matmul(q_rot, k_expanded.transpose(2, 3)) * attn.scaling
    aw = aw_raw + mask if mask is not None else aw_raw
    sinks = attn.sinks.reshape(1, -1, 1, 1).expand(
        1, -1, seq_len, -1)
    combined = torch.cat([aw, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(combined, dim=-1,
                                        dtype=combined.dtype)
    scores = probs[..., :-1]
    attn_output = torch.matmul(scores.to(v_expanded.dtype), v_expanded)
    attn_output = attn_output.transpose(1, 2).contiguous()
    with_sink_output = attn.o_proj(attn_output.reshape(1, seq_len, -1))
    with torch.no_grad():
        module_output, _ = attn(
            hidden_states=h_norm, position_embeddings=(cos, sin),
            attention_mask=mask, past_key_values=None)
    assert torch.equal(module_output, with_sink_output), (
        "[gen_bf16_gptoss_01_layer_internals] manual sink attention replay "
        "diverged from the module forward")

    # Keys-only softmax, the same recipe minus the sink column, the discrimination row
    # that shows what the sink moves.
    probs_no = torch.nn.functional.softmax(
        aw - aw.max(dim=-1, keepdim=True).values, dim=-1, dtype=aw.dtype)
    scores_no = probs_no.to(v_expanded.dtype)
    attn_output_no = torch.matmul(scores_no, v_expanded)
    attn_output_no = attn_output_no.transpose(1, 2).contiguous()
    without_sink_output = attn.o_proj(attn_output_no.reshape(1, seq_len, -1))
    return {
        "q_rot": q_rot,
        "k_rot": k_rot,
        "v": v,
        "k_expanded": k_expanded,
        "v_expanded": v_expanded,
        "aw_logits": aw_raw,
        "sinks": attn.sinks,
        "sink_log_delta": sink_log_delta(aw, sinks),
        "attn_output_with_sink": with_sink_output,
        "attn_output_without_sink": without_sink_output,
    }


def layer_chain_capture(layer, x: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, mask, pos_ids: torch.Tensor) -> dict:
    """Runs the full decoder-layer chain over one input, the manual replay
    asserted equal to the module forward before saving.

    The sequential block runs input_layernorm, the attention mixer, one
    residual add, post_attention_layernorm, the routed block and one closing
    residual add before the layer returns.

    Args:
    - layer, the decoder layer whose forward the replay mirrors
    - x, pos_ids, the bf16 input and its position ids
    - cos, sin, mask, the position embeddings and the layer-kind mask

    Returns:
    - the capture dict, the norm outputs, the attention rows, the routed
      block rows and the layer output, plus the attention-level capture
    """
    with torch.no_grad():
        h_norm = layer.input_layernorm(x)
        cap = attention_capture(layer.self_attn, h_norm, cos, sin, mask)
        h1 = x + cap["attn_output_with_sink"]
        h2 = layer.post_attention_layernorm(h1)
        mlp_out, router_scores = layer.mlp(h2)
        layer_out = h1 + mlp_out
        module_out = layer(
            x, attention_mask=mask, position_ids=pos_ids,
            past_key_values=None, use_cache=False,
            position_embeddings=(cos, sin))
    assert torch.equal(module_out, layer_out), (
        "[gen_bf16_gptoss_01_layer_internals] manual layer chain diverged "
        "from the module forward")
    cap.update({
        "layer.input_layernorm_output": h_norm,
        "layer.post_attention_layernorm_output": h1,
        "layer.mlp_output": mlp_out,
        "layer.layer_output": layer_out,
    })
    return cap


def attention_stats_prefix(cap: dict, prefix: str, cos: torch.Tensor,
                           sin: torch.Tensor) -> dict:
    """Namespaced stats entries of one attention capture, the rope rows
    and the sink-aware rows included."""
    return {
        prefix + "cos": cos,
        prefix + "sin": sin,
        prefix + "q_rot": cap["q_rot"],
        prefix + "k_rot": cap["k_rot"],
        prefix + "v": cap["v"],
        prefix + "k_expanded": cap["k_expanded"],
        prefix + "v_expanded": cap["v_expanded"],
        prefix + "aw_logits": cap["aw_logits"],
        prefix + "sinks": cap["sinks"],
        prefix + "sink_log_delta": cap["sink_log_delta"],
        prefix + "attn_output_with_sink": cap["attn_output_with_sink"],
        prefix + "attn_output_without_sink": cap["attn_output_without_sink"],
    }


def topk_boundary_margin(router_logits: torch.Tensor, top_k: int) -> float:
    """Smallest gap between the Kth and the K+1th sorted router logits,
    minimized over all rows, the exact-index condition of the top-k
    selection (the router ranks the raw logits).

    Returns:
    - the margin as a float, the minimum gap over all router rows
    """
    sorted_logits = router_logits.sort(dim=-1, descending=True).values
    gaps = sorted_logits[:, top_k - 1] - sorted_logits[:, top_k]
    return gaps.min().item()


def margin_clean_input(seed: int, hidden_size: int, routed_block) -> tuple:
    """Builds a margin-clean routed-block input, the seed advancing one step
    at a time until the top-k boundary margin clears the floor.

    Args:
    - seed, hidden_size, the first seed tried and the checkpoint width
    - routed_block, the weighted routed block, its router ranks the logits

    Returns:
    - the bf16 input, the advancing seed, the achieved margin
    """
    top_k = routed_block.router.top_k
    margin = -1.0
    for _ in range(MAX_SEED_TRIES):
        gen = torch.Generator(device="mps")
        gen.manual_seed(seed)
        h = torch.randn(1, SEQ, hidden_size, generator=gen, dtype=torch.bfloat16,
                        device="mps")
        with torch.no_grad():
            router_logits, _, _ = routed_block.router(h.view(-1, hidden_size))
        margin = topk_boundary_margin(router_logits, top_k)
        if margin > MARGIN_FLOOR:
            return h, seed, margin
        seed += 1
    raise SystemExit(
        "[gen_bf16_gptoss_01_layer_internals] no margin-clean seed found "
        "for the routed block")


def routed_rows(routed_block, h: torch.Tensor, margin: float, seed: int) -> tuple:
    """Runs the routed block over one margin-clean input, the softmax scores
    and the eager expert loop self-checked before saving.

    Args:
    - routed_block, the weighted routed block, its top-4 softmax router plus
      the eager per-expert loop
    - h, the margin-clean bf16 input
    - margin, seed, the achieved boundary margin and its seed

    Returns:
    - meta, the routed-decision metadata rows
    - captured, the stats-frame entries with the expert ids in the metadata
      (integer ids carry no stats record)
    """
    flat = h.view(-1, h.shape[-1])
    with torch.no_grad():
        router_logits, router_scores, selected_experts = routed_block.router(flat)
        moe_output = routed_block(h)[0]
        experts_output = routed_block.experts(flat, selected_experts,
                                              router_scores)

    # Self-check before saving:
    # - the scores stay the softmax over the selected top-4 logits
    # - the block output stays the eager expert loop output, recomputed
    #   through the same op chain the module ran
    top_vals = router_logits.gather(1, selected_experts)
    assert torch.equal(router_scores, torch.nn.functional.softmax(
        top_vals, dim=1, dtype=top_vals.dtype)), (
        "[gen_bf16_gptoss_01_layer_internals] routing scores diverge from "
        "the softmax over the selected top-4 logits")
    assert torch.equal(moe_output.view(flat.shape), experts_output), (
        "[gen_bf16_gptoss_01_layer_internals] block output diverges from the "
        "eager expert loop output")
    meta = {
        "seed": seed,
        "margin_floor": MARGIN_FLOOR,
        "topk_boundary_margin": margin,
        "router": "softmax_topk",
        "num_experts": routed_block.router.num_experts,
        "num_experts_per_tok": routed_block.router.top_k,
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
        "moe_output": moe_output.view(flat.shape),
        "router_weight": routed_block.router.weight,
        "router_bias": routed_block.router.bias,
    }
    return meta, captured


def generate_layer0_mixture(model: GptOssForCausalLM, cfg) -> tuple:
    """Records the layer-0 row, sliding attention with the sink rows plus the routed block over
    one margin-clean seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under layer0.
    """
    layer = model.model.layers[SLIDING_LAYER_IDX]
    x, seed, margin = margin_clean_input(
        SEED_LAYER0, cfg.hidden_size, layer.mlp)
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids)
    mask = build_mask(cfg, x, pos_ids, cfg.layer_types[SLIDING_LAYER_IDX])
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids)
    routed_meta, routed_captured = routed_rows(layer.mlp, x, margin, seed)
    meta = {
        "case": "prefill_seq6_margin_clean",
        "layer": f"model.layers.{SLIDING_LAYER_IDX}",
        "layer_type": cfg.layer_types[SLIDING_LAYER_IDX],
        "seq_len": SEQ,
        "seed": seed,
        "softmax_scaling": float(layer.self_attn.scaling),
        "sink_rows": "one sink row per q head, the softmax runs over the "
            "keys plus the sink column and drops the sink before the value "
            "mix, sink_log_delta is the log-space relative denominator delta "
            "computed f64 on cpu and recorded f32",
        "mask_note": "seq 6 stays inside the 128 window, the sliding mask "
            "carries no window cutoff at this length, the window behavior "
            "is recorded at tier 04",
        "moe": routed_meta,
    }
    payload = OrderedDict([("layer0.input", x)])
    captured = attention_stats_prefix(cap, "layer0.", cos, sin)
    captured.update({
        "layer0.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "layer0.layer.post_attention_layernorm_output":
            cap["layer.post_attention_layernorm_output"],
        "layer0.layer.mlp_output": cap["layer.mlp_output"],
        "layer0.layer.layer_output": cap["layer.layer_output"],
        "layer0.moe.router_logits": routed_captured["router_logits"],
        "layer0.moe.topk_weights": routed_captured["topk_weights"],
        "layer0.moe.moe_output": routed_captured["moe_output"],
    })
    print(f"[gen_bf16_gptoss_01_layer_internals] layer0 mixture (seed {seed}): "
          f"topk margin {margin:.3e}")
    return meta, payload, captured


def generate_layer1_mixture(model: GptOssForCausalLM, cfg) -> tuple:
    """Records the layer-1 row, full attention with the sink rows plus the routed block over
    one margin-clean seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under layer1.
    """
    layer = model.model.layers[FULL_LAYER_IDX]
    x, seed, margin = margin_clean_input(
        SEED_LAYER1, cfg.hidden_size, layer.mlp)
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids)
    mask = build_mask(cfg, x, pos_ids, cfg.layer_types[FULL_LAYER_IDX])
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids)
    routed_meta, routed_captured = routed_rows(layer.mlp, x, margin, seed)
    meta = {
        "case": "prefill_seq6_margin_clean",
        "layer": f"model.layers.{FULL_LAYER_IDX}",
        "layer_type": cfg.layer_types[FULL_LAYER_IDX],
        "seq_len": SEQ,
        "seed": seed,
        "softmax_scaling": float(layer.self_attn.scaling),
        "sink_rows": "one sink row per q head, the softmax runs over the "
            "keys plus the sink column and drops the sink before the value "
            "mix, sink_log_delta is the log-space relative denominator delta "
            "computed f64 on cpu and recorded f32",
        "mask_note": "seq 6 stays inside the 128 window, the causal mask "
            "carries no window cutoff at this length, the window behavior "
            "is recorded at tier 04",
        "moe": routed_meta,
    }
    payload = OrderedDict([("layer1.input", x)])
    captured = attention_stats_prefix(cap, "layer1.", cos, sin)
    captured.update({
        "layer1.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "layer1.layer.post_attention_layernorm_output":
            cap["layer.post_attention_layernorm_output"],
        "layer1.layer.mlp_output": cap["layer.mlp_output"],
        "layer1.layer.layer_output": cap["layer.layer_output"],
        "layer1.moe.router_logits": routed_captured["router_logits"],
        "layer1.moe.topk_weights": routed_captured["topk_weights"],
        "layer1.moe.moe_output": routed_captured["moe_output"],
    })
    print(f"[gen_bf16_gptoss_01_layer_internals] layer1 mixture (seed {seed}): "
          f"topk margin {margin:.3e}")
    return meta, payload, captured


def generate_moe_mixture(model: GptOssForCausalLM, cfg) -> tuple:
    """Records the routed-block surface of layer 0, the router decision rows
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
        "expert_loop": "the eager per-expert loop over the hit experts with "
            "the index_add scatter, the interleaved gate_up columns with the "
            "clamped glu activation, the accumulation order the reference "
            "module ran",
        "flip_budget_policy": routed_meta["flip_budget_policy"],
        "margin_floor": MARGIN_FLOOR,
        "topk_boundary_margin": margin,
        "seed": seed,
        "router": routed_meta["router"],
        "num_experts": routed_meta["num_experts"],
        "num_experts_per_tok": routed_meta["num_experts_per_tok"],
        "topk_indices": routed_meta["topk_indices"],
        "weights_dtype": routed_meta["weights_dtype"],
        "logits_dtype": routed_meta["logits_dtype"],
        "indices_dtype": routed_meta["indices_dtype"],
        "router_weight_key": f"model.layers.{MOE_LAYER_IDX}.mlp.router.weight",
        "router_bias_key": f"model.layers.{MOE_LAYER_IDX}.mlp.router.bias",
    }
    payload = OrderedDict([("moe.h", h)])
    captured = {
        "moe.router_logits": routed_captured["router_logits"],
        "moe.topk_weights": routed_captured["topk_weights"],
        "moe.moe_output": routed_captured["moe_output"],
        # Loader cross-check fingerprints, the suite compares the checkpoint
        # router weight and bias rows against these records.
        "moe.router_weight": routed_captured["router_weight"],
        "moe.router_bias": routed_captured["router_bias"],
    }
    print(f"[gen_bf16_gptoss_01_layer_internals] moe mixture (seed {seed}): "
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
    the three bare named driving tensors, one metadata sidecar and one stats
    sidecar over every recorded tensor.

    Args:
    - metadata, the merged metadata frame
    - mixtures, one (payload, captured) pair per mixture, the payload tensors
      carrying their file names and the captured tensors already carrying
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
        "weight_note": "the checkpoint ships MXFP4 expert weights, the "
            "reference quantizer dequantizes them to bf16 at load (mps has "
            "no MXFP4 kernel), every recorded tensor carries the bf16 rows",
        "implementation_status": "fixtures only, the Nim implementation of "
            "this family stays parked, no consumer exists yet",
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "sliding_window": cfg.sliding_window,
        "attention_shape": "1:1 alternating, sliding_attention at layers 0, "
            "2, ... 22 and full_attention between, window 128, 64 q heads "
            "over 8 kv heads, one sink row per q head, the routed block on "
            "every layer",
        "rope_parameters": cfg.rope_parameters,
        "layer_types": cfg.layer_types,
        "intermediate_size": cfg.intermediate_size,
        "mixtures": {
            "layer0": layer0_meta,
            "layer1": layer1_meta,
            "moe": moe_meta,
        },
    }
    save_fixture(metadata, [
        (layer0_payload, layer0_captured),
        (layer1_payload, layer1_captured),
        (moe_payload, moe_captured),
    ])

    print(f"[gen_bf16_gptoss_01_layer_internals] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_gptoss_01_layer_internals] wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
