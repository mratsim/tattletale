#!/usr/bin/env python3
"""Layer-internals fixture file of the gemma-4-26B-A4B checkpoint, recorded
with torch bf16 on Metal (mps) under the installed reference modeling.

Single-file grammar with one fixture file per family layer group, one bare bf16
driving tensor per mixture, all recorded intermediates live on the stats
frame as fingerprints.

No Qwen3 analog exists for these tier-01 rows. Qwen3 runs one uniform
full-attention kind over one head dim and one dense FFN. This checkpoint is
the dual-dim k_eq_v shape with a routed block on every layer:

- 5 sliding_attention layers then 1 full_attention layer, repeating, window 1024
- sliding layers run head_dim 256 over 8 kv heads, full layers run head_dim 512
  over 2 kv heads
- full layers carry attention_k_eq_v with no v_proj weight, the value rows
  derive from the shared k projection through the unscaled v_norm

Every decoder layer pairs the dense mlp (intermediate 2112) with the routed
block over 128 experts with top 8, a softmax router feeding the eager
per-expert loop over gate_up/down projections of moe_intermediate 704.

| mixture | row                                                                                      |
| ------- | ---------------------------------------------------------------------------------------- |
| layer0  | decoder layer 0, sliding attention with own kv plus the dense mlp and the routed block   |
| layer5  | decoder layer 5, the k_eq_v full attention with its cache boundary plus the routed block |
| moe     | the routed block surface of layer 0, the router decision rows plus the eager expert loop |

| file                                                     | contents                                    |
| -------------------------------------------------------- | ------------------------------------------- |
| layer0-5-gemma-4-26B-A4B-00.safetensor                   | layer0.input, layer5.input, moe.h           |
| layer0-5-gemma-4-26B-A4B-00.safetensor.metadata.json.zst | per-mixture metadata under the mixtures key |
| layer0-5-gemma-4-26B-A4B-00.safetensor.stats.json.zst    | one uniform record per recorded tensor      |

Stats keys carry the mixture-level `layer0.` / `layer5.` / `moe.` prefixes.

Recorded expert ids live in the metadata, integer ids carry no stats record.

Every routed mixture stands on a margin-clean seed, the seed advances one
step at a time until the top-k boundary margin clears the 1e-4 floor
protecting the exact expert-id comparisons in the consuming suite.

The layer mixtures search on the chain the router actually sees, the seed
advances until the router margin over the post-attention residual
clears the floor.

The k_eq_v cache boundary, measured against the reference before recording.
The reference does not store identical K and V tensors:

- the full-layer value rows are v_norm applied to the same k projection
  the keys consume, the keys are the rotated scaled k_norm of that projection
- the reference DynamicCache keeps K and V as separate entries, the tying
  lives at the projection level with no v_proj weight on the full layers

The layer5 mixture records the cache-boundary rows the reference stores,
`layer5.cache_k` / `layer5.cache_v`, beside the k projection source row
`layer5.k_proj_output` the V derivation consumes.

At seq 6 both mask kinds skip to the sdpa is_causal path (mask None)
and the 1024-token window does not constrain these rows, tier-04 carries
the window behavior.

Consumed by tests/q_bf16/t_bf16_gemma426b_01_layer_internals.nim, one
assertion block per mixture.

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_gemma426b_01_layer_internals.py

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
from transformers import Gemma4ForConditionalGeneration  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.masking_utils import (  # noqa: E402
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    apply_rotary_pos_emb,
    repeat_kv,
)

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity constants of the checkpoint.
MODEL_NAME = "gemma-4-26B-A4B"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals",
    f"{MODEL_NAME}-layer-0-5"
)
FIXTURE_STEM = "layer0-5-" + MODEL_NAME + "-00"
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")

NUM_THREADS = 1

# Recorded rows of the dual-dim k_eq_v shape, verified against the parsed
# layer_types at load time:
# - layer 0, the first sliding layer, the standard mixer with own kv
# - layer 5, the first full layer, the k_eq_v row with its cache boundary
#
# The moe mixture replays the layer-0 routed block surface over a separate
# margin-clean driving input.
SLIDING_LAYER_IDX = 0
FULL_LAYER_IDX = 5
MOE_LAYER_IDX = SLIDING_LAYER_IDX

# Per-mixture seeds, independent and order-agnostic. The routed mixtures
# advance their seed one step at a time until the top-k boundary margin
# clears the floor.
SEED_LAYER0 = 321
SEED_LAYER5 = 322
SEED_MOE = 323

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
        "[gen_bf16_gemma426b_01_layer_internals] vm_stat gave no page size line")


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
            "[gen_bf16_gemma426b_01_layer_internals] vm_stat gave no pool lines")
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
      this process chain, whose own command line spells the torch dependency
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_gemma426b_01_layer_internals] free+inactive+speculative "
            f"pool {pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_gemma426b_01_layer_internals] other python/torch processes "
            f"hold RAM: {stray}, stop and retry when idle")


def load_model() -> Gemma4ForConditionalGeneration:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps).

    Returns:
    - the model in eval mode, the recorded decoder layers 0 and 5 plus
      the per-layer-kind rotary feed every mixture consumes

    Identity asserts of the checkpoint config, the dual-dim k_eq_v routed
    shape on every layer:

| property      | recorded value                                                     |
| ------------- | ------------------------------------------------------------------ |
| layer pattern | 5 sliding_attention layers then 1 full_attention, window 1024      |
| head dims     | sliding 256 over 8 kv heads, full 512 over 2 kv heads              |
| attention     | attention_k_eq_v on the full layers, v_proj None, unscaled v_norm  |
| block         | dense mlp 2112 plus the routed block everywhere, 128 experts top 8 |
| routed widths | moe_intermediate 704, no kv sharing, no PLE                        |
    """
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    tcfg = model.config.text_config
    assert tcfg.layer_types[SLIDING_LAYER_IDX] == "sliding_attention", (
        "layer 0 must be a sliding_attention row of the 5:1 pattern")
    assert tcfg.layer_types[FULL_LAYER_IDX] == "full_attention", (
        "layer 5 must be a full_attention row of the 5:1 pattern")
    assert tcfg.sliding_window == 1024, (
        "the recorded rows assume the 1024 sliding window")
    assert tcfg.per_layer_config[SLIDING_LAYER_IDX].head_dim == 256 and \
        tcfg.per_layer_config[FULL_LAYER_IDX].head_dim == 512, (
        "the recorded rows assume the dual head dims 256 (sliding) / 512 (full)")
    assert tcfg.per_layer_config[FULL_LAYER_IDX].num_key_value_heads == 2, (
        "the recorded rows assume the full layers run 2 kv heads")
    assert tcfg.num_attention_heads == 16, (
        "the recorded rows assume 16 q heads")
    assert tcfg.per_layer_config[SLIDING_LAYER_IDX].num_key_value_heads == 8, (
        "the recorded rows assume 8 sliding kv heads")
    assert tcfg.attention_k_eq_v is True, (
        "the recorded rows assume attention_k_eq_v")
    assert tcfg.num_kv_shared_layers == 0 and \
        tcfg.hidden_size_per_layer_input == 0, (
        "the recorded rows assume no kv sharing and no PLE")
    assert tcfg.enable_moe_block is True, (
        "the recorded rows assume the routed block on every layer")
    assert tcfg.num_experts == 128 and tcfg.top_k_experts == 8, (
        "the recorded rows assume 128 experts with top 8")
    assert tcfg.moe_intermediate_size == 704 and tcfg.intermediate_size == 2112, (
        "the recorded rows assume moe_intermediate 704 beside the dense 2112")
    assert tcfg.final_logit_softcapping == 30.0, (
        "the recorded rows assume the 30.0 final logit softcapping")
    assert tcfg.rope_parameters["full_attention"]["rope_theta"] == 1e6 and \
        tcfg.rope_parameters["full_attention"]["partial_rotary_factor"] == 0.25, (
        "the full-attention rope must stay proportional theta 1e6, partial 0.25")
    assert tcfg.rope_parameters["sliding_attention"]["rope_theta"] == 1e4, (
        "the sliding rope must stay the default theta 1e4")
    assert tcfg.num_hidden_layers == 30 and tcfg.hidden_size == 2816 \
        and tcfg.vocab_size == 262144, (
        "the recorded rows assume the 30-layer, 2816-hidden, 262144-vocab checkpoint")
    layers = model.model.language_model.layers
    assert layers[SLIDING_LAYER_IDX].self_attn.v_proj is not None, (
        "sliding layers keep their own v_proj")
    assert layers[FULL_LAYER_IDX].self_attn.v_proj is None, (
        "the k_eq_v full layer must carry no v_proj weight")
    assert not layers[SLIDING_LAYER_IDX].self_attn.is_kv_shared_layer, (
        "this checkpoint shares no kv across layers")
    return model


def build_mask(tcfg, embeds: torch.Tensor, pos_ids: torch.Tensor,
               layer_type: str):
    """Returns the mask for one layer kind, built through the same masking
    entry points the model forward uses.

    Args:
    - tcfg, embeds, pos_ids, the text config, the bf16 embeds,
      the matching position ids
    - layer_type, the recorded layer kind

    Returns None when the sdpa path skips to is_causal.
    """
    kwargs = {
        "config": tcfg,
        "inputs_embeds": embeds,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": pos_ids,
    }
    if layer_type == "sliding_attention":
        return create_sliding_window_causal_mask(**kwargs)
    return create_causal_mask(**kwargs)


def attention_capture(attn, h_norm: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor, mask) -> dict:
    """Replays the reference Gemma4TextAttention.forward, the intermediate
    capture asserted equal to the module's own forward before saving.

    Args:
    - attn, the attention module whose forward the replay mirrors
    - h_norm, cos, sin, mask, the post-input-layernorm input, the per-layer-kind
      position embeddings and mask

    Returns:
    - the capture dict, q post-norm and rope, the k projection source row
      for the k_eq_v kind, the k and v rows, the expanded kv, the sdpa
      output and the o_proj output
    """
    seq_len = h_norm.shape[1]
    head_dim = attn.head_dim
    q = attn.q_proj(h_norm).view(1, seq_len, -1, head_dim)
    q = attn.q_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    k_source = attn.k_proj(h_norm).view(1, seq_len, -1, head_dim)
    k = attn.k_norm(k_source)
    k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    if attn.v_proj is not None:
        v_source = attn.v_proj(h_norm).view(1, seq_len, -1, head_dim)
    else:
        v_source = k_source
    v = attn.v_norm(v_source).transpose(1, 2)
    k_expanded = repeat_kv(k, attn.num_key_value_groups)
    v_expanded = repeat_kv(v, attn.num_key_value_groups)
    sdpa_output = torch.nn.functional.scaled_dot_product_attention(
        q, k_expanded, v_expanded, attn_mask=mask, dropout_p=0.0,
        is_causal=(mask is None and seq_len > 1), scale=attn.scaling)
    attn_output = sdpa_output.transpose(1, 2).contiguous()
    output = attn.o_proj(attn_output.reshape(1, seq_len, -1))
    with torch.no_grad():
        module_output, _ = attn(
            hidden_states=h_norm, position_embeddings=(cos, sin),
            attention_mask=mask, shared_kv_states={})
    assert torch.equal(module_output, output), (
        "[gen_bf16_gemma426b_01_layer_internals] manual attention replay "
        "diverged from the module forward")
    capture = {
        "q_rot": q,
        "k_rot": k,
        "v": v,
        "k_expanded": k_expanded,
        "v_expanded": v_expanded,
        "sdpa_output": sdpa_output,
        "attn_output": output,
    }
    if attn.v_proj is None:
        capture["k_proj_output"] = k_source
    return capture


def cache_boundary_rows(attn, k: torch.Tensor, v: torch.Tensor,
                        layer_idx: int) -> dict:
    """Runs one DynamicCache update over the layer's k and v rows, the cache-boundary tensors
    the reference stores.

    Args:
    - attn, k, v, layer_idx, the attention module, its post-rope key rows,
      its value rows and the cache slot index

    Returns:
    - the capture dict, the stored cache_k and cache_v rows
    """
    cache = DynamicCache()
    stored_k, stored_v = cache.update(k, v, layer_idx)
    assert torch.equal(stored_k, k) and torch.equal(stored_v, v), (
        "[gen_bf16_gemma426b_01_layer_internals] the cache update must store "
        "the attention rows verbatim on the first step")
    return {"cache_k": stored_k, "cache_v": stored_v}


def routed_rows(router, experts, norms, residual: torch.Tensor) -> dict:
    """Runs the routed block over one residual, the weights self-checked
    against the router probabilities before saving.

    Args:
    - router, experts, norms, the layer's router module, its expert loop,
      plus the (pre_feedforward_layernorm_2, post_feedforward_layernorm_2)
      pair the block consumes
    - residual, the bf16 post-attention residual the router sees, [1, seq, hidden]

    Returns:
    - the capture dict, the router probabilities, the renormed scaled
      top-k weights, the experts input and output rows and the normalized
      routed block output
    """
    with torch.no_grad():
        flat = residual.reshape(-1, residual.shape[-1])
        router_probs, topk_weights, topk_index = router(flat)
        experts_input = norms[0](flat)
        experts_output = experts(experts_input, topk_index, topk_weights)
        moe_output = norms[1](experts_output.reshape(residual.shape))

    # Self-check before saving, the recorded weights stay the per-expert
    # scaled renorm of the selected probabilities, both sides on the f32
    # grid the reference router computed.
    selected = router_probs.gather(1, topk_index)
    renorm = selected / selected.sum(dim=-1, keepdim=True)
    assert torch.equal(
        topk_weights, renorm * router.per_expert_scale[topk_index]), (
        "[gen_bf16_gemma426b_01_layer_internals] routing weights diverge "
        "from the scaled renorm of the selected probabilities")
    return {
        "router_probs": router_probs,
        "topk_weights": topk_weights,
        "topk_index": topk_index,
        "experts_input": experts_input,
        "experts_output": experts_output,
        "moe_output": moe_output,
    }


def topk_boundary_margin(router_probs: torch.Tensor, top_k: int) -> float:
    """Smallest gap between the Kth and the K+1th sorted probabilities,
    minimized over all rows, the exact-index condition of the top-k selection."""
    sorted_probs = router_probs.sort(dim=-1, descending=True).values
    gaps = sorted_probs[:, top_k - 1] - sorted_probs[:, top_k]
    return gaps.min().item()


def layer_chain_capture(layer, x: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, mask,
                        pos_ids: torch.Tensor) -> dict:
    """Runs the full decoder-layer chain over one input, the manual replay
    asserted equal to the module forward before saving.

    Args:
    - layer, the decoder layer whose forward the replay mirrors
    - x, pos_ids, the bf16 input and the matching position ids
    - cos, sin, mask, the per-layer-kind position embeddings and mask

    Returns:
    - the capture dict, every norm output, the attention rows
    - the dense mlp rows, the routed block rows, the scaled layer output

The decoder layer closes on the learned per-layer scale, the module output
is the residual sum times `layer.layer_scalar`.

The routed block reads the post-attention residual twice, the router over
the flat residual, the experts over its pre_feedforward_layernorm_2 output.
    """
    with torch.no_grad():
        h_norm = layer.input_layernorm(x)
        cap = attention_capture(layer.self_attn, h_norm, cos, sin, mask)
        h1 = x + layer.post_attention_layernorm(cap["attn_output"])
        h2 = layer.pre_feedforward_layernorm(h1)
        mlp_out = layer.mlp(h2)
        moe_1 = layer.post_feedforward_layernorm_1(mlp_out)
        routed = routed_rows(
            layer.router, layer.experts,
            (layer.pre_feedforward_layernorm_2,
             layer.post_feedforward_layernorm_2), h1)
        h3 = moe_1 + routed["moe_output"]
        h4 = layer.post_feedforward_layernorm(h3)
        layer_out = h1 + h4
        layer_out = layer_out * layer.layer_scalar
        module_out = layer(
            x, shared_kv_states={}, position_embeddings=(cos, sin),
            attention_mask=mask, position_ids=pos_ids)
    assert torch.equal(module_out, layer_out), (
        "[gen_bf16_gemma426b_01_layer_internals] manual layer chain diverged "
        "from the module forward")
    cap.update({
        "layer.input_layernorm_output": h_norm,
        "layer.post_attention_layernorm_output": h1,
        "layer.pre_feedforward_layernorm_output": h2,
        "layer.mlp_output": mlp_out,
        "layer.moe.router_probs": routed["router_probs"],
        "layer.moe.topk_weights": routed["topk_weights"],
        "layer.moe.experts_input": routed["experts_input"],
        "layer.moe.experts_output": routed["experts_output"],
        "layer.moe.moe_output": routed["moe_output"],
        "layer.layer_output": layer_out,
    })
    cap["routed_topk_index"] = routed["topk_index"]
    return cap


def namespaced_entries(cap: dict, prefix: str, cos: torch.Tensor,
                       sin: torch.Tensor) -> dict:
    """Namespaced stats entries of one layer capture, the attention rows under
    the bare prefix, the layer rows under `layer.`."""
    entries = {
        prefix + "cos": cos,
        prefix + "sin": sin,
        prefix + "q_rot": cap["q_rot"],
        prefix + "k_rot": cap["k_rot"],
        prefix + "v": cap["v"],
        prefix + "k_expanded": cap["k_expanded"],
        prefix + "v_expanded": cap["v_expanded"],
        prefix + "sdpa_output": cap["sdpa_output"],
        prefix + "attn_output": cap["attn_output"],
        prefix + "layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        prefix + "layer.post_attention_layernorm_output":
            cap["layer.post_attention_layernorm_output"],
        prefix + "layer.pre_feedforward_layernorm_output":
            cap["layer.pre_feedforward_layernorm_output"],
        prefix + "layer.mlp_output": cap["layer.mlp_output"],
        prefix + "layer.moe.router_probs": cap["layer.moe.router_probs"],
        prefix + "layer.moe.topk_weights": cap["layer.moe.topk_weights"],
        prefix + "layer.moe.experts_input": cap["layer.moe.experts_input"],
        prefix + "layer.moe.experts_output": cap["layer.moe.experts_output"],
        prefix + "layer.moe.moe_output": cap["layer.moe.moe_output"],
        prefix + "layer.layer_output": cap["layer.layer_output"],
    }
    if "k_proj_output" in cap:
        entries[prefix + "k_proj_output"] = cap["k_proj_output"]
        entries[prefix + "cache_k"] = cap["cache_k"]
        entries[prefix + "cache_v"] = cap["cache_v"]
    return entries


def seeded_input(seed: int, shape: tuple) -> torch.Tensor:
    """One seeded bf16 tensor on mps, the driving input of a mixture."""
    gen = torch.Generator(device="mps")
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.bfloat16, device="mps")


def margin_clean_layer_input(model: Gemma4ForConditionalGeneration, tcfg,
                             layer, seed: int, layer_idx: int) -> tuple:
    """Builds a margin-clean layer input, the seed advancing step by step
    until the router margin clears the recorded floor.

    Args:
    - model, tcfg, the loaded reference model and its parsed text config
    - layer, seed, layer_idx, the decoder layer, its index in the layer stack,
      the first seed tried

    Returns:
    - the bf16 input, the advancing seed, the achieved margin
    - the position ids, position embeddings and mask of the winning seed

    The router sees the post-attention residual, so the margin search runs
    the whole chain per seed, the winning chain capture feeds the mixture.
    """
    lm = model.model.language_model
    layer_type = tcfg.layer_types[layer_idx]
    for _ in range(MAX_SEED_TRIES):
        x = seeded_input(seed, (1, SEQ, tcfg.hidden_size))
        pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
        cos, sin = lm.rotary_emb(x, pos_ids, layer_type=layer_type)
        mask = build_mask(tcfg, x, pos_ids, layer_type)
        cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids)
        margin = topk_boundary_margin(
            cap["layer.moe.router_probs"], tcfg.top_k_experts)
        if margin > MARGIN_FLOOR:
            return x, pos_ids, cos, sin, mask, cap, seed, margin
        seed += 1
    raise SystemExit(
        "[gen_bf16_gemma426b_01_layer_internals] no margin-clean seed found "
        "for the layer router")


def routed_meta(routed_topk_index: list, margin: float, seed: int) -> dict:
    """Routed-decision metadata rows of one margin-clean capture."""
    return {
        "seed": seed,
        "margin_floor": MARGIN_FLOOR,
        "topk_boundary_margin": margin,
        "router": "softmax_topk_renorm",
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "norm_topk_prob": True,
        "topk_indices": routed_topk_index,
        "weights_dtype": "float32",
        "logits_dtype": "float32",
        "indices_dtype": "int64",
        "flip_budget_policy": "exact expert-id comparisons in the consuming "
            "suites stand on a recorded positive boundary margin with the "
            "1e-4 floor, a row below the floor would carry an explicit "
            "per-mixture recorded exception, never a silent widening",
    }


def margin_clean_routed_input(layer, seed: int, tcfg) -> tuple:
    """Builds a margin-clean routed-block input over one decoder layer's
    router and experts, the seed advancing step by step until the top-k
    boundary margin clears the recorded floor.

    Args:
    - layer, seed, the decoder layer whose routed block replays, plus the first seed tried

    Returns:
    - the bf16 input, the routed capture, the advancing seed and the achieved margin
    """
    for _ in range(MAX_SEED_TRIES):
        h = seeded_input(seed, (1, SEQ, tcfg.hidden_size))
        routed = routed_rows(
            layer.router, layer.experts,
            (layer.pre_feedforward_layernorm_2,
             layer.post_feedforward_layernorm_2), h)
        margin = topk_boundary_margin(routed["router_probs"], tcfg.top_k_experts)
        if margin > MARGIN_FLOOR:
            return h, routed, seed, margin
        seed += 1
    raise SystemExit(
        "[gen_bf16_gemma426b_01_layer_internals] no margin-clean seed found "
        "for the routed block")


def generate_layer0_mixture(model: Gemma4ForConditionalGeneration,
                            tcfg) -> tuple:
    """Records the layer-0 row, sliding attention with own kv plus the dense
    mlp and the routed block over one margin-clean seeded input.

    Args:
    - model, tcfg, the loaded reference model and its parsed text config

    Returns:
    - meta, the mixture metadata
    - payload, the driving input tensor under its file name
    - captured, the stats-frame entries namespaced under layer0.
    """
    layer = model.model.language_model.layers[SLIDING_LAYER_IDX]
    x, pos_ids, cos, sin, mask, cap, seed, margin = margin_clean_layer_input(
        model, tcfg, layer, SEED_LAYER0, SLIDING_LAYER_IDX)
    meta = {
        "case": "prefill_seq6_margin_clean",
        "layer": f"model.language_model.layers.{SLIDING_LAYER_IDX}",
        "layer_type": tcfg.layer_types[SLIDING_LAYER_IDX],
        "kv_shared": False,
        "head_dim": layer.self_attn.head_dim,
        "seq_len": SEQ,
        "seed": seed,
        "softmax_scaling": float(layer.self_attn.scaling),
        "layer_scalar": float(layer.layer_scalar.item()),
        "rope_theta": tcfg.rope_parameters["sliding_attention"]["rope_theta"],
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "mask_note": "seq 6 stays inside the 1024 window, the sliding mask "
            "skips to the sdpa is_causal path (mask None), the window behavior "
            "is recorded at tier 04",
        "moe": routed_meta(
            cap.pop("routed_topk_index").tolist(), margin, seed),
    }
    payload = OrderedDict([("layer0.input", x)])
    captured = namespaced_entries(cap, "layer0.", cos, sin)
    print(f"[gen_bf16_gemma426b_01_layer_internals] layer0 mixture "
          f"(seed {seed}): topk margin {margin:.3e}")
    return meta, payload, captured


def generate_layer5_mixture(model: Gemma4ForConditionalGeneration,
                            tcfg) -> tuple:
    """Records the layer-5 row, the k_eq_v full attention with its shared
    k projection row and cache boundary plus the routed block over one
    margin-clean seeded input.

    Args:
    - model, tcfg, the loaded reference model and its parsed text config

    Returns:
    - meta, the mixture metadata
    - payload, the driving input tensor under its file name
    - captured, the stats-frame entries namespaced under layer5.
    """
    lm = model.model.language_model
    layer = lm.layers[FULL_LAYER_IDX]
    x, pos_ids, cos, sin, mask, cap, seed, margin = margin_clean_layer_input(
        model, tcfg, layer, SEED_LAYER5, FULL_LAYER_IDX)
    attn = layer.self_attn
    cap.update(cache_boundary_rows(attn, cap["k_rot"], cap["v"], FULL_LAYER_IDX))
    meta = {
        "case": "prefill_seq6_k_eq_v_margin_clean",
        "layer": f"model.language_model.layers.{FULL_LAYER_IDX}",
        "layer_type": tcfg.layer_types[FULL_LAYER_IDX],
        "kv_shared": False,
        "head_dim": attn.head_dim,
        "seq_len": SEQ,
        "seed": seed,
        "softmax_scaling": float(attn.scaling),
        "layer_scalar": float(layer.layer_scalar.item()),
        "rope_theta": tcfg.rope_parameters["full_attention"]["rope_theta"],
        "partial_rotary_factor":
            tcfg.rope_parameters["full_attention"]["partial_rotary_factor"],
        "kv_tying": "attention_k_eq_v: the full layer carries no v_proj "
            "weight, the value rows are v_norm applied to the same k "
            "projection the keys consume while the keys are the rotated "
            "scaled k_norm of that projection, v_norm runs without a scale "
            "so the value rows stay the unscaled normalization, the k "
            "projection source row is recorded as layer5.k_proj_output",
        "cache_boundary": "the reference DynamicCache keeps K and V as "
            "separate entries on both layer kinds, cache_k and cache_v are "
            "the tensors one update stores, cache_k equals layer5.k_rot and "
            "cache_v equals layer5.v on this row",
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "mask_note": "seq 6 stays inside the 1024 window, both mask kinds "
            "skip to the sdpa is_causal path (mask None), the window behavior "
            "is recorded at tier 04",
        "moe": routed_meta(
            cap.pop("routed_topk_index").tolist(), margin, seed),
    }
    payload = OrderedDict([("layer5.input", x)])
    captured = namespaced_entries(cap, "layer5.", cos, sin)
    print(f"[gen_bf16_gemma426b_01_layer_internals] layer5 mixture "
          f"(seed {seed}, k_eq_v): topk margin {margin:.3e}")
    return meta, payload, captured


def generate_moe_mixture(model: Gemma4ForConditionalGeneration,
                         tcfg) -> tuple:
    """Records the routed block surface of layer 0, the router decision rows
    plus the eager expert loop over one margin-clean input.

    Args:
    - model, tcfg, the loaded reference model and its parsed text config

    Returns:
    - meta, the mixture metadata
    - payload, the driving input tensor under its file name
    - captured, the stats-frame entries namespaced under moe.
    """
    layer = model.model.language_model.layers[MOE_LAYER_IDX]
    h, routed, seed, margin = margin_clean_routed_input(layer, SEED_MOE, tcfg)
    meta = routed_meta(routed["topk_index"].tolist(), margin, seed)
    meta.update({
        "case": "routed_block_seq6_margin_clean",
        "layer": f"model.language_model.layers.{MOE_LAYER_IDX}.router "
            f"+ model.language_model.layers.{MOE_LAYER_IDX}.experts",
        "seq_len": SEQ,
        "hidden_dtype": "bfloat16",
        "router_input": "the driving input plays the post-attention residual "
            "role, the router reads it flat and the experts read its "
            "pre_feedforward_layernorm_2 output",
        "expert_loop": "the eager per-expert loop over the hit experts with "
            "the index_add scatter, the accumulation order the reference "
            "module ran",
    })
    payload = OrderedDict([("moe.h", h)])
    captured = {
        "moe.router_probs": routed["router_probs"],
        "moe.topk_weights": routed["topk_weights"],
        "moe.experts_input": routed["experts_input"],
        "moe.experts_output": routed["experts_output"],
        "moe.moe_output": routed["moe_output"],
    }
    print(f"[gen_bf16_gemma426b_01_layer_internals] moe mixture "
          f"(seed {seed}): topk margin {margin:.3e}")
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
    the bare named driving tensors, one metadata sidecar and one stats
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
    tcfg = model.config.text_config

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    layer0_meta, layer0_payload, layer0_captured = generate_layer0_mixture(
        model, tcfg)
    layer5_meta, layer5_payload, layer5_captured = generate_layer5_mixture(
        model, tcfg)
    moe_meta, moe_payload, moe_captured = generate_moe_mixture(model, tcfg)

    metadata = {
        "model": MODEL_NAME,
        "file": FIXTURE_STEM + ".safetensor",
        "dtype": "bfloat16",
        "num_threads": NUM_THREADS,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "recorded_from": recorded_from,
        "device": "mps",
        "hidden_size": tcfg.hidden_size,
        "num_attention_heads": tcfg.num_attention_heads,
        "num_key_value_heads":
            tcfg.per_layer_config[SLIDING_LAYER_IDX].num_key_value_heads,
        "head_dim": tcfg.per_layer_config[SLIDING_LAYER_IDX].head_dim,
        "global_head_dim": tcfg.per_layer_config[FULL_LAYER_IDX].head_dim,
        "global_num_key_value_heads":
            tcfg.per_layer_config[FULL_LAYER_IDX].num_key_value_heads,
        "sliding_window": tcfg.sliding_window,
        "num_kv_shared_layers": tcfg.num_kv_shared_layers,
        "attention_k_eq_v": tcfg.attention_k_eq_v,
        "attention_shape": "5 sliding_attention layers then 1 full_attention "
            "layer, repeating, window 1024, sliding head_dim 256 over 8 kv "
            "heads and full head_dim 512 over 2 kv heads, no kv sharing",
        "moe_shape": "every layer pairs the dense mlp (intermediate 2112) "
            "with the routed block, 128 experts top 8 over moe_intermediate "
            "704, the router renorms its top-8 probabilities and applies the "
            "per-expert scale",
        "num_experts": tcfg.num_experts,
        "top_k_experts": tcfg.top_k_experts,
        "moe_intermediate_size": tcfg.moe_intermediate_size,
        "rope_parameters": tcfg.rope_parameters,
        "tie_word_embeddings": tcfg.tie_word_embeddings,
        "mixtures": {
            "layer0": layer0_meta,
            "layer5": layer5_meta,
            "moe": moe_meta,
        },
    }
    save_fixture(metadata, [
        (layer0_payload, layer0_captured),
        (layer5_payload, layer5_captured),
        (moe_payload, moe_captured),
    ])

    print(f"[gen_bf16_gemma426b_01_layer_internals] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_gemma426b_01_layer_internals] wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
