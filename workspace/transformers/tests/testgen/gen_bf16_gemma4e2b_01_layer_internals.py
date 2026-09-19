#!/usr/bin/env python3
"""Layer-internals fixture file of the gemma-4-E2B-it checkpoint, recorded
with torch bf16 on Metal (mps) under the installed reference modeling.

Single-file grammar with one fixture file per family layer group, one bare bf16
driving tensor per mixture, all recorded intermediates live on the stats
frame as fingerprints.

No Qwen3 analog exists for these tier-01 rows. Qwen3 runs one uniform
full-attention kind over one head dim. This checkpoint is the dual-dim PLE shape:

- 4 sliding_attention layers then 1 full_attention layer, repeating, window 512
- sliding layers run head_dim 256, full layers run head_dim 512
- layers 15 through 34 share the kv stored at layers 13 (sliding) and 14 (full)

| mixture | row                                                                                          |
| ------- | -------------------------------------------------------------------------------------------- |
| layer0  | decoder layer 0, sliding attention with own kv plus the full PLE block, one seeded input     |
| chain   | layers 13, 14, 15 and 19 in model order, the kv-sharing handoff both layer kinds consume     |
| ple     | the model-level PLE pipeline over real token ids, token identity plus the context projection |

| file                                                              | contents                                                        |
| ----------------------------------------------------------------- | --------------------------------------------------------------- |
| layer0-13-14-15-19-gemma-4-E2B-it-00.safetensor                   | layer0.input, layer0.ple_input, chain inputs, ple.inputs_embeds |
| layer0-13-14-15-19-gemma-4-E2B-it-00.safetensor.metadata.json.zst | per-mixture metadata under the mixtures key                     |
| layer0-13-14-15-19-gemma-4-E2B-it-00.safetensor.stats.json.zst    | one uniform record per recorded tensor                          |

Stats keys carry the mixture-level `layer0.` / `chain.layer13.` prefixes,
the PLE block rows sit under each layer's own `ple.` prefix.

At seq 6 both mask kinds skip to the sdpa is_causal path (mask None)
and the 512-token window does not constrain these rows, tier-04 carries
the window behavior.

Consumed by tests/q_bf16/t_bf16_gemma4e2b_01_layer_internals.nim, one
assertion block per mixture.

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_gemma4e2b_01_layer_internals.py

RAM guard:

- the script refuses the weight load when the free+inactive+speculative pool
  sits below 32 GiB
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
MODEL_NAME = "gemma-4-E2B-it"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals",
    f"{MODEL_NAME}-layer-0-13-14-15-19"
)
FIXTURE_STEM = "layer0-13-14-15-19-" + MODEL_NAME + "-00"
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")

NUM_THREADS = 1

# Recorded rows of the dual-dim PLE shape, verified against the parsed
# layer_types at load time:
# - layer 0, the first sliding layer, own kv, the PLE block beside attention
# - layers 13 and 14, the last own-kv layers of each kind, they store
#   the full-length kv rows the shared layers reuse
# - layers 15 and 19, the first shared layers of each kind, no k/v projections
SLIDING_LAYER_IDX = 0
LAST_SLIDING_OWN_KV_IDX = 13
LAST_FULL_OWN_KV_IDX = 14
FIRST_SLIDING_SHARED_IDX = 15
FIRST_FULL_SHARED_IDX = 19
CHAIN_LAYER_IDXS = (LAST_SLIDING_OWN_KV_IDX, LAST_FULL_OWN_KV_IDX,
                    FIRST_SLIDING_SHARED_IDX, FIRST_FULL_SHARED_IDX)

# Per-mixture seeds, independent and order-agnostic. The chain mixture seeds
# each layer's PLE input at SEED_CHAIN + layer index.
SEED_LAYER0 = 321
SEED_CHAIN = 323
PLE_TOKEN_IDS = [9259, 993, 338, 4453, 2082, 531]

SEQ = 6

MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_gemma4e2b_01_layer_internals] vm_stat gave no page size line")


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
            "[gen_bf16_gemma4e2b_01_layer_internals] vm_stat gave no pool lines")
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

    - the free+inactive+speculative pool sits above the 32 GiB floor
    - no other python/torch process holds RAM, the pgrep match excludes
      this process chain (its own command line spells the torch dependency)
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_gemma4e2b_01_layer_internals] free+inactive+speculative "
            f"pool {pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_gemma4e2b_01_layer_internals] other python/torch processes "
            f"hold RAM: {stray}, stop and retry when idle")


def load_model() -> Gemma4ForConditionalGeneration:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps).

    Returns:
    - the model in eval mode, the recorded decoder layers 0, 13, 14, 15 and 19
      plus the per-layer-kind rotary feed every mixture consumes

    Identity asserts of the checkpoint config, dual-dim PLE shape:

    - 4 sliding_attention layers then 1 full_attention layer, repeating, window 512
    - sliding head_dim 256, full head_dim 512, 8 q heads over 1 kv head
    - layers 15 through 34 share kv, the stores sit at layers 13 and 14

    The PLE pipeline carries width 256 over 35 layers, the mlp is the dense
    double-wide kind with no routed block.
    """
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    tcfg = model.config.text_config
    assert tcfg.layer_types[SLIDING_LAYER_IDX] == "sliding_attention", (
        "layer 0 must be a sliding_attention row of the 4:1 pattern")
    assert tcfg.layer_types[LAST_FULL_OWN_KV_IDX] == "full_attention", (
        "layer 14 must be a full_attention row of the 4:1 pattern")
    assert tcfg.layer_types[FIRST_FULL_SHARED_IDX] == "full_attention", (
        "layer 19 must be a full_attention row of the 4:1 pattern")
    assert tcfg.sliding_window == 512, (
        "the recorded rows assume the 512 sliding window")
    assert tcfg.per_layer_config[0].head_dim == 256 and \
        tcfg.per_layer_config[LAST_FULL_OWN_KV_IDX].head_dim == 512, (
        "the recorded rows assume the dual head dims 256 (sliding) / 512 (full)")
    assert tcfg.num_attention_heads == 8 and tcfg.num_key_value_heads == 1, (
        "the recorded rows assume 8 q heads over 1 kv head")
    assert tcfg.num_hidden_layers - tcfg.num_kv_shared_layers == 15, (
        "the kv-sharing point must sit at layer 15")
    assert tcfg.hidden_size_per_layer_input == 256, (
        "the recorded rows assume the PLE width 256")
    assert tcfg.enable_moe_block is False and tcfg.use_double_wide_mlp is True, (
        "the recorded rows assume the dense double-wide mlp, no routed block")
    assert tcfg.final_logit_softcapping == 30.0, (
        "the recorded rows assume the 30.0 final logit softcapping")
    assert tcfg.rope_parameters["full_attention"]["rope_theta"] == 1e6 and \
        tcfg.rope_parameters["full_attention"]["partial_rotary_factor"] == 0.25, (
        "the full-attention rope must stay proportional theta 1e6, partial 0.25")
    assert tcfg.rope_parameters["sliding_attention"]["rope_theta"] == 1e4, (
        "the sliding rope must stay the default theta 1e4")
    assert tcfg.num_hidden_layers == 35 and tcfg.hidden_size == 1536 \
        and tcfg.vocab_size == 262144, (
        "the recorded rows assume the 35-layer, 1536-hidden, 262144-vocab checkpoint")
    assert model.model.language_model.layers[FIRST_SLIDING_SHARED_IDX] \
        .self_attn.is_kv_shared_layer, "layer 15 must be a kv-shared layer"
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
                      sin: torch.Tensor, mask, shared_kv: dict) -> dict:
    """Replays the reference Gemma4TextAttention.forward, the intermediate
    capture asserted equal to the module's own forward before saving.

    Args:
    - attn, the attention module whose forward the replay mirrors
    - h_norm, cos, sin, mask, the post-input-layernorm input, the per-layer-kind
      position embeddings and mask
    - shared_kv, the shared kv dict, kv-shared layers read the post-rope states
      the storing layer left under their layer kind, own-kv storing layers
      leave their states in the dict exactly as the module forward does

    Returns:
    - the capture dict, q post-norm and rope, the k and v rows
      (local or shared), the expanded kv, the sdpa output and the o_proj output
    """
    seq_len = h_norm.shape[1]
    head_dim = attn.head_dim
    q = attn.q_proj(h_norm).view(1, seq_len, -1, head_dim)
    q = attn.q_norm(q)
    q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    if attn.is_kv_shared_layer:
        k = shared_kv[attn.layer_type][0].to(q.device)
        v = shared_kv[attn.layer_type][1].to(q.device)
    else:
        k = attn.k_proj(h_norm).view(1, seq_len, -1, head_dim)
        k = attn.k_norm(k)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        v = attn.v_proj(h_norm).view(1, seq_len, -1, head_dim)
        v = attn.v_norm(v).transpose(1, 2)
        if attn.store_full_length_kv:
            shared_kv[attn.layer_type] = (k, v)
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
            attention_mask=mask, shared_kv_states=shared_kv)
    assert torch.equal(module_output, output), (
        "[gen_bf16_gemma4e2b_01_layer_internals] manual attention replay "
        "diverged from the module forward")
    return {
        "q_rot": q,
        "k_rot": k,
        "v": v,
        "k_expanded": k_expanded,
        "v_expanded": v_expanded,
        "sdpa_output": sdpa_output,
        "attn_output": output,
    }


def ple_block_capture(layer, h: torch.Tensor, ple_input: torch.Tensor) -> dict:
    """Runs the per-layer-input (PLE) block over one hidden state, recording
    the intermediate rows the attention path sits beside.

    Args:
    - layer, the decoder layer carrying the PLE submodules
    - h, ple_input, the post-attention hidden state and the per-layer input

    Returns:
    - the capture dict, the PLE rows ple.gate_output, ple.gated_product,
      ple.projection_output, ple.post_norm_output, ple.block_output
    """
    with torch.no_grad():
        gated = layer.per_layer_input_gate(h)
        activated = layer.act_fn(gated)
        product = activated * ple_input
        projected = layer.per_layer_projection(product)
        normalized = layer.post_per_layer_input_norm(projected)
        block_out = (h + normalized) * layer.layer_scalar
    return {
        "ple.gate_output": gated,
        "ple.gated_product": product,
        "ple.projection_output": projected,
        "ple.post_norm_output": normalized,
        "ple.block_output": block_out,
    }


def layer_chain_capture(layer, x: torch.Tensor, ple_input: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor, mask,
                        pos_ids: torch.Tensor, shared_kv: dict) -> dict:
    """Runs the full decoder-layer chain over one input, the manual replay
    asserted equal to the module forward before saving.

    Args:
    - layer, the decoder layer whose forward the replay mirrors
    - x, ple_input, pos_ids, the bf16 input, the per-layer input
      and the matching position ids
    - cos, sin, mask, shared_kv, the per-layer-kind position embeddings, mask
      and shared kv dict

    Returns:
    - the capture dict, every norm output, the attention rows, the mlp rows,
      the PLE block rows, the layer output and the attention-level capture
    """
    with torch.no_grad():
        h_norm = layer.input_layernorm(x)
        cap = attention_capture(layer.self_attn, h_norm, cos, sin, mask, shared_kv)
        h1 = x + layer.post_attention_layernorm(cap["attn_output"])
        h2 = layer.pre_feedforward_layernorm(h1)
        mlp_out = layer.mlp(h2)
        h3 = h1 + layer.post_feedforward_layernorm(mlp_out)
        ple_cap = ple_block_capture(layer, h3, ple_input)
        layer_out = ple_cap.pop("ple.block_output")
        module_out = layer(
            x, per_layer_input=ple_input, shared_kv_states=shared_kv,
            position_embeddings=(cos, sin), attention_mask=mask,
            position_ids=pos_ids)
    assert torch.equal(module_out, layer_out), (
        "[gen_bf16_gemma4e2b_01_layer_internals] manual layer chain diverged "
        "from the module forward")
    cap.update({
        "layer.input_layernorm_output": h_norm,
        "layer.post_attention_layernorm_output": h1,
        "layer.pre_feedforward_layernorm_output": h2,
        "layer.mlp_output": mlp_out,
        "layer.post_feedforward_layernorm_output": h3,
    })
    cap.update(ple_cap)
    cap["layer.layer_output"] = layer_out
    return cap


def namespaced_entries(cap: dict, prefix: str, cos: torch.Tensor,
                       sin: torch.Tensor) -> dict:
    """Namespaced stats entries of one layer capture, the attention rows under
    the bare prefix, the layer rows under `layer.`, the PLE rows under `ple.`."""
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
        prefix + "layer.post_feedforward_layernorm_output":
            cap["layer.post_feedforward_layernorm_output"],
        prefix + "layer.layer_output": cap["layer.layer_output"],
        prefix + "ple.gate_output": cap["ple.gate_output"],
        prefix + "ple.gated_product": cap["ple.gated_product"],
        prefix + "ple.projection_output": cap["ple.projection_output"],
        prefix + "ple.post_norm_output": cap["ple.post_norm_output"],
    }
    return entries


def seeded_input(seed: int, shape: tuple) -> torch.Tensor:
    """One seeded bf16 tensor on mps, the driving input of a mixture."""
    gen = torch.Generator(device="mps")
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.bfloat16, device="mps")


def generate_layer0_mixture(model: Gemma4ForConditionalGeneration,
                            tcfg) -> tuple:
    """Records the layer-0 row, sliding attention with own kv plus the full
    PLE block over one seeded input.

    Args:
    - model, tcfg, the loaded reference model and its parsed text config

    Returns:
    - meta, the mixture metadata
    - payload, the driving input tensors under their file names
    - captured, the stats-frame entries namespaced under layer0.
    """
    lm = model.model.language_model
    layer = lm.layers[SLIDING_LAYER_IDX]
    x = seeded_input(SEED_LAYER0, (1, SEQ, tcfg.hidden_size))
    ple_input = seeded_input(SEED_LAYER0 + 1,
                             (1, SEQ, tcfg.hidden_size_per_layer_input))
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = lm.rotary_emb(x, pos_ids, layer_type="sliding_attention")
    mask = build_mask(tcfg, x, pos_ids, tcfg.layer_types[SLIDING_LAYER_IDX])
    cap = layer_chain_capture(layer, x, ple_input, cos, sin, mask, pos_ids, {})
    meta = {
        "case": "prefill_seq6",
        "layer": f"model.language_model.layers.{SLIDING_LAYER_IDX}",
        "layer_type": tcfg.layer_types[SLIDING_LAYER_IDX],
        "kv_shared": False,
        "head_dim": layer.self_attn.head_dim,
        "seq_len": SEQ,
        "seed": SEED_LAYER0,
        "ple_input_seed": SEED_LAYER0 + 1,
        "softmax_scaling": float(layer.self_attn.scaling),
        "rope_theta": tcfg.rope_parameters["sliding_attention"]["rope_theta"],
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "mask_note": "seq 6 stays inside the 512 window, the sliding mask "
            "skips to the sdpa is_causal path (mask None), the window behavior "
            "is recorded at tier 04",
    }
    payload = OrderedDict([("layer0.input", x), ("layer0.ple_input", ple_input)])
    captured = namespaced_entries(cap, "layer0.", cos, sin)
    print(f"[gen_bf16_gemma4e2b_01_layer_internals] layer0 mixture "
          f"(seed {SEED_LAYER0})")
    return meta, payload, captured


def generate_chain_mixture(model: Gemma4ForConditionalGeneration, tcfg) -> tuple:
    """Records the kv-sharing chain row over layers 13, 14, 15 and 19, each
    layer replayed in model order and asserted against its module forward.

    Args:
    - model, tcfg, the loaded reference model and its parsed text config

    Returns:
    - meta, the mixture metadata
    - payload, the driving input tensors under their file names
    - captured, the stats-frame entries namespaced under chain.layerNN.
    """
    lm = model.model.language_model
    x = seeded_input(SEED_CHAIN, (1, SEQ, tcfg.hidden_size))
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    shared_kv = {}
    payload = OrderedDict([("chain.input", x)])
    captured = {}
    chain_meta = {}
    for idx in CHAIN_LAYER_IDXS:
        layer = lm.layers[idx]
        layer_type = tcfg.layer_types[idx]
        ple_input = seeded_input(SEED_CHAIN + idx,
                                 (1, SEQ, tcfg.hidden_size_per_layer_input))
        cos, sin = lm.rotary_emb(x, pos_ids, layer_type=layer_type)
        mask = build_mask(tcfg, x, pos_ids, layer_type)
        cap = layer_chain_capture(layer, x, ple_input, cos, sin, mask, pos_ids,
                                  shared_kv)
        prefix = f"chain.layer{idx}."
        captured.update(namespaced_entries(cap, prefix, cos, sin))
        payload[f"chain.ple{idx}"] = ple_input
        chain_meta[f"layer{idx}"] = {
            "layer_type": layer_type,
            "kv_shared": layer.self_attn.is_kv_shared_layer,
            "store_full_length_kv": layer.self_attn.store_full_length_kv,
            "head_dim": layer.self_attn.head_dim,
            "rope_theta": tcfg.rope_parameters[layer_type]["rope_theta"],
            "seed": SEED_CHAIN,
            "ple_input_seed": SEED_CHAIN + idx,
        }
        x = cap["layer.layer_output"]
    assert set(shared_kv.keys()) == {"sliding_attention", "full_attention"}, (
        "the chain must leave the shared kv of both layer kinds in the dict")
    assert shared_kv["sliding_attention"][0].shape[-1] == 256 and \
        shared_kv["full_attention"][0].shape[-1] == 512, (
        "the stored kv must carry the dual head dims 256 / 512")
    meta = {
        "case": "kv_sharing_chain_seq6",
        "layers": "model.language_model.layers.{%s}" %
                  ",".join(str(i) for i in CHAIN_LAYER_IDXS),
        "seed": SEED_CHAIN,
        "seq_len": SEQ,
        "kv_sharing": "layers 15 through 34 reuse the full-length kv stored "
            "at layer 13 (sliding) and layer 14 (full), the shared layers "
            "carry no k/v projections, the stored states are the post-rope "
            "rows the shared layers consume",
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "mask_note": "seq 6 stays inside the 512 window, both mask kinds skip "
            "to the sdpa is_causal path (mask None), the window behavior is "
            "recorded at tier 04",
        "chain": chain_meta,
    }
    print(f"[gen_bf16_gemma4e2b_01_layer_internals] chain mixture "
          f"(seed {SEED_CHAIN})")
    return meta, payload, captured


def generate_ple_mixture(model: Gemma4ForConditionalGeneration, tcfg) -> tuple:
    """Records the model-level PLE pipeline over real token ids, the token
    identity embedding plus the context projection and their combination.

    Args:
    - model, tcfg, the loaded reference model and its parsed text config

    Returns:
    - meta, the mixture metadata
    - payload, the driving input tensor under its file name
    - captured, the stats-frame entries namespaced under ple.
    """
    lm = model.model.language_model
    ids = torch.tensor([PLE_TOKEN_IDS], device="mps")
    with torch.no_grad():
        inputs_embeds = lm.embed_tokens(ids)
        token_identity = lm.get_per_layer_inputs(ids, inputs_embeds)
        combined = lm.project_per_layer_inputs(inputs_embeds, token_identity)
    meta = {
        "case": "ple_pipeline_seq6",
        "modules": "model.language_model.{embed_tokens_per_layer, "
                   "per_layer_model_projection, per_layer_projection_norm}",
        "input_tokens": PLE_TOKEN_IDS,
        "seq_len": SEQ,
        "per_layer_input_width": tcfg.hidden_size_per_layer_input,
        "num_layers": tcfg.num_hidden_layers,
        "pipeline": "embed_tokens_per_layer lookup scaled by sqrt(256) gives "
            "the token identity, per_layer_model_projection scaled by "
            "1/sqrt(1536) plus the per_layer_projection_norm gives the context "
            "projection, the combined output is (projection + identity) * "
            "2^-0.5, layer i consumes combined[:, :, i, :]",
    }
    payload = OrderedDict([("ple.inputs_embeds", inputs_embeds)])
    captured = {
        "ple.token_identity_output": token_identity,
        "ple.combined_output": combined,
    }
    print(f"[gen_bf16_gemma4e2b_01_layer_internals] ple mixture "
          f"(token ids {PLE_TOKEN_IDS})")
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
    chain_meta, chain_payload, chain_captured = generate_chain_mixture(
        model, tcfg)
    ple_meta, ple_payload, ple_captured = generate_ple_mixture(model, tcfg)

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
        "num_key_value_heads": tcfg.num_key_value_heads,
        "head_dim": tcfg.per_layer_config[0].head_dim,
        "global_head_dim": tcfg.per_layer_config[LAST_FULL_OWN_KV_IDX].head_dim,
        "sliding_window": tcfg.sliding_window,
        "num_kv_shared_layers": tcfg.num_kv_shared_layers,
        "attention_shape": "4 sliding_attention layers then 1 full_attention "
            "layer, repeating, window 512, sliding head_dim 256 and full "
            "head_dim 512, layers 15 through 34 share the kv stored at "
            "layers 13 and 14",
        "rope_parameters": tcfg.rope_parameters,
        "layer_types": tcfg.layer_types,
        "hidden_size_per_layer_input": tcfg.hidden_size_per_layer_input,
        "tie_word_embeddings": tcfg.tie_word_embeddings,
        "mixtures": {
            "layer0": layer0_meta,
            "chain": chain_meta,
            "ple": ple_meta,
        },
    }
    save_fixture(metadata, [
        (layer0_payload, layer0_captured),
        (chain_payload, chain_captured),
        (ple_payload, ple_captured),
    ])

    print(f"[gen_bf16_gemma4e2b_01_layer_internals] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_gemma4e2b_01_layer_internals] wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
