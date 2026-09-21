#!/usr/bin/env python3
"""Tier-01 boundary-pair fixture generator, gemma-3-270m-it,
torch bf16 on Metal (mps) under the installed reference modeling,

- consumer tests/q_bf16/t_bf16_gemma3270m_01_layer_internals.nim
- one bare bf16 driving tensor per mixture, recorded intermediates stay on the stats frame as fingerprints
- layer 4 sliding_attention over rope_local_base_freq 1e4, layer 5 full_attention over rope_theta 1e6, the 5:1 layer pattern

- the boundary mixture chains the layer-5 full block on the layer-4 output over one seeded input

- at seq 6 both mask kinds skip to the sdpa is_causal path (mask None), the tier-04 records carry the window behavior
- fixture dir tests/fixtures/bf16-01-layer-internals/gemma-3-270m-it-layer-4-5/, file layer4-5-gemma-3-270m-it-00.safetensor
- the run refuses the weight load under 8 GiB free+inactive+speculative pool, other python/torch processes holding RAM block it

Regenerate from the worktree root:

  uv run python workspace/transformers/tests/testgen/gen_bf16_gemma3270m_01_layer_internals.py
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
from transformers import Gemma3ForCausalLM  # noqa: E402
from transformers.masking_utils import (  # noqa: E402
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma3.modeling_gemma3 import (  # noqa: E402
    apply_rotary_pos_emb,
    repeat_kv,
)

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity constants of the checkpoint.
MODEL_NAME = "gemma-3-270m-it"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals", f"{MODEL_NAME}-layer-4-5"
)
FIXTURE_STEM = "layer4-5-" + MODEL_NAME + "-00"
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")

NUM_THREADS = 1

# Recorded boundary pair of the 5:1 pattern, layer 4 (sliding_attention)
# plus layer 5 (full_attention), verified against the parsed layer_types
# at load time.
SLIDING_LAYER_IDX = 4
FULL_LAYER_IDX = 5

# Per-mixture seeds, independent and order-agnostic.
SEED_SLIDING = 101
SEED_FULL = 102
SEED_BOUNDARY = 103

SEQ = 6

MIN_POOL_BYTES = 8 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_gemma3270m_01_layer_internals] vm_stat gave no page size line")


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
            "[gen_bf16_gemma3270m_01_layer_internals] vm_stat gave no pool lines")
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

    - the free+inactive+speculative pool sits above the 8 GiB floor
    - no other python/torch process holds RAM, the pgrep match excludes
      this process chain (its own command line spells the torch dependency)
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_gemma3270m_01_layer_internals] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_gemma3270m_01_layer_internals] other python/torch processes hold RAM: {stray}, "
            "stop and retry when idle")


def load_model() -> Gemma3ForCausalLM:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps).

    Returns:
    - the model in eval mode, its layer 4 and layer 5 decoder layers plus
      the shared dual-theta rotary feed every mixture

    Identity asserts of the checkpoint config:

    - layer 4 is sliding_attention, layer 5 is full_attention, the 5:1
      pattern rows of the recorded boundary pair
    - dual rope theta, 1e6 on full layers and 1e4 on sliding layers
    - the 512 sliding window
    """
    model = Gemma3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    cfg = model.config
    assert cfg.layer_types[SLIDING_LAYER_IDX] == "sliding_attention", (
        "layer 4 must be the sliding_attention row of the boundary pair")
    assert cfg.layer_types[FULL_LAYER_IDX] == "full_attention", (
        "layer 5 must be the full_attention row of the boundary pair")
    assert cfg.rope_parameters["full_attention"]["rope_theta"] == 1e6, (
        "the full-attention rope theta must stay the global 1e6")
    assert cfg.rope_parameters["sliding_attention"]["rope_theta"] == 1e4, (
        "the sliding-attention rope theta must stay the local 1e4")
    assert cfg.sliding_window == 512, (
        "the recorded rows assume the 512 sliding window")
    return model


def build_mask(cfg, embeds: torch.Tensor, pos_ids: torch.Tensor,
               layer_type: str):
    """Returns the mask for one layer kind, built through the same masking
    entry points the model forward uses.

    Args:
    - cfg, the parsed config
    - embeds, pos_ids, the bf16 embeds and their position ids
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
                      sin: torch.Tensor, mask) -> dict:
    """Replays the reference Gemma3Attention.forward with intermediate capture,
    asserted equal to the module's own forward before saving.

    Args:
    - attn, the weighted attention module whose forward the replay mirrors
    - h_norm, cos, sin, the post-input-layernorm input and position embeddings
    - mask, None runs the sdpa is_causal path

    Returns:
    - the capture dict, q and k post-norm and post-rope, the expanded kv,
      the sdpa output and the o_proj output
    """
    seq_len = h_norm.shape[1]
    q = attn.q_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    q_normed = attn.q_norm(q)
    k_normed = attn.k_norm(k)
    q_rot, k_rot = apply_rotary_pos_emb(q_normed, k_normed, cos, sin)
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
        "[gen_bf16_gemma3270m_01_layer_internals] manual attention replay "
        "diverged from the module forward")
    return {
        "q_normed": q_normed,
        "k_normed": k_normed,
        "q_rot": q_rot,
        "k_rot": k_rot,
        "v": v,
        "k_expanded": k_expanded,
        "v_expanded": v_expanded,
        "sdpa_output": sdpa_output,
        "attn_output": output,
    }


def layer_chain_capture(layer, x: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, mask, pos_ids: torch.Tensor) -> dict:
    """Runs the full decoder-layer chain over one input, the manual replay
    asserted equal to the module forward before saving.

    Args:
    - layer, the weighted decoder layer whose forward the replay mirrors
    - x, pos_ids, the bf16 input and its position ids
    - cos, sin, mask, the position embeddings and mask of the layer kind

    Returns:
    - the capture dict, every norm output, the attention and mlp outputs
      and the layer output, plus the attention-level capture
    """
    with torch.no_grad():
        h_norm = layer.input_layernorm(x)
        cap = attention_capture(layer.self_attn, h_norm, cos, sin, mask)
        post_attn_normed = layer.post_attention_layernorm(cap["attn_output"])
        h1 = x + post_attn_normed
        h2 = layer.pre_feedforward_layernorm(h1)
        mlp_out = layer.mlp(h2)
        post_ffn_normed = layer.post_feedforward_layernorm(mlp_out)
        layer_out = h1 + post_ffn_normed
        module_out = layer(
            x, position_embeddings=(cos, sin), attention_mask=mask,
            position_ids=pos_ids, past_key_values=None)
    assert torch.equal(module_out, layer_out), (
        "[gen_bf16_gemma3270m_01_layer_internals] manual layer chain diverged "
        "from the module forward")
    cap.update({
        "cos": cos,
        "sin": sin,
        "layer.input_layernorm_output": h_norm,
        "layer.post_attention_layernorm_output": post_attn_normed,
        "layer.pre_feedforward_layernorm_output": h2,
        "layer.mlp_output": mlp_out,
        "layer.post_feedforward_layernorm_output": post_ffn_normed,
        "layer.layer_output": layer_out,
    })
    return cap


def attention_stats_prefix(cap: dict, prefix: str) -> dict:
    """Namespaced stats entries of one attention capture, the rope rows
    included (the reference rotation consumed these bf16 rows)."""
    return {
        prefix + "cos": cap["cos"],
        prefix + "sin": cap["sin"],
        prefix + "q_normed": cap["q_normed"],
        prefix + "k_normed": cap["k_normed"],
        prefix + "q_rot": cap["q_rot"],
        prefix + "k_rot": cap["k_rot"],
        prefix + "v": cap["v"],
        prefix + "k_expanded": cap["k_expanded"],
        prefix + "v_expanded": cap["v_expanded"],
        prefix + "sdpa_output": cap["sdpa_output"],
        prefix + "attn_output": cap["attn_output"],
    }


def generate_sliding_mixture(model: Gemma3ForCausalLM, cfg) -> tuple:
    """Records the sliding_attention row, layer 4, attention op surface plus
    the full decoder-layer chain over one seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under sliding.
    """
    layer = model.model.layers[SLIDING_LAYER_IDX]
    gen = torch.Generator(device="mps")
    gen.manual_seed(SEED_SLIDING)
    x = torch.randn(1, SEQ, cfg.hidden_size, generator=gen, dtype=torch.bfloat16,
                    device="mps")
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids, "sliding_attention")
    mask = build_mask(cfg, x, pos_ids, "sliding_attention")
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids)
    meta = {
        "case": "prefill_seq6",
        "layer": f"model.layers.{SLIDING_LAYER_IDX}",
        "layer_type": "sliding_attention",
        "seq_len": SEQ,
        "seed": SEED_SLIDING,
        "softmax_scaling": float(layer.self_attn.scaling),
        "rope_theta": cfg.rope_parameters["sliding_attention"]["rope_theta"],
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "mask_note": "seq 6 stays inside the 512 window, both mask kinds skip "
            "to the sdpa is_causal path (mask None), the window behavior is "
            "recorded at tier 04",
    }
    payload = OrderedDict([("sliding.input", x)])
    captured = attention_stats_prefix(cap, "sliding.")
    captured.update({
        "sliding.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "sliding.layer.post_attention_layernorm_output":
            cap["layer.post_attention_layernorm_output"],
        "sliding.layer.pre_feedforward_layernorm_output":
            cap["layer.pre_feedforward_layernorm_output"],
        "sliding.layer.mlp_output": cap["layer.mlp_output"],
        "sliding.layer.post_feedforward_layernorm_output":
            cap["layer.post_feedforward_layernorm_output"],
        "sliding.layer.layer_output": cap["layer.layer_output"],
    })
    print("[gen_bf16_gemma3270m_01_layer_internals] sliding mixture "
          f"(seed {SEED_SLIDING})")
    return meta, payload, captured


def generate_full_mixture(model: Gemma3ForCausalLM, cfg) -> tuple:
    """Records the full_attention row, layer 5, attention op surface plus
    the full decoder-layer chain over one seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under full.
    """
    layer = model.model.layers[FULL_LAYER_IDX]
    gen = torch.Generator(device="mps")
    gen.manual_seed(SEED_FULL)
    x = torch.randn(1, SEQ, cfg.hidden_size, generator=gen, dtype=torch.bfloat16,
                    device="mps")
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids, "full_attention")
    mask = build_mask(cfg, x, pos_ids, "full_attention")
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids)
    meta = {
        "case": "prefill_seq6",
        "layer": f"model.layers.{FULL_LAYER_IDX}",
        "layer_type": "full_attention",
        "seq_len": SEQ,
        "seed": SEED_FULL,
        "softmax_scaling": float(layer.self_attn.scaling),
        "rope_theta": cfg.rope_parameters["full_attention"]["rope_theta"],
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "mask_note": "seq 6 stays inside the 512 window, both mask kinds skip "
            "to the sdpa is_causal path (mask None)",
    }
    payload = OrderedDict([("full.input", x)])
    captured = attention_stats_prefix(cap, "full.")
    captured.update({
        "full.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "full.layer.post_attention_layernorm_output":
            cap["layer.post_attention_layernorm_output"],
        "full.layer.pre_feedforward_layernorm_output":
            cap["layer.pre_feedforward_layernorm_output"],
        "full.layer.mlp_output": cap["layer.mlp_output"],
        "full.layer.post_feedforward_layernorm_output":
            cap["layer.post_feedforward_layernorm_output"],
        "full.layer.layer_output": cap["layer.layer_output"],
    })
    print("[gen_bf16_gemma3270m_01_layer_internals] full mixture "
          f"(seed {SEED_FULL})")
    return meta, payload, captured


def generate_boundary_mixture(model: Gemma3ForCausalLM, cfg) -> tuple:
    """Records the boundary pair over one seeded input, the layer-4 sliding
    chain and the layer-5 full chain, the layer-5 chain consumes the layer-4
    output as its input.

    Takes the loaded reference model and its parsed config.

    Returns the mixture metadata, the driving input tensor under its file name,
    and the stats-frame entries namespaced under boundary.
    """
    gen = torch.Generator(device="mps")
    gen.manual_seed(SEED_BOUNDARY)
    x = torch.randn(1, SEQ, cfg.hidden_size, generator=gen, dtype=torch.bfloat16,
                    device="mps")
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    rotary = model.model.rotary_emb
    cos4, sin4 = rotary(x, pos_ids, "sliding_attention")
    mask4 = build_mask(cfg, x, pos_ids, "sliding_attention")
    cap4 = layer_chain_capture(
        model.model.layers[SLIDING_LAYER_IDX], x, cos4, sin4, mask4, pos_ids)
    h5_in = cap4["layer.layer_output"]
    cos5, sin5 = rotary(h5_in, pos_ids, "full_attention")
    mask5 = build_mask(cfg, h5_in, pos_ids, "full_attention")
    cap5 = layer_chain_capture(
        model.model.layers[FULL_LAYER_IDX], h5_in, cos5, sin5, mask5, pos_ids)
    meta = {
        "case": "boundary_pair_seq6",
        "layer4": f"model.layers.{SLIDING_LAYER_IDX}",
        "layer5": f"model.layers.{FULL_LAYER_IDX}",
        "seq_len": SEQ,
        "seed": SEED_BOUNDARY,
        "pair_note": "the layer-5 chain consumes the layer-4 output as its "
            "input, the rope rows for both kinds at positions 0..5 are "
            "recorded under the sliding and full mixtures",
    }
    payload = OrderedDict([("boundary.input", x)])
    captured = {
        "boundary.layer4_output": cap4["layer.layer_output"],
        "boundary.layer5_input_layernorm_output": cap5["layer.input_layernorm_output"],
        "boundary.layer5_post_attention_layernorm_output":
            cap5["layer.post_attention_layernorm_output"],
        "boundary.layer5_pre_feedforward_layernorm_output":
            cap5["layer.pre_feedforward_layernorm_output"],
        "boundary.layer5_mlp_output": cap5["layer.mlp_output"],
        "boundary.layer5_post_feedforward_layernorm_output":
            cap5["layer.post_feedforward_layernorm_output"],
        "boundary.layer5_output": cap5["layer.layer_output"],
    }
    print("[gen_bf16_gemma3270m_01_layer_internals] boundary mixture "
          f"(seed {SEED_BOUNDARY})")
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
    """Records the boundary-pair fixture file set after the RAM guard."""
    recorded_from = os.environ.get("TTT_RECORD_FROM", "m4max-metal")
    check_ram()

    model = load_model()
    cfg = model.config

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    sliding_meta, sliding_payload, sliding_captured = generate_sliding_mixture(model, cfg)
    full_meta, full_payload, full_captured = generate_full_mixture(model, cfg)
    boundary_meta, boundary_payload, boundary_captured = generate_boundary_mixture(model, cfg)

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
        "query_pre_attn_scalar": cfg.query_pre_attn_scalar,
        "sliding_window": cfg.sliding_window,
        "layer_types": cfg.layer_types,
        "rope_parameters": cfg.rope_parameters,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "mixtures": {
            "sliding": sliding_meta,
            "full": full_meta,
            "boundary": boundary_meta,
        },
    }
    save_fixture(metadata, [
        (sliding_payload, sliding_captured),
        (full_payload, full_captured),
        (boundary_payload, boundary_captured),
    ])

    print(f"[gen_bf16_gemma3270m_01_layer_internals] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_gemma3270m_01_layer_internals] wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
