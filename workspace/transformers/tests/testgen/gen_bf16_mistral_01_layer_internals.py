#!/usr/bin/env python3
"""Layer-0 fixture file of the Mistral-7B-v0.1 checkpoint, recorded
with torch bf16 on Metal (mps) under the installed reference modeling.

Single-file grammar with one fixture file per family layer, one bare bf16
driving tensor per mixture, all recorded intermediates live on the stats
frame as fingerprints.

No Qwen3 analog exists for these tier-01 rows. Qwen3 runs one uniform
full-attention kind, this checkpoint is the all-sliding shape:

- the config carries one global sliding_window and no layer_types row
- every decoder layer windows through create_sliding_window_causal_mask

| mixture | row                                                                                  |
| ------- | ------------------------------------------------------------------------------------ |
| sliding | layer 0 (the single mixer variant the family instantiates), the attention op surface |
|         | plus the full decoder-layer chain over one seeded input                              |

| file                                                   | contents                                                         |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| layer0-Mistral-7B-v0.1-00.safetensor                   | sliding.input                                                    |
| layer0-Mistral-7B-v0.1-00.safetensor.metadata.json.zst | per-mixture metadata under the mixtures key                      |
| layer0-Mistral-7B-v0.1-00.safetensor.stats.json.zst    | one uniform record per recorded tensor, keys namespaced sliding. |

At seq 6 the mask skips to the sdpa is_causal path (mask None)
and the window does not constrain these rows, tier-04 carries
the window behavior over a prompt whose prefill crosses 4096.

Consumed by tests/q_bf16/t_bf16_mistral_01_layer_internals.nim, one
assertion block per mixture.

Run from the worktree root

  uv run python workspace/transformers/tests/testgen/gen_bf16_mistral_01_layer_internals.py

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
from transformers import MistralForCausalLM  # noqa: E402
from transformers.masking_utils import create_sliding_window_causal_mask  # noqa: E402
from transformers.models.mistral.modeling_mistral import (  # noqa: E402
    apply_rotary_pos_emb,
    repeat_kv,
)

from fixture_stats import write_stats_file  # noqa: E402, the path insert precedes the import

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity constants of the checkpoint.
MODEL_NAME = "Mistral-7B-v0.1"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(TESTS_DIR, "hf_models", MODEL_NAME)
FIXTURE_DIR = os.path.join(
    TESTS_DIR, "fixtures", "bf16-01-layer-internals", f"{MODEL_NAME}-layer-0"
)
FIXTURE_STEM = "layer0-" + MODEL_NAME + "-00"
FIXTURE_PATH = os.path.join(FIXTURE_DIR, FIXTURE_STEM + ".safetensor")

NUM_THREADS = 1

# Recorded row, layer 0, the single mixer variant this family instantiates.
# The all-sliding shape (no full layer exists) is proven by the config
# asserts of load_model and the metadata, not by a second row.
LAYER_IDX = 0

SEED_SLIDING = 301

SEQ = 6

MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit(
        "[gen_bf16_mistral_01_layer_internals] vm_stat gave no page size line")


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
            "[gen_bf16_mistral_01_layer_internals] vm_stat gave no pool lines")
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
            f"[gen_bf16_mistral_01_layer_internals] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor, stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_mistral_01_layer_internals] other python/torch processes hold RAM: {stray}, "
            "stop and retry when idle")


def load_model() -> MistralForCausalLM:
    """Loads the full reference model through the installed from_pretrained,
    bf16, eval, Metal (mps).

    Returns:
    - the model in eval mode, its layer 0 decoder layer plus the shared
      single-theta rotary feed every mixture consumes

    Identity asserts of the checkpoint config, all-sliding shape:

    - one global sliding_window 4096 with no layer_types row, the model
      forward builds create_sliding_window_causal_mask for every layer,
      no full layer exists
    - the single rope theta 1e4 of the flat rope_parameters spelling
    - 32 q heads over 8 kv heads, head_dim 128
    """
    model = MistralForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.to("mps")
    model.eval()
    cfg = model.config
    assert getattr(cfg, "layer_types", None) is None, (
        "the all-sliding shape carries no layer_types row, every layer windows")
    assert cfg.sliding_window == 4096, (
        "the recorded rows assume the 4096 sliding window on every layer")
    assert cfg.rope_parameters["rope_theta"] == 1e4, (
        "the rope theta must stay the single global 1e4")
    assert cfg.head_dim == 128 and cfg.num_attention_heads == 32 \
        and cfg.num_key_value_heads == 8, (
        "the recorded rows assume 32 q heads / 8 kv heads over head_dim 128")
    assert cfg.num_hidden_layers == 32 and cfg.vocab_size == 32000, (
        "the recorded rows assume the 32-layer, 32000-vocab checkpoint")
    return model


def build_mask(cfg, embeds: torch.Tensor, pos_ids: torch.Tensor):
    """Returns the sliding-window causal mask through the same masking entry
    point the model forward uses for every layer.

    Args:
    - cfg, embeds, pos_ids, the parsed config, the bf16 embeds, the matching position ids

    Returns None when the sdpa path skips to is_causal.
    """
    return create_sliding_window_causal_mask(
        config=cfg, inputs_embeds=embeds, attention_mask=None,
        past_key_values=None, position_ids=pos_ids)


def attention_capture(attn, h_norm: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor, mask) -> dict:
    """Replays the reference MistralAttention.forward with intermediate capture,
    asserted equal to the module's own forward before saving.

    Args:
    - attn, the attention module whose forward the replay mirrors
    - h_norm, cos, sin, the post-input-layernorm input and position embeddings
    - mask, None runs the sdpa is_causal path

    Returns:
    - the capture dict, q and k post-rope, the expanded kv, the sdpa output
      and the o_proj output
    """
    seq_len = h_norm.shape[1]
    q = attn.q_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(h_norm).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
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
        "[gen_bf16_mistral_01_layer_internals] manual attention replay "
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
                        sin: torch.Tensor, mask, pos_ids: torch.Tensor) -> dict:
    """Runs the full decoder-layer chain over one input, the manual replay
    asserted equal to the module forward before saving.

    Args:
    - layer, the decoder layer whose forward the replay mirrors
    - x, pos_ids, the bf16 input and its position ids
    - cos, sin, mask, the position embeddings and the sliding mask

    Returns:
    - the capture dict, every norm output, the attention and mlp outputs
      and the layer output, plus the attention-level capture
    """
    with torch.no_grad():
        h_norm = layer.input_layernorm(x)
        cap = attention_capture(layer.self_attn, h_norm, cos, sin, mask)
        h1 = x + cap["attn_output"]
        h2 = layer.post_attention_layernorm(h1)
        mlp_out = layer.mlp(h2)
        layer_out = h1 + mlp_out
        module_out = layer(
            x, attention_mask=mask, position_ids=pos_ids,
            position_embeddings=(cos, sin), past_key_values=None)
    assert torch.equal(module_out, layer_out), (
        "[gen_bf16_mistral_01_layer_internals] manual layer chain diverged "
        "from the module forward")
    cap.update({
        "cos": cos,
        "sin": sin,
        "layer.input_layernorm_output": h_norm,
        "layer.post_attention_layernorm_output": h2,
        "layer.mlp_output": mlp_out,
        "layer.layer_output": layer_out,
    })
    return cap


def attention_stats_prefix(cap: dict, prefix: str) -> dict:
    """Namespaced stats entries of one attention capture, the rope rows
    included (the reference rotation consumed these bf16 rows)."""
    return {
        prefix + "cos": cap["cos"],
        prefix + "sin": cap["sin"],
        prefix + "q_rot": cap["q_rot"],
        prefix + "k_rot": cap["k_rot"],
        prefix + "v": cap["v"],
        prefix + "k_expanded": cap["k_expanded"],
        prefix + "v_expanded": cap["v_expanded"],
        prefix + "sdpa_output": cap["sdpa_output"],
        prefix + "attn_output": cap["attn_output"],
    }


def generate_sliding_mixture(model: MistralForCausalLM, cfg) -> tuple:
    """Records the layer-0 sliding row, attention op surface plus the full
    decoder-layer chain over one seeded input.

    Args:
    - model, cfg, the loaded reference model and its parsed config

    Returns:
    - meta, the mixture metadata
    - payload, the single driving input tensor under its file name
    - captured, the stats-frame entries namespaced under sliding.
    """
    layer = model.model.layers[LAYER_IDX]
    gen = torch.Generator(device="mps")
    gen.manual_seed(SEED_SLIDING)
    x = torch.randn(1, SEQ, cfg.hidden_size, generator=gen, dtype=torch.bfloat16,
                    device="mps")
    pos_ids = torch.arange(SEQ, device="mps").unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, pos_ids)
    mask = build_mask(cfg, x, pos_ids)
    cap = layer_chain_capture(layer, x, cos, sin, mask, pos_ids)
    meta = {
        "case": "prefill_seq6",
        "layer": f"model.layers.{LAYER_IDX}",
        "layer_type": "sliding_attention",
        "seq_len": SEQ,
        "seed": SEED_SLIDING,
        "softmax_scaling": float(layer.self_attn.scaling),
        "rope_theta": cfg.rope_parameters["rope_theta"],
        "rope_reference_cast": "the rotary computes cos/sin f32 and casts to "
            "the hidden dtype before the rotation, the recorded rows are the "
            "bf16 rows the reference rotation consumed",
        "mask_note": "seq 6 stays inside the 4096 window, the sliding mask "
            "skips to the sdpa is_causal path (mask None), the window behavior "
            "is recorded at tier 04",
    }
    payload = OrderedDict([("sliding.input", x)])
    captured = attention_stats_prefix(cap, "sliding.")
    captured.update({
        "sliding.layer.input_layernorm_output": cap["layer.input_layernorm_output"],
        "sliding.layer.post_attention_layernorm_output":
            cap["layer.post_attention_layernorm_output"],
        "sliding.layer.mlp_output": cap["layer.mlp_output"],
        "sliding.layer.layer_output": cap["layer.layer_output"],
    })
    print("[gen_bf16_mistral_01_layer_internals] sliding mixture "
          f"(seed {SEED_SLIDING})")
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


def save_fixture(metadata: dict, payload: dict, captured: dict) -> None:
    """Writes the single-file fixture set, one safetensors payload carrying
    the bare named driving tensor, one metadata sidecar and one stats
    sidecar over every recorded tensor.

    Args:
    - metadata, the metadata frame
    - payload, the driving tensor carrying its file name
    - captured, the recorded tensors carrying their namespaced stats keys
    """
    file_tensors = OrderedDict()
    stats_entries = []
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
    """Records the layer-0 fixture file set after the RAM guard."""
    recorded_from = os.environ.get("TTT_RECORD_FROM", "m4max-metal")
    check_ram()

    model = load_model()
    cfg = model.config

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    sliding_meta, sliding_payload, sliding_captured = generate_sliding_mixture(model, cfg)

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
        "attention_shape": "all-sliding, one global sliding_window and no "
            "layer_types row, every decoder layer windows through "
            "create_sliding_window_causal_mask, no full layer exists",
        "rope_parameters": cfg.rope_parameters,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "mixtures": {
            "sliding": sliding_meta,
        },
    }
    save_fixture(metadata, sliding_payload, sliding_captured)

    print(f"[gen_bf16_mistral_01_layer_internals] torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"[gen_bf16_mistral_01_layer_internals] wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
