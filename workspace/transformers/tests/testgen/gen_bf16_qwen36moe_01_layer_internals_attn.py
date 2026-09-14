#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B gated full-attention layer-3 fixtures
on CPU torch bf16, from the real checkpoint safetensors files.

Generated under tests/fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-3/:

  - attn-00/attn-01, the two gated full-attention replay cases of layer 3
  - the payload carries the suite-read driving tensors, hidden_states plus position_ids
  - the 004 stats frame carries the q/k norm tensors, the sigmoid gate, the gated attention output and the o_proj output

Consumed by tests/q_bf16/t_bf16_qwen36moe_01_layer_internals_attn.nim.

Replay contract:

  - the attention forward is replayed step by step with the reference ops
  - the intermediates reach the stats frame
  - the replay output is asserted against the module's own forward before saving

Run twice, cmp proves byte determinism:
  cd <worktree root> && .venv/bin/python workspace/transformers/tests/testgen/gen_bf16_qwen36moe_01_layer_internals_attn.py

RAM:

  - the script refuses to load weights when free memory sits below the floor
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


import torch  # noqa: E402

from fixture_stats import assert_path_equivalent  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors import torch as st


import transformers  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeAttention,
    Qwen3_5MoeTextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

# Determinism, single intra-op thread, deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Checkpoint, fixture and config paths.
MODEL_NAME = "Qwen3.6-35B-A3B"
LAYER_IDX = 3
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-01-layer-internals", "Qwen3.6-35B-A3B-layer-3"
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
WEIGHTS_FILE_3 = os.path.join(MODEL_DIR, "model-00003-of-00026.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

NUM_THREADS = 1

# Per-generator seeds, independent and order-agnostic.
SEED_ATTN = 82

PREFIX = f"model.language_model.layers.{LAYER_IDX}.self_attn."
MIN_FREE_BYTES = 8 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals_attn] vm_stat gave no page size line")


def free_bytes() -> int:
    """Free physical memory in bytes from `vm_stat`."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    for line in out.stdout.splitlines():
        if line.startswith("Pages free:"):
            pages = int(line.split()[2].rstrip("."))
            return pages * page
    raise SystemExit("[gen_bf16_qwen36moe_01_layer_internals_attn] vm_stat gave no 'Pages free' line")


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
            f"[gen_bf16_qwen36moe_01_layer_internals_attn] free memory {free / 1024 ** 3:.1f} GiB below "
            f"the {MIN_FREE_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_01_layer_internals_attn] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def load_text_config() -> Qwen3_5MoeTextConfig:
    """Load the nested text_config from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeTextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


# Attention geometry of the checkpoint, config is king, the parsed values
# equal the 16/2/256/2048/64 tuple the checkpoint carries.
_CFG = load_text_config()
NUM_QO_HEADS = _CFG.num_attention_heads
NUM_KV_HEADS = _CFG.num_key_value_heads
HEAD_DIM = _CFG.head_dim
HIDDEN = _CFG.hidden_size
ROTARY_DIM = int(_CFG.head_dim * _CFG.partial_rotary_factor)


def load_layer3_weights() -> dict:
    """Load the six layer-3 self_attn tensors from the safetensors file
    that holds them (memory-mapped, only these tensors are copied).

    Returns:
    - the weight dict keyed with the PREFIX-stripped suffixes
    """
    weights = {}
    with safe_open(WEIGHTS_FILE_3, framework="pt") as f:
        for key in f.keys():
            if key.startswith(PREFIX):
                weights[key[len(PREFIX):]] = f.get_tensor(key).clone()
    return weights


def build_attention(weights: dict, cfg: Qwen3_5MoeTextConfig) -> Qwen3_5MoeAttention:
    """Build the layer-3 attention module with real weights.

    Args:
    - weights, the PREFIX-stripped weight dict from load_layer3_weights
    - cfg, the parsed text config

    Returns:
    - the attention module in eval mode
    """
    attn = Qwen3_5MoeAttention(cfg, layer_idx=LAYER_IDX)
    with torch.no_grad():
        attn.q_proj.weight.data = weights["q_proj.weight"]
        attn.k_proj.weight.data = weights["k_proj.weight"]
        attn.v_proj.weight.data = weights["v_proj.weight"]
        attn.o_proj.weight.data = weights["o_proj.weight"]
        attn.q_norm.weight.data = weights["q_norm.weight"]
        attn.k_norm.weight.data = weights["k_norm.weight"]
    attn.eval()
    return attn


def attention_forward_capture(attn, hidden_states, position_embeddings):
    """Replay of the reference Qwen3_5MoeAttention.forward with intermediate capture.

    The replay copies the reference forward body op for op (sdpa interface):
    the q and sigmoid-gate chunk, Gemma qk-norm, partial rope, repeat_kv,
    torch SDPA, sigmoid gating, o_proj.

    Args:
    - attn, hidden_states, position_embeddings, the module and its inputs

    Returns:
    - (output, attn_output_gated, gate, q_normed, k_normed, q_rot, k_rot)
    - the caller asserts output equals the module's own forward output
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states, gate = torch.chunk(
        attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    q_normed = attn.q_norm(query_states.view(hidden_shape))  # (b, s, heads, dim)
    query_states = q_normed.transpose(1, 2)
    k_normed = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))  # (b, s, kv, dim)
    key_states = k_normed.transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    q_rot = query_states
    k_rot = key_states

    key_states_r = repeat_kv(key_states, attn.num_key_value_groups)
    value_states_r = repeat_kv(value_states, attn.num_key_value_groups)

    is_causal = query_states.shape[2] > 1 and attn.is_causal
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states_r, value_states_r,
        attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=attn.scaling,
    )

    # sdpa_attention_forward transposes back to (batch, seq, heads, dim)
    # and makes the result contiguous before the model reshapes to (batch, seq, -1).
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output_gated = attn_output * torch.sigmoid(gate)
    output = attn.o_proj(attn_output_gated)
    return output, attn_output_gated, gate, q_normed, k_normed, q_rot, k_rot


def generate_attn_fixtures(attn: Qwen3_5MoeAttention, rotary: Qwen3_5MoeTextRotaryEmbedding) -> None:
    """Records the two gated full-attention replay cases of layer 3.

    Args:
    - attn, the weighted attention module
    - rotary, the checkpoint text rotary module
    """
    torch.manual_seed(SEED_ATTN)
    layer_name = "attn"

    cases = [
        (0, "prefill_seq8", (1, 8), [[0, 1, 2, 3, 4, 5, 6, 7]]),
        (1, "decode_single_token_pos5", (1, 1), [[5]]),
    ]
    for case_num, case, shape, pos_rows in cases:
        batch, seq_len = shape
        hidden_states = torch.randn(batch, seq_len, HIDDEN, dtype=torch.bfloat16)
        position_ids = torch.tensor(pos_rows).reshape(batch, seq_len).contiguous()
        cos, sin = rotary(hidden_states, position_ids)

        # Real forward (ground truth).
        output_real, _ = attn(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )

        # Replay with capture, asserted against the real forward within
        # the path-equivalence guard below.
        output_cap, _, _, _, _, _, _ = (
            attention_forward_capture(attn, hidden_states, (cos, sin))
        )
        assert_path_equivalent(output_real, output_cap,
            f"attention replay vs the real forward for case {case_num}")

        save_fixture(
            layer_name, case_num,
            {
                "model": MODEL_NAME,
                "layer": PREFIX,
                "case": case,
                "num_qo_heads": NUM_QO_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "rotary_dim": ROTARY_DIM,
                "hidden_size": HIDDEN,
                "seed": SEED_ATTN,
                "num_threads": NUM_THREADS,
                "dtype": "bfloat16",
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            },
            {
                # the suite-read driving tensors only, the replay
                # intermediates stay on the 004 stats frame
                "hidden_states": hidden_states,
                "position_ids": position_ids,
            },
        )
    print(f"Generated {layer_name} fixtures")


def save_fixture(layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
    """Save one fixture payload with a separate deterministic metadata frame."""
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
    return filepath


def main() -> None:
    """Records the layer-3 attention fixtures."""
    check_ram()

    cfg = load_text_config()
    weights = load_layer3_weights()
    attn = build_attention(weights, cfg)
    rotary = Qwen3_5MoeTextRotaryEmbedding(cfg)

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    generate_attn_fixtures(attn, rotary)

    print(f"[gen_bf16_qwen36moe_01_layer_internals_attn] torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"[gen_bf16_qwen36moe_01_layer_internals_attn] wrote {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
