"""
Generate fixtures for the Qwen3.5-0.8B gated full-attention layer (layer 3)
using the VENDORED prod transformers modeling on CPU torch bf16.

Reference: gen_03_layer_fixtures_Qwen3-0.6B.py conventions.

What is generated (under tests/fixtures/layers/Qwen3.5-0.8B-layer-3/):

  rope-*   partial rotary: q/k (bf16) + positions + cos/sin (bf16, 64 wide)
           -> q_rot/k_rot via the vendored apply_rotary_pos_emb
  norm-*   GemmaRMSNorm (1+w): x (bf16) -> norm(x), real q_norm weight
  attn-*   gated full attention layer 3: x -> output (post o_proj), plus
           intermediates (q_normed, k_normed, q_rot, k_rot, gate,
           attn_output_gated)

Determinism: torch.manual_seed per generator, cudnn flags, and metadata in
separate .metadata.json files (safetensors HashMap order is randomized).

The attention forward is replayed step by step with the vendored ops so the
intermediates can be captured. The replay output is asserted bit-identical to
the module's own forward before saving.
"""

import json
from collections import OrderedDict
import os
import sys
import torch
from safetensors import safe_open
from safetensors import torch as st

# Vendored prod transformers is the source of truth.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
VENDORED_SRC = os.environ.get(
    "QWEN35_VENDORED_SRC",
    os.path.join(_REPO_ROOT, "_references_prod", "transformers", "src"))
if not os.path.isdir(VENDORED_SRC):
    raise SystemExit(
        f"[gen_qwen3_5_attn_fixtures] vendored modeling not found at {VENDORED_SRC}. "
        "Set QWEN35_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, VENDORED_SRC)

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

# Determinism (called once at import time).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config.
MODEL_NAME = "Qwen3.5-0.8B"
LAYER_IDX = 3
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "layers", f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), f"tests/hf_models/{MODEL_NAME}"
)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Per-generator seeds, independent and order-agnostic.
SEED_NORM = 42
SEED_ATTN = 44
SEED_ROPE = 45

NUM_QO_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
HIDDEN = 1024
ROTARY_DIM = 64


def load_text_config() -> Qwen3_5TextConfig:
    """Load the nested text_config from the wrapper config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5TextConfig.from_dict(wrapper["text_config"])
    cfg._attn_implementation = "sdpa"
    return cfg


def ensure_fixture_dir() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata file."""
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

    metadata_path = filepath + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")
    return filepath


def load_layer3_weights() -> dict:
    """Load the six layer-3 self_attn tensors from the single shard."""
    prefix = f"model.language_model.layers.{LAYER_IDX}.self_attn."
    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                weights[key[len(prefix):]] = f.get_tensor(key).clone()
    return weights


def build_attention(weights: dict) -> Qwen3_5Attention:
    """Qwen3_5Attention with real layer-3 weights loaded."""
    config = load_text_config()
    attn = Qwen3_5Attention(config, layer_idx=LAYER_IDX)
    attn.q_proj.weight.data = weights["q_proj.weight"]
    attn.k_proj.weight.data = weights["k_proj.weight"]
    attn.v_proj.weight.data = weights["v_proj.weight"]
    attn.o_proj.weight.data = weights["o_proj.weight"]
    attn.q_norm.weight.data = weights["q_norm.weight"]
    attn.k_norm.weight.data = weights["k_norm.weight"]
    return attn


# ── Partial rotary ─────────────────────────────────────────────────────────

def generate_rope_fixtures(rotary: Qwen3_5TextRotaryEmbedding, config) -> None:
    """Partial-rope apply fixtures: only the first 64 of 256 dims rotate."""
    torch.manual_seed(SEED_ROPE)
    layer_name = "rope"

    def run_case(case_num, metadata, batch, seq_len, position_ids):
        position_ids = torch.tensor(position_ids).reshape(batch, seq_len).contiguous()
        dummy = torch.randn(batch, seq_len, HIDDEN, dtype=torch.bfloat16)
        cos, sin = rotary(dummy, position_ids)  # (batch, seq, 64) bf16
        cos_2d = cos[0].contiguous()  # (seq, 64)
        sin_2d = sin[0].contiguous()
        q = torch.randn(batch, seq_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        k = torch.randn(batch, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        save_fixture(
            layer_name, case_num, metadata,
            {
                "q": q, "k": k,
                "cos": cos_2d, "sin": sin_2d,
                "q_rot": q_rot, "k_rot": k_rot,
                "position_ids": position_ids,
            },
        )

    # Case 00: prefill, batch 2, seq 8, positions 0..7.
    run_case(
        0,
        {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{LAYER_IDX}.self_attn",
            "case": "prefill_batch2_seq8",
            "rotary_dim": ROTARY_DIM,
            "theta": 10000000,
        },
        2, 8, [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
    )

    # Case 01: decode, single token at nonzero position.
    run_case(
        1,
        {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{LAYER_IDX}.self_attn",
            "case": "decode_single_token_pos5",
            "rotary_dim": ROTARY_DIM,
            "theta": 10000000,
        },
        1, 1, [[5]],
    )

    # Case 02: scattered positions (index_select path, large angles).
    run_case(
        2,
        {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{LAYER_IDX}.self_attn",
            "case": "scattered_positions",
            "rotary_dim": ROTARY_DIM,
            "theta": 10000000,
        },
        1, 4, [[3, 17, 255, 4096]],
    )

    # Case 03: rotate_half on a 64-wide slice (pair split over the partial dim).
    torch.manual_seed(SEED_ROPE)
    x = torch.randn(2, 8, NUM_QO_HEADS, ROTARY_DIM, dtype=torch.bfloat16)
    rotated = rotate_half(x)
    save_fixture(
        layer_name, 3,
        {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{LAYER_IDX}.self_attn",
            "case": "rotate_half_64wide",
        },
        {"input": x, "output": rotated},
    )
    print(f"Generated {layer_name} fixtures")


# ── GemmaRMSNorm ───────────────────────────────────────────────────────────

def generate_norm_fixtures(q_norm: Qwen3_5RMSNorm, weights: dict) -> None:
    """GemmaRMSNorm (1+w) fixtures using the real q_norm weight."""
    torch.manual_seed(SEED_NORM)
    layer_name = "norm"

    cases = [
        (0, "head_dim_forward", torch.randn(2, 8, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16)),
        (1, "single_token", torch.randn(1, 1, HEAD_DIM, dtype=torch.bfloat16)),
        (2, "zeros_input", torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16)),
    ]
    for case_num, case, x in cases:
        output = q_norm(x)
        save_fixture(
            layer_name, case_num,
            {
                "model": MODEL_NAME,
                "layer": f"model.language_model.layers.{LAYER_IDX}.self_attn.q_norm",
                "case": case,
                "eps": q_norm.eps,
            },
            {"input": x, "output": output, "weight": q_norm.weight.data},
        )
    print(f"Generated {layer_name} fixtures")


# ── Gated full attention ───────────────────────────────────────────────────

def attention_forward_capture(attn, hidden_states, position_embeddings):
    """Replay of the vendored Qwen3_5Attention.forward with intermediate capture.

    The replay copies the vendored forward body op for op (modeling_qwen3_5.py
    Qwen3_5Attention.forward, sdpa interface): q|gate chunk, Gemma qk-norm,
    partial rope, repeat_kv, torch SDPA, sigmoid gate, o_proj. The caller
    asserts the replayed output equals the module's own forward output.
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

    # sdpa_attention_forward transposes back to (batch, seq, heads, dim) and
    # makes the result contiguous before the model reshapes to (batch, seq, -1).
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output_gated = attn_output * torch.sigmoid(gate)
    output = attn.o_proj(attn_output_gated)
    return output, attn_output_gated, gate, q_normed, k_normed, q_rot, k_rot


def generate_attn_fixtures(attn: Qwen3_5Attention, rotary: Qwen3_5TextRotaryEmbedding) -> None:
    """Gated full-attention fixtures with real weights (layer 3)."""
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

        # Replay with capture. Must be bit-identical to the real forward.
        output_cap, attn_output_gated, gate, q_normed, k_normed, q_rot, k_rot = (
            attention_forward_capture(attn, hidden_states, (cos, sin))
        )
        assert torch.equal(output_real, output_cap), (
            f"replay diverged from real forward for case {case_num}"
        )

        save_fixture(
            layer_name, case_num,
            {
                "model": MODEL_NAME,
                "layer": f"model.language_model.layers.{LAYER_IDX}.self_attn",
                "case": case,
                "num_qo_heads": NUM_QO_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "head_dim": HEAD_DIM,
                "rotary_dim": ROTARY_DIM,
            },
            {
                "hidden_states": hidden_states,
                "position_ids": position_ids,
                "cos": cos, "sin": sin,
                "q_normed": q_normed, "k_normed": k_normed,
                "q_rot": q_rot, "k_rot": k_rot,
                "gate": gate,
                "attn_output_gated": attn_output_gated,
                "output": output_real,
            },
        )
    print(f"Generated {layer_name} fixtures")


def main() -> None:
    print(f"Generating {MODEL_NAME} layer {LAYER_IDX} fixtures")
    print("=" * 60)
    ensure_fixture_dir()

    config = load_text_config()
    weights = load_layer3_weights()
    attn = build_attention(weights)
    q_norm = attn.q_norm
    rotary = Qwen3_5TextRotaryEmbedding(config)

    generate_rope_fixtures(rotary, config)
    generate_norm_fixtures(q_norm, weights)
    generate_attn_fixtures(attn, rotary)

    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
