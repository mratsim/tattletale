"""
Generate Safetensor fixtures for transformer layer testing using real Qwen3-0.6B weights.

This script:
1. Loads layer 8 weights directly from the main model file (no separate weights file)
2. Generates test fixtures using those real weights

Space-saving: Weights are loaded from tests/hf_models/Qwen3-0.6B/model.safetensors
instead of being saved to a separate file (~30 MiB saved per layer).

Determinism:
- Each generator calls torch.manual_seed with its own seed constant (SEED_NORM=42, SEED_MLP=43, etc.)
- Fixture files are fully deterministic across separate process invocations
- Metadata saved to separate `.metadata.json` files (safetensors HashMap has randomized order in Rust)
- JSON metadata uses `sort_keys=True` for deterministic serialization
- Use ``--only <name>`` to regenerate a single fixture type without affecting others
"""

import json
from collections import OrderedDict
import os
import sys
import torch
from safetensors import safe_open
from safetensors import torch as st
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

# ── Determinism (called ONCE at import time) ──────────────────────────
# Determinism: cudnn settings for reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = False

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen3-0.6B"
LAYER_IDX = 8
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "layers", f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
MODEL_PATH = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), f"tests/hf_models/{MODEL_NAME}/model.safetensors"
)

# Per-generator seeds — independent, order-agnostic.
SEED_NORM  = 42
SEED_MLP   = 43
SEED_ATTN  = 44
SEED_ROPE  = 45


# ── Helpers ───────────────────────────────────────────────────────────

def ensure_fixture_dir() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def create_layers_from_weights(weights: dict) -> tuple:
    """Create Qwen3 layers initialized with real weights."""
    config = Qwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    # Force SDPA to match our Nim implementation
    config._attn_implementation = "sdpa"

    # Create norm layers with weights
    input_layernorm = Qwen3RMSNorm(1024, eps=1e-6)
    input_layernorm.weight.data = weights["input_layernorm.weight"]

    post_attention_layernorm = Qwen3RMSNorm(1024, eps=1e-6)
    post_attention_layernorm.weight.data = weights["post_attention_layernorm.weight"]

    # Create MLP with weights
    mlp = Qwen3MLP(config)
    mlp.gate_proj.weight.data = weights["mlp.gate_proj.weight"]
    mlp.up_proj.weight.data = weights["mlp.up_proj.weight"]
    mlp.down_proj.weight.data = weights["mlp.down_proj.weight"]

    # Create attention with weights
    attn = Qwen3Attention(config, layer_idx=LAYER_IDX)
    attn.q_proj.weight.data = weights["self_attn.q_proj.weight"]
    attn.k_proj.weight.data = weights["self_attn.k_proj.weight"]
    attn.v_proj.weight.data = weights["self_attn.v_proj.weight"]
    attn.o_proj.weight.data = weights["self_attn.o_proj.weight"]
    attn.q_norm.weight.data = weights["self_attn.q_norm.weight"]
    attn.k_norm.weight.data = weights["self_attn.k_norm.weight"]

    return input_layernorm, post_attention_layernorm, mlp, attn


def save_fixture(layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors format with separate metadata file.

    Metadata is saved to a separate .metadata.json file for determinism.
    safetensors uses HashMap which has randomized iteration order in Rust.
    """
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    # Sort tensors for deterministic serialization (safetensors sorts by dtype then name)
    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in sorted(tensors.items())
        if tensor is not None
    )
    # Save tensors without metadata (deterministic)
    serialized = st.save(sorted_tensors, metadata=None)
    with open(filepath, "wb") as f:
        f.write(serialized)

    # Save metadata to separate JSON file (deterministic with sorted keys)
    metadata_path = filepath + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")

    return filepath


# ── Generators ────────────────────────────────────────────────────────

def generate_norm_fixtures(
    input_layernorm: Qwen3RMSNorm, post_attention_layernorm: Qwen3RMSNorm
) -> None:
    """Generate fixtures for RMSNorm layers using real weights."""
    torch.manual_seed(SEED_NORM)
    layer_name = "norm"

    # Case 00: input_layernorm normal forward
    x = torch.randn(2, 8, 1024, dtype=torch.bfloat16)
    output = input_layernorm(x)
    save_fixture(
        layer_name,
        0,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.input_layernorm",
            "case": "normal_forward",
        },
        {"input_hidden_states": x, "output": output},
    )

    # Case 01: input_layernorm single token
    x = torch.randn(1, 1, 1024, dtype=torch.bfloat16)
    output = input_layernorm(x)
    save_fixture(
        layer_name,
        1,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.input_layernorm",
            "case": "single_token",
        },
        {"input_hidden_states": x, "output": output},
    )

    # Case 02: post_attention_layernorm
    x = torch.randn(2, 4, 1024, dtype=torch.bfloat16)
    output = post_attention_layernorm(x)
    save_fixture(
        layer_name,
        2,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.post_attention_layernorm",
            "case": "normal_forward",
        },
        {"input_hidden_states": x, "output": output},
    )

    # Case 03: Zeros
    x = torch.zeros(2, 4, 1024, dtype=torch.bfloat16)
    output = input_layernorm(x)
    save_fixture(
        layer_name,
        3,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.input_layernorm",
            "case": "zeros_input",
        },
        {"input_hidden_states": x, "output": output},
    )

    print(f"Generated {layer_name} fixtures")


def generate_mlp_fixtures(mlp: Qwen3MLP) -> None:
    """Generate fixtures for MLP layer using real weights."""
    torch.manual_seed(SEED_MLP)
    layer_name = "mlp"

    # Case 00: Normal forward
    x = torch.randn(2, 8, 1024, dtype=torch.bfloat16)
    output = mlp(x)
    save_fixture(
        layer_name,
        0,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.mlp",
            "case": "normal_forward",
        },
        {"input_x": x, "output": output},
    )

    # Case 01: Single token
    x = torch.randn(1, 1, 1024, dtype=torch.bfloat16)
    output = mlp(x)
    save_fixture(
        layer_name,
        1,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.mlp",
            "case": "single_token",
        },
        {"input_x": x, "output": output},
    )

    # Case 02: Short sequence
    x = torch.randn(1, 4, 1024, dtype=torch.bfloat16)
    output = mlp(x)
    save_fixture(
        layer_name,
        2,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.mlp",
            "case": "short_sequence",
        },
        {"input_x": x, "output": output},
    )

    # Case 03: Zeros
    x = torch.zeros(2, 4, 1024, dtype=torch.bfloat16)
    output = mlp(x)
    save_fixture(
        layer_name,
        3,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.mlp",
            "case": "zeros_input",
        },
        {"input_x": x, "output": output},
    )

    print(f"Generated {layer_name} fixtures")


def generate_rope_fixtures(rotary: Qwen3RotaryEmbedding, config) -> None:
    """Generate fixtures for RoPE apply — matches Nim tensor layout."""
    torch.manual_seed(SEED_ROPE)
    layer_name = "rope"

    # Case 00: batch=2, seq=8, full heads (16), GQA k_heads (8)
    batch, seq_len, num_heads, num_kv_heads, head_dim = 2, 8, 16, 8, 128
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
    cos, sin = rotary(torch.randn(batch, seq_len, config.hidden_size, dtype=torch.bfloat16), position_ids)
    # cos/sin from HF: (batch, seq, head_dim)
    # Save 2D (seq, head_dim) for our Nim cache
    cos_2d = cos[0]  # drop batch: (seq, head_dim)
    sin_2d = sin[0]
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16)
    # Use HF's apply_rotary_pos_emb with unsqueeze_dim=2 for 4D tensors
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

    save_fixture(
        layer_name, 0,
        {"model": MODEL_NAME, "layer": f"model.layers.{LAYER_IDX}.self_attn.rotary_emb", "case": "batch2_seq8_gqa"},
        {"q": q, "k": k, "cos": cos_2d, "sin": sin_2d, "q_rot": q_rot, "k_rot": k_rot, "position_ids": position_ids},
    )

    # Case 01: single token
    batch, seq_len = 1, 1
    position_ids = torch.tensor([[0]]).contiguous()
    cos, sin = rotary(torch.randn(batch, seq_len, config.hidden_size, dtype=torch.bfloat16), position_ids)
    cos_2d = cos[0]
    sin_2d = sin[0]
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

    save_fixture(
        layer_name, 1,
        {"model": MODEL_NAME, "layer": f"model.layers.{LAYER_IDX}.self_attn.rotary_emb", "case": "single_token"},
        {"q": q, "k": k, "cos": cos_2d, "sin": sin_2d, "q_rot": q_rot, "k_rot": k_rot, "position_ids": position_ids},
    )

    # Case 02: rotate_half direct test
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    rotated_half = rotate_half(x)
    save_fixture(
        layer_name, 2,
        {"model": MODEL_NAME, "layer": f"model.layers.{LAYER_IDX}.self_attn.rotary_emb.rotate_half", "case": "rotate_half_4d"},
        {"input": x, "output": rotated_half},
    )

    print(f"Generated {layer_name} fixtures")


def generate_attn_fixtures(attn: Qwen3Attention, rotary: Qwen3RotaryEmbedding) -> None:
    """Generate fixtures for attention layer using real weights."""
    torch.manual_seed(SEED_ATTN)
    layer_name = "attn"

    # Case 00: Normal forward
    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, 1024, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
    cos, sin = rotary(hidden_states, position_ids)

    output, _ = attn(
        hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
    )

    save_fixture(
        layer_name,
        0,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.self_attn",
            "case": "normal_forward",
        },
        {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "cos": cos,
            "sin": sin,
            "output": output,
        },
    )

    # Case 01: Single token
    hidden_states = torch.randn(1, 1, 1024, dtype=torch.bfloat16)
    position_ids = torch.tensor([[0]]).contiguous()
    cos, sin = rotary(hidden_states, position_ids)

    output, _ = attn(
        hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
    )

    save_fixture(
        layer_name,
        1,
        {
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}.self_attn",
            "case": "single_token",
        },
        {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "cos": cos,
            "sin": sin,
            "output": output,
        },
    )

    print(f"Generated {layer_name} fixtures")


# ── Generators registry (for --only) ──────────────────────────────────

GENERATORS = {
    "norm": lambda layers: generate_norm_fixtures(*layers["norm"]),
    "mlp":  lambda layers: generate_mlp_fixtures(*layers["mlp"]),
    "rope": lambda layers: generate_rope_fixtures(*layers["rope"]),
    "attn": lambda layers: generate_attn_fixtures(*layers["attn"]),
}


# ── Main ──────────────────────────────────────────────────────────────

def _load_weights() -> dict:
    prefix = f"model.layers.{LAYER_IDX}."
    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                weights[key.replace(prefix, "")] = f.get_tensor(key).clone()
    return weights


def _build_layers(weights: dict):
    input_ln, post_ln, mlp, attn = create_layers_from_weights(weights)
    rotary = Qwen3RotaryEmbedding(Qwen3Config())
    return {
        "norm": (input_ln, post_ln),
        "mlp":  (mlp,),
        "rope": (rotary, Qwen3Config()),
        "attn": (attn, rotary),
    }


def generate_all_fixtures() -> None:
    """Generate all layer fixtures."""
    print(f"Generating {MODEL_NAME} layer {LAYER_IDX} fixtures")
    print("=" * 60)

    ensure_fixture_dir()

    weights = _load_weights()
    layers = _build_layers(weights)

    generate_norm_fixtures(*layers["norm"])
    generate_mlp_fixtures(*layers["mlp"])
    generate_rope_fixtures(*layers["rope"])
    generate_attn_fixtures(*layers["attn"])

    print("=" * 60)
    print(f"Fixture generation complete!")
    print(f"Fixtures saved to: {FIXTURE_DIR}")


def main():
    # --only: regenerate just one fixture type (doesn't touch the rest)
    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "--only":
        if len(sys.argv) < 3:
            print(f"Usage: python {sys.argv[0]} [--only norm|mlp|rope|attn]")
            sys.exit(1)
        only = sys.argv[2]
        if only not in GENERATORS:
            print(f"Unknown: {only}. Available: {', '.join(sorted(GENERATORS))}")
            sys.exit(1)

    if only is None:
        generate_all_fixtures()
    else:
        ensure_fixture_dir()
        weights = _load_weights()
        layers = _build_layers(weights)
        GENERATORS[only](layers)
        print(f"Done: {only} fixtures written to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
