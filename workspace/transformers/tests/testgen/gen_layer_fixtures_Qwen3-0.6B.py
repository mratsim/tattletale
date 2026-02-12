"""
Generate safetensor fixtures for transformer layer testing using real Qwen3-0.6B weights.

This script:
1. Extracts layer 8 weights from the model
2. Saves them to Qwen3-0.6B.model.layers.8.safetensors
3. Generates test fixtures using those real weights
"""

import json
import os
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
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

MODEL_NAME = "Qwen3-0.6B"
LAYER_IDX = 8
FIXTURE_DIR = os.path.join(os.path.dirname(__file__).parent(), "fixtures", "layers", f"{MODEL_NAME}-layer-{LAYER_IDX}")
WEIGHTS_FILE = f"{FIXTURE_DIR}/Weights-{MODEL_NAME}-layer-{LAYER_IDX}.safetensor"
MODEL_PATH = f"tests/hf_models/{MODEL_NAME}/model.safetensors" # Assuming a very small model were everything fits in a safetensor
FIXED_SEED = 42


def set_seed(seed: int = FIXED_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_fixture_dir() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def extract_layer_weights() -> dict:
    """Extract weights for a specific layer from the model safetensors."""
    prefix = f"model.layers.{LAYER_IDX}."
    weights = {}

    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                weights[new_key] = f.get_tensor(key).clone()

    return weights


def save_layer_weights(weights: dict) -> str:
    """Save layer weights to a separate safetensors file."""
    serialized = st.save(weights)
    with open(WEIGHTS_FILE, "wb") as f:
        f.write(serialized)

    print(f"Saved layer weights to {WEIGHTS_FILE}")
    return WEIGHTS_FILE


def load_layer_weights() -> dict:
    """Load layer weights from the separate safetensors file."""
    with safe_open(WEIGHTS_FILE, framework="pt") as f:
        weights = {key: f.get_tensor(key).clone() for key in f.keys()}

    return weights


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
    """Save a fixture to safetensors format."""
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    safe_tensors = {}
    for name, tensor in tensors.items():
        if tensor is not None:
            safe_tensors[name] = tensor.detach().cpu().contiguous()

    serialized = st.save(safe_tensors, metadata=metadata)
    with open(filepath, "wb") as f:
        f.write(serialized)

    return filepath


def generate_norm_fixtures(
    input_layernorm: Qwen3RMSNorm, post_attention_layernorm: Qwen3RMSNorm
) -> None:
    """Generate fixtures for RMSNorm layers using real weights."""
    set_seed(FIXED_SEED)
    layer_name = "norm"

    # Case 00: input_layernorm normal forward
    x = torch.randn(2, 8, 1024, dtype=torch.bfloat16)
    # print(f"input_x[0, 0:5, 0:5]:\n{x[0, 0:5, 0:5]}")
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
    set_seed(FIXED_SEED + 1)
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


def generate_attn_fixtures(attn: Qwen3Attention, rotary: Qwen3RotaryEmbedding) -> None:
    """Generate fixtures for attention layer using real weights."""
    set_seed(FIXED_SEED + 2)
    layer_name = "attn"

    num_kv_groups = attn.num_key_value_groups

    # Case 00: Normal forward
    batch, seq_len = 2, 8
    hidden_states = torch.randn(batch, seq_len, 1024, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
    cos, sin = rotary(hidden_states, position_ids)

    query_states = attn.q_norm(
        attn.q_proj(hidden_states).view(batch, seq_len, -1, 128)
    ).transpose(1, 2)
    key_states = attn.k_norm(
        attn.k_proj(hidden_states).view(batch, seq_len, -1, 128)
    ).transpose(1, 2)
    value_states = (
        attn.v_proj(hidden_states).view(batch, seq_len, -1, 128).transpose(1, 2)
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
    value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

    attn_out = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True
    )
    output = attn.o_proj(attn_out.transpose(1, 2).reshape(batch, seq_len, -1))

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
            "query_states": query_states,
            "key_states": key_states,
            "value_states": value_states,
            "output": output,
        },
    )

    # Case 01: Single token
    hidden_states = torch.randn(1, 1, 1024, dtype=torch.bfloat16)
    position_ids = torch.tensor([[0]]).contiguous()
    cos, sin = rotary(hidden_states, position_ids)

    query_states = attn.q_norm(
        attn.q_proj(hidden_states).view(1, 1, -1, 128)
    ).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden_states).view(1, 1, -1, 128)).transpose(
        1, 2
    )
    value_states = attn.v_proj(hidden_states).view(1, 1, -1, 128).transpose(1, 2)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
    value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

    attn_out = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True
    )
    output = attn.o_proj(attn_out.transpose(1, 2).reshape(1, 1, -1))

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
            "query_states": query_states,
            "key_states": key_states,
            "value_states": value_states,
            "output": output,
        },
    )

    print(f"Generated {layer_name} fixtures")


def generate_all_fixtures() -> None:
    """Generate all layer fixtures."""
    print(f"Generating {MODEL_NAME} layer {LAYER_IDX} fixtures")
    print("=" * 60)

    ensure_fixture_dir()

    # Step 1: Extract and save layer weights
    print("Extracting layer weights...")
    weights = extract_layer_weights()
    save_layer_weights(weights)

    # Step 2: Create layers with real weights
    print("Creating layers with real weights...")
    input_layernorm, post_attention_layernorm, mlp, attn = create_layers_from_weights(
        weights
    )
    rotary = Qwen3RotaryEmbedding(Qwen3Config())

    # Step 3: Generate fixtures
    generate_norm_fixtures(input_layernorm, post_attention_layernorm)
    generate_mlp_fixtures(mlp)
    generate_attn_fixtures(attn, rotary)

    print("=" * 60)
    print(f"Fixture generation complete!")
    print(f"Weights saved to: {WEIGHTS_FILE}")
    print(f"Fixtures saved to: {FIXTURE_DIR}")


if __name__ == "__main__":
    generate_all_fixtures()
