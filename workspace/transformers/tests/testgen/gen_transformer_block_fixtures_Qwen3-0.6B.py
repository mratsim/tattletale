"""
Generate TransformerBlock fixtures with long residual stream pattern for Qwen3-0.6B.

This script:
1. Loads layer 8 weights directly from the main model file
2. Creates Qwen3DecoderLayer with real weights
3. Generates 4 test cases with/without residual stream

Space-saving: Weights are loaded from tests/hf_models/Qwen3-0.6B/model.safetensors
instead of being saved to a separate file.
"""

import os
import torch
from safetensors import safe_open
from safetensors import torch as st
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    Qwen3Config,
)

MODEL_NAME = "Qwen3-0.6B"
LAYER_IDX = 8
FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fixtures", "layers", f"{MODEL_NAME}-block-{LAYER_IDX}"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), f"hf_models/{MODEL_NAME}/model.safetensors"
)
FIXED_SEED = 42


def set_seed(seed: int = FIXED_SEED) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_fixture_dir() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(case_num: int, tensors: dict, metadata: dict) -> str:
    """Save a fixture to safetensors format."""
    filename = f"transformer-block-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    safe_tensors = {}
    for name, tensor in tensors.items():
        if tensor is not None:
            safe_tensors[name] = tensor.detach().cpu().contiguous()

    serialized = st.save(safe_tensors, metadata=metadata)
    with open(filepath, "wb") as f:
        f.write(serialized)

    return filepath


def create_layer_from_weights(config: Qwen3Config, weights: dict) -> Qwen3DecoderLayer:
    """Create Qwen3DecoderLayer initialized with real weights."""
    layer = Qwen3DecoderLayer(config, layer_idx=LAYER_IDX)
    
    # Load weights
    layer.input_layernorm.weight.data = weights["input_layernorm.weight"]
    layer.post_attention_layernorm.weight.data = weights["post_attention_layernorm.weight"]
    layer.mlp.gate_proj.weight.data = weights["mlp.gate_proj.weight"]
    layer.mlp.up_proj.weight.data = weights["mlp.up_proj.weight"]
    layer.mlp.down_proj.weight.data = weights["mlp.down_proj.weight"]
    layer.self_attn.q_proj.weight.data = weights["self_attn.q_proj.weight"]
    layer.self_attn.k_proj.weight.data = weights["self_attn.k_proj.weight"]
    layer.self_attn.v_proj.weight.data = weights["self_attn.v_proj.weight"]
    layer.self_attn.o_proj.weight.data = weights["self_attn.o_proj.weight"]
    layer.self_attn.q_norm.weight.data = weights["self_attn.q_norm.weight"]
    layer.self_attn.k_norm.weight.data = weights["self_attn.k_norm.weight"]
    
    return layer


def generate_transformer_block_fixtures() -> None:
    """Generate fixtures for TransformerBlock with long residual stream pattern."""
    print(f"Generating {MODEL_NAME} TransformerBlock (layer {LAYER_IDX}) fixtures")
    print("=" * 60)

    ensure_fixture_dir()
    set_seed(FIXED_SEED)

    # Load layer weights directly from main model
    print("Loading layer weights from main model...")
    prefix = f"model.layers.{LAYER_IDX}."
    weights = {}
    with safe_open(MODEL_PATH, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                weights[key.replace(prefix, "")] = f.get_tensor(key).clone()

    # Create config and layer
    config = Qwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    config._attn_implementation = "sdpa"
    
    print("Creating Qwen3DecoderLayer with real weights...")
    layer = create_layer_from_weights(config, weights)
    rotary = Qwen3RotaryEmbedding(config)

    # Test cases: (batch, seq, with_residual)
    test_cases = [
        (1, 1, False),  # 00: single token, no residual (first block, decode)
        (2, 8, False),  # 01: short sequence, no residual (first block, prefill)
        (1, 1, True),   # 02: single token, with residual (middle block, decode)
        (2, 8, True),   # 03: short sequence, with residual (middle block, prefill)
    ]

    for case_num, (batch, seq, with_residual) in enumerate(test_cases):
        print(f"Generating case {case_num:02d}: batch={batch}, seq={seq}, with_residual={with_residual}...")
        
        # Generate input hidden states
        input_hidden_states = torch.randn(batch, seq, config.hidden_size, dtype=torch.bfloat16)
        
        # Generate residual (if applicable)
        residual = torch.randn(batch, seq, config.hidden_size, dtype=torch.bfloat16) if with_residual else None
        
        # Generate position IDs and RoPE cos/sin
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1).contiguous()
        cos, sin = rotary(input_hidden_states, position_ids)
        
        # Run forward pass simulating long residual stream pattern
        # This matches the Nim implementation in transformer.nim
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        
        # Get norm layers
        attn_norm = layer.input_layernorm
        mlp_norm = layer.post_attention_layernorm
        
        # Step 1: Handle residual
        if residual is None:
            residual = input_hidden_states.clone()
        
        # Step 2: attn_norm.forward_with_residual(x, residual)
        # In long residual: normalizes (x + residual), returns (normalized, x + residual)
        attn_norm_input = input_hidden_states + residual
        attn_norm_out = attn_norm(attn_norm_input)
        r_after_attn_norm = attn_norm_input  # Passed through unchanged after norm
        
        # Step 3: Attention forward
        attn_out, _ = layer.self_attn(
            attn_norm_out,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )
        
        # Step 4: mlp_norm.forward_with_residual(h + attn_out, residual)
        mlp_norm_input = attn_norm_out + attn_out + r_after_attn_norm
        mlp_norm_out = mlp_norm(mlp_norm_input)
        r_after_mlp_norm = mlp_norm_input  # Passed through unchanged after norm
        
        # Step 5: MLP forward
        mlp_out = layer.mlp(mlp_norm_out)
        
        # Step 6: Final output
        output = mlp_norm_out + mlp_out
        output_residual = r_after_mlp_norm
        
        save_fixture(case_num, {
            "input_hidden_states": input_hidden_states,
            "residual": residual,
            "position_ids": position_ids,
            "cos": cos,
            "sin": sin,
            "output": output,
            "output_residual": output_residual,
        }, metadata={
            "model": MODEL_NAME,
            "layer": f"model.layers.{LAYER_IDX}",
            "case": ["single_token_no_residual", "seq_no_residual", "single_token_with_residual", "seq_with_residual"][case_num],
            "tie_word_embeddings": "true",
        })

    print("=" * 60)
    print(f"Fixture generation complete!")
    print(f"Fixtures saved to: {FIXTURE_DIR}")
    print(f"Note: Weights loaded from main model (no separate weights file)")


if __name__ == "__main__":
    generate_transformer_block_fixtures()