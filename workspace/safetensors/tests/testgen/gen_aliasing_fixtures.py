#!/usr/bin/env python3
"""
Generate minimal fixtures for tensor aliasing bug reproduction.

These fixtures are standalone and don't depend on the full Qwen model.
"""

import os
import torch
import safetensors.torch as st

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "aliasing")

def main():
    print("=== Generating Minimal Fixtures for Tensor Aliasing Bug ===")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Step 1: Generate model file with weights (simulates the main model)
    model_path = os.path.join(OUTPUT_DIR, "model.safetensor")
    print(f"Generating model file: {model_path}")
    
    # Create weight tensors (RMSNorm weights)
    input_ln_weight = torch.ones(64, dtype=torch.float32)
    post_attn_weight = torch.ones(64, dtype=torch.float32)
    
    model_tensors = {
        "input_layernorm.weight": input_ln_weight,
        "post_attention_layernorm.weight": post_attn_weight,
    }
    
    with open(model_path, "wb") as f:
        f.write(st.save(model_tensors))
    
    print(f"  input_layernorm.weight shape: {input_ln_weight.shape}")
    print(f"  post_attention_layernorm.weight shape: {post_attn_weight.shape}")
    print()

    # Step 2: Generate fixture files with different input shapes
    fixtures = [
        {"case_num": 0, "input_shape": (2, 8, 64), "weight_shape": (64,), "layer": "input_layernorm"},
        {"case_num": 1, "input_shape": (1, 1, 64), "weight_shape": (64,), "layer": "input_layernorm"},
        {"case_num": 2, "input_shape": (2, 4, 64), "weight_shape": (64,), "layer": "input_layernorm"},
        {"case_num": 3, "input_shape": (3, 5, 64), "weight_shape": (64,), "layer": "post_attention_layernorm"},
    ]

    for fixture in fixtures:
        case_num = fixture["case_num"]
        input_shape = fixture["input_shape"]
        weight_shape = fixture["weight_shape"]
        layer = fixture["layer"]

        output_path = os.path.join(OUTPUT_DIR, f"norm-{case_num:02d}.safetensor")
        print(f"Generating fixture {case_num}: {output_path}")

        # Create input tensor with random values
        input_hidden_states = torch.randn(input_shape, dtype=torch.float32)

        # Create weight tensor (RMSNorm weight, typically all 1s or learned)
        weight = torch.ones(weight_shape, dtype=torch.float32)

        # Create expected output (just copy input for simplicity - we're testing aliasing, not correctness)
        expected_output = input_hidden_states.clone()

        # Build metadata
        metadata = {
            "layer": layer,
            "case_num": str(case_num),
            "input_shape": str(input_shape),
            "weight_shape": str(weight_shape),
        }

        # Save to safetensors
        tensors = {
            "input_hidden_states": input_hidden_states,
            "weight": weight,
            "output": expected_output,
        }

        with open(output_path, "wb") as f:
            f.write(st.save(tensors, metadata=metadata))

        print(f"  Input shape: {input_hidden_states.shape}")
        print(f"  Weight shape: {weight.shape}")
        print(f"  Layer: {layer}")
        print()

    print(f"=== Generated 1 model file + {len(fixtures)} fixture files ===")

if __name__ == "__main__":
    main()