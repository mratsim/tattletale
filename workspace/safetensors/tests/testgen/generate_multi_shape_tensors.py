#!/usr/bin/env python3
"""
Generate a minimal safetensors fixture to reproduce the shape aliasing bug.

This creates a file with multiple tensors of different shapes to test
that loading multiple tensors doesn't corrupt their shape metadata.
"""

import torch
from safetensors.torch import save_file
import os

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")

def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    
    # Create tensors with distinctly different shapes
    # Minimal sizes for aliasing test - only shape diversity matters
    tensors = {
        "tensor_a": torch.randn([4, 2], dtype=torch.bfloat16),
        "tensor_b": torch.randn([2, 2], dtype=torch.bfloat16),
        "tensor_c": torch.randn([2, 4], dtype=torch.bfloat16),
        "tensor_d": torch.randn([6, 2], dtype=torch.bfloat16),
        "tensor_e": torch.randn([2, 6], dtype=torch.bfloat16),
    }
    
    output_path = os.path.join(FIXTURES_DIR, "shape_aliasing_multi_tensor.safetensors")
    save_file(tensors, output_path)
    
    print(f"Generated fixture: {output_path}")
    print("Tensor shapes:")
    for name, tensor in tensors.items():
        print(f"  {name}: {list(tensor.shape)}")

if __name__ == "__main__":
    main()