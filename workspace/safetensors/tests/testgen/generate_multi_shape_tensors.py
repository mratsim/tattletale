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
    tensors = {
        "tensor_a": torch.randn([2048, 1024], dtype=torch.bfloat16),
        "tensor_b": torch.randn([1024, 1024], dtype=torch.bfloat16),
        "tensor_c": torch.randn([1024, 2048], dtype=torch.bfloat16),
        "tensor_d": torch.randn([3072, 1024], dtype=torch.bfloat16),
        "tensor_e": torch.randn([1024, 3072], dtype=torch.bfloat16),
    }
    
    output_path = os.path.join(FIXTURES_DIR, "shape_aliasing_multi_tensor.safetensors")
    save_file(tensors, output_path)
    
    print(f"Generated fixture: {output_path}")
    print("Tensor shapes:")
    for name, tensor in tensors.items():
        print(f"  {name}: {list(tensor.shape)}")

if __name__ == "__main__":
    main()