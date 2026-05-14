#!/usr/bin/env python3
"""
Fuzzing test: Compare Nim transformers against HuggingFace transformers.
Tests that our model produces logits matching HF within BF16 tolerance.
"""
import sys
import os
import random
import torch
from pathlib import Path
import ctypes

# Add the tests directory to path for pytttransformers import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure libtorch shared libs are discoverable at runtime
# (pytttransformers.so links against libc10, libtorch, etc.)
torch_lib_dir = str(Path(torch.__file__).parent / "lib")
os.environ["LD_LIBRARY_PATH"] = torch_lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

from transformers import Qwen3ForCausalLM, AutoTokenizer

MODEL_PATH = str(Path(__file__).parent / "hf_models" / "Qwen3-0.6B")

def main():
    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH)
    hf_model.eval()
    hf_model.to("cpu")

    # Load Nim model
    print("Loading Nim model...")
    import pytttransformers
    nim_model = pytttransformers.init_model(MODEL_PATH)

    # Run fuzzing tests
    print(f"Running 20 fuzzing cases...")
    random.seed(42)
    torch.manual_seed(42)

    passed = 0
    for i in range(20):
        seq_len = random.randint(1, 20)
        input_ids = torch.randint(100, 1000, (1, seq_len))

        # HF forward pass
        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        # Nim forward pass
        nim_logits = nim_model.forward(input_ids)

        # Compare
        hf_f32 = hf_logits.float()
        nim_f32 = nim_logits.float()

        max_diff = (hf_f32 - nim_f32).abs().max().item()
        allclose = torch.allclose(hf_f32, nim_f32, rtol=5e-2, atol=5e-2)

        if allclose:
            print(f"  ✅ Case {i} passed (seq_len={seq_len}, max_diff={max_diff:.6f})")
            passed += 1
        else:
            print(f"  ❌ Case {i} FAILED (seq_len={seq_len}, max_diff={max_diff:.6f})")
            # Show where the biggest difference is
            diff = (hf_f32 - nim_f32).abs()
            max_idx = diff.argmax()
            print(f"     HF logits[..., {max_idx.item()}] = {hf_f32.flatten()[max_idx].item():.4f}")
            print(f"     Nim logits[..., {max_idx.item()}] = {nim_f32.flatten()[max_idx].item():.4f}")

    print(f"\nResults: {passed}/20 passed")
    if passed == 20:
        print("All tests passed!")
        return 0
    else:
        print(f"{20 - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
