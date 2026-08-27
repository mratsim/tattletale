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
    hf_model = hf_model.to("cpu")

    # Preserve inv_freq in float32 — bfloat16 loses too much precision
    # for RoPE frequency values (up to 1.2e-3 per element).
    inv_freq = hf_model.model.rotary_emb.inv_freq.float()
    original_inv_freq = hf_model.model.rotary_emb.original_inv_freq.float()

    # Convert model weights to bfloat16 (matches Nim model dtype).
    # Without this, HF runs in float32 and Nim runs in bfloat16,
    # producing apparent diffs of 0.2-0.8 even with identical weights.
    hf_model = hf_model.to(torch.bfloat16)

    # Restore inv_freq in float32 after dtype conversion
    hf_model.model.rotary_emb.inv_freq = inv_freq
    hf_model.model.rotary_emb.original_inv_freq = original_inv_freq

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

        # Compare (both cast to float32 for comparison)
        hf_f32 = hf_logits.float()
        nim_f32 = nim_logits.float()

        max_diff = (hf_f32 - nim_f32).abs().max().item()
        allclose = torch.allclose(hf_f32, nim_f32, rtol=1e-3, atol=1e-3)

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


def qwen35_main():
    """Qwen3.5-0.8B fuzz: Nim sequential GDN vs vendored HF patched to the
    sequential (recurrent) rule. The HF chunked prefill diverges from the
    sequential reference through 24 bf16 layers (~1e-2..1e-1 logits), so the
    HF side is patched to the recurrent rule the Nim implementation mirrors
    (the sequential reference). The comparison then runs at the
    rtol/atol 1e-3 bar.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
            torch_recurrent_gated_delta_rule,
        )
    except ModuleNotFoundError:
        import transformers
        print(
            "[test_vs_hf_transformers] transformers >= 5.2.0 required for qwen3_5; "
            f"got {transformers.__version__}; run uv pip install -U transformers"
        )
        sys.exit(1)

    MODEL_PATH = str(Path(__file__).parent / "hf_models" / "Qwen3.5-0.8B")

    print("Loading Qwen3.5 HuggingFace model (bf16, CPU)...")
    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16)
    hf_model.eval()
    hf_model = hf_model.to("cpu")

    # Patch every GDN layer to the recurrent (sequential) rule so the HF
    # reference computes the same recurrence as the Nim implementation.
    for layer in hf_model.model.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = torch_recurrent_gated_delta_rule

    print("Loading Nim model...")
    import pytttransformers
    nim_model = pytttransformers.init_model(MODEL_PATH)

    print("Running 20 Qwen3.5 fuzzing cases...")
    random.seed(42)
    torch.manual_seed(42)

    passed = 0
    for i in range(20):
        seq_len = random.randint(1, 20)
        input_ids = torch.randint(100, 1000, (1, seq_len))

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        nim_logits = nim_model.forward(input_ids)

        hf_f32 = hf_logits.float()
        nim_f32 = nim_logits.float()

        max_diff = (hf_f32 - nim_f32).abs().max().item()
        allclose = torch.allclose(hf_f32, nim_f32, rtol=1e-3, atol=1e-3)

        if allclose:
            print(f"  ✅ Qwen3.5 case {i} passed (seq_len={seq_len}, max_diff={max_diff:.6f})")
            passed += 1
        else:
            print(f"  ❌ Qwen3.5 case {i} FAILED (seq_len={seq_len}, max_diff={max_diff:.6f})")
            diff = (hf_f32 - nim_f32).abs()
            max_idx = diff.argmax()
            print(f"     HF logits[..., {max_idx.item()}] = {hf_f32.flatten()[max_idx].item():.4f}")
            print(f"     Nim logits[..., {max_idx.item()}] = {nim_f32.flatten()[max_idx].item():.4f}")

    print(f"\nQwen3.5 Results: {passed}/20 passed")
    return 0 if passed == 20 else 1


if __name__ == "__main__":
    rc = main()
    if rc != 0:
        sys.exit(rc)
    sys.exit(qwen35_main())
