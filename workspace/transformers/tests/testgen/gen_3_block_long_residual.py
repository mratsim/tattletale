#!/usr/bin/env python3
"""
Proof and fixtures for the long residual stream invariant.

Generates fixtures for the 3-block long residual stream test.
Computes:
1. HF local residual outputs (x_local)
2. Long residual stream outputs (mlp_out, r2)

Verifies the invariant: mlp_out + r2 == x_local at each layer boundary.
The invariant holds with EXACT equality (diff=0.0) when both paths use:
- BF16 addition for residuals (not FP32)
- FP32 RMSNorm internally (HF's Qwen3RMSNorm, not F.rms_norm)

Usage:
    cd tattletale
    .venv/bin/python workspace/transformers/tests/testgen/gen_3_block_long_residual.py
"""
import torch
from safetensors import torch as st
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen3-0.6B"
MODEL_PATH = str(Path(__file__).parent.parent / "hf_models" / MODEL_NAME)
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "long-residual-3-block"
NUM_LAYERS = 3
INPUT_TEXT = "Hello, how are you?"


def main():
    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()
    model.to("cpu")
    model = model.to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    input_ids = tokenizer(INPUT_TEXT, return_tensors="pt")["input_ids"]
    print(f"Input tokens: {input_ids[0].tolist()}")

    # Get embedding
    x_embed = model.model.embed_tokens(input_ids)

    # Store fixtures
    fixtures = []

    # HF local residual stream state
    x_hf = x_embed.clone()

    # Long residual stream state
    x_long = x_embed.clone()
    r_long = None

    for layer_idx in range(NUM_LAYERS):
        layer = model.model.layers[layer_idx]
        pos_ids = torch.arange(x_hf.size(1)).unsqueeze(0)
        cos, sin = model.model.rotary_emb(x_hf, pos_ids)

        # ── HF LOCAL RESIDUAL (actual model forward) ─────────────────────
        # Sublayer 1: attention
        res = x_hf
        h_hf = layer.input_layernorm(x_hf)
        attn_out_hf, _ = layer.self_attn(
            hidden_states=h_hf,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )
        x_hf = res + attn_out_hf

        # Sublayer 2: mlp
        res2 = x_hf
        h2_hf = layer.post_attention_layernorm(x_hf)
        mlp_out_hf = layer.mlp(h2_hf)
        x_hf = res2 + mlp_out_hf

        # ── LONG RESIDUAL STREAM (matches Nim implementation) ────────────
        # Step 1: attn_norm.forward_with_residual(x, residual)
        # CRITICAL: use BF16 addition (not FP32), then HF's FP32 RMSNorm module
        if r_long is None:
            # First layer: attn_norm(x), residual = x
            h_l = layer.input_layernorm(x_long)
            r_l = x_long.clone()
        else:
            # Subsequent layers: fused norm(x + r), BF16 addition
            combined = x_long + r_long  # BF16 addition
            r_l = combined.clone()
            h_l = layer.input_layernorm(combined)  # HF's FP32 RMSNorm

        # Step 2: Attention
        attn_l, _ = layer.self_attn(
            hidden_states=h_l,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )

        # Step 3: mlp_norm.forward_with_residual(attn_out, residual)
        # CRITICAL: BF16 addition, then HF's FP32 RMSNorm
        combined2 = attn_l + r_l  # BF16 addition
        r2_l = combined2.clone()
        h2_l = layer.post_attention_layernorm(combined2)  # HF's FP32 RMSNorm

        # Step 4: MLP
        mlp_l = layer.mlp(h2_l)

        # ── Assert invariant ─────────────────────────────────────────────
        invariant_check = (mlp_l + r2_l).float() - x_hf.float()
        max_inv_diff = invariant_check.abs().max().item()
        # The invariant should hold EXACTLY (diff=0.0) because both paths
        # use identical BF16 addition and FP32 RMSNorm.
        assert max_inv_diff == 0.0, (
            f"Layer {layer_idx:02d}: invariant check FAILED! "
            f"mlp_out + r2 vs HF out: max_diff={max_inv_diff:.6e}"
        )
        print(f"Layer {layer_idx:02d}: invariant check (mlp+r vs HF out): max_diff={max_inv_diff:.2e}")

        # ── Save ALL intermediate values ──────────────────────────────────
        fixture = {
            # Input
            "layer_input": x_long.clone(),

            # After attn_norm.forward_with_residual(x, residual)
            "after_attn_norm": h_l.clone(),
            "after_attn_norm_residual": r_l.clone(),

            # After attention
            "after_attn": attn_l.clone(),

            # After mlp_norm.forward_with_residual(attn_out, residual)
            "after_mlp_norm": h2_l.clone(),
            "after_mlp_norm_residual": r2_l.clone(),

            # After MLP
            "mlp_out": mlp_l.clone(),

            # HF local reference
            "hf_layer_output": x_hf.clone(),

            # RoPE (for Nim test)
            "position_ids": pos_ids.clone(),
            "cos": cos.clone(),
            "sin": sin.clone(),
        }
        fixtures.append(fixture)

        # Update long stream for next layer
        x_long = mlp_l
        r_long = r2_l

    # Save fixtures
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for layer_idx, fixture in enumerate(fixtures):
        filename = f"block-{layer_idx:02d}.safetensor"
        filepath = FIXTURE_DIR / filename

        sorted_tensors = {
            name: tensor.detach().cpu().to(torch.bfloat16).contiguous()
            for name, tensor in sorted(fixture.items())
            if tensor is not None
        }

        serialized = st.save(sorted_tensors, metadata=None)
        with open(filepath, "wb") as f:
            f.write(serialized)

        print(f"  Saved: {filepath}")

    print("\n✓ All invariants verified (EXACT equality, diff=0.0)")
    print("  The invariant is tested in test_qwen3_long_residual_3_blocks.nim")


if __name__ == "__main__":
    main()
