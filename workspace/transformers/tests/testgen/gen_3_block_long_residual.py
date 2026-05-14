#!/usr/bin/env python3
"""
Proof and fixtures for the long residual stream invariant.

This script generates fixtures for the 3-block long residual stream test.
It computes:
1. HF local residual outputs (x_local)
2. Long residual stream outputs (mlp_out, r2)

Then verifies the invariant: mlp_out + r2 == x_local at each layer boundary.

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
NUM_LAYERS = 3  # Only first 3 layers
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
        h = layer.input_layernorm(x_hf)
        attn_out, _ = layer.self_attn(
            hidden_states=h,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )
        x_hf = res + attn_out

        # Sublayer 2: mlp
        res2 = x_hf
        h2 = layer.post_attention_layernorm(x_hf)
        mlp_out_hf = layer.mlp(h2)
        x_hf = res2 + mlp_out_hf

        # ── LONG RESIDUAL STREAM (matches Nim implementation) ────────────
        if r_long is None:
            # First layer: attn_norm(x), residual = x
            h_l, r_l = layer.input_layernorm(x_long), x_long.clone()
        else:
            # Subsequent layers: fused norm(x + r)
            r_l = x_long + r_long
            h_l = torch.nn.functional.rms_norm(
                r_l, layer.input_layernorm.weight.shape,
                weight=layer.input_layernorm.weight, eps=1e-6
            )

        # Attention
        attn_l, _ = layer.self_attn(
            hidden_states=h_l,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )

        # mlp_norm.forward_with_residual(attn_out, r)
        # Invariant: mlp_out + r2 == x_local (HF local output)
        r2_l = attn_l + r_l
        h2_l = torch.nn.functional.rms_norm(
            r2_l, layer.post_attention_layernorm.weight.shape,
            weight=layer.post_attention_layernorm.weight, eps=1e-6
        )

        mlp_l = layer.mlp(h2_l)

        # Verify invariant
        invariant_check = (mlp_l + r2_l).float() - x_hf.float()
        max_inv_diff = invariant_check.abs().max().item()
        print(f"Layer {layer_idx:02d}: invariant check (mlp+r vs HF out): max_diff={max_inv_diff:.2e}")

        # Save fixtures
        fixture = {
            "layer_input": x_long.clone(),
            "residual": r_l.clone() if r_long is not None else x_long.clone(),
            "hf_layer_output": x_hf.clone(),
            "long_mlp_out": mlp_l.clone(),
            "long_mlp_residual": r2_l.clone(),
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

    # Final check
    print("\nInvariant summary:")
    print("  At each layer boundary: mlp_out + r2 == x_local")
    print("  This is the key invariant that allows the long residual stream")
    print("  to be mathematically equivalent to the HF local residual pattern.")
    print("\n  The invariant is proven in WIP/Spec_Residual_Patterns.md")
    print("  and tested in test_qwen3_long_residual_3_blocks.nim")


if __name__ == "__main__":
    main()
