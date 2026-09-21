#!/usr/bin/env python3
"""Tier-02 fixture generator, Qwen3-0.6B first-8-layers chain plus the final
pre-norm tail, CPU torch bf16, the long residual stream invariant proof.

The first 8 decoder blocks carry full per-block intermediates, the tail
checkpoint carries the decoder stack output pre-final-norm. The model has
28 uniform-attention layers, the 8-block prefix is class-complete.

- per block the fixtures record the HF local residual outputs (x_local) and the long residual stream outputs (mlp_out, r2)
- the invariant mlp_out + r2 == x_local holds with exact equality (diff 0.0) at all 28 layer boundaries

- both paths use bf16 residual addition and fp32 RMSNorm internally (HF Qwen3RMSNorm)
- one metadata.json sidecar per block fixture (model, layer, case, seq length), the tail carries the decoder stack depth

Regenerate:

- .venv/bin/python workspace/transformers/tests/testgen/gen_bf16_qwen3_02_first_8_layers_plus_final.py
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import torch as st
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import write_text_zst  # noqa, the path insert precedes the import

MODEL_NAME = "Qwen3-0.6B"
MODEL_PATH = str(Path(__file__).parent.parent / "hf_models" / MODEL_NAME)
FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "bf16-02-first-8-layers-plus-final" / MODEL_NAME
PREFIX_BLOCKS = 8
INPUT_TEXT = "Hello, how are you?"


def main():
    """Generates the bf16-02 first-8-layers-plus-final chain fixtures."""
    # load the checkpoint
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()
    model.to("cpu")
    num_layers = model.config.num_hidden_layers

    # Preserve inv_freq buffers in float32, model.to(bfloat16) would corrupt them.
    # bfloat16 loses too much precision for RoPE frequency values (up to 1.2e-3 per element).
    # This causes ~4e-3 cos/sin discrepancy that propagates through every layer.
    inv_freq = model.model.rotary_emb.inv_freq.float()
    original_inv_freq = model.model.rotary_emb.original_inv_freq.float()

    model = model.to(torch.bfloat16)

    # Restore inv_freq in float32 after dtype conversion
    model.model.rotary_emb.inv_freq = inv_freq
    model.model.rotary_emb.original_inv_freq = original_inv_freq

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    input_ids = tokenizer(INPUT_TEXT, return_tensors="pt")["input_ids"]
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Decoder layers: {num_layers}, recorded prefix blocks: {PREFIX_BLOCKS}")

    # get the embedding
    x_embed = model.model.embed_tokens(input_ids)

    # store the fixtures
    fixtures = []

    # HF local residual stream state
    x_hf = x_embed.clone()

    # Long residual stream state
    x_long = x_embed.clone()
    r_long = None

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        pos_ids = torch.arange(x_hf.size(1)).unsqueeze(0)
        cos, sin = model.model.rotary_emb(x_hf, pos_ids)

        # ── HF LOCAL RESIDUAL (actual model forward) ─────────────────────
        # sublayer 1 runs the attention
        res = x_hf
        h_hf = layer.input_layernorm(x_hf)
        attn_out_hf, _ = layer.self_attn(
            hidden_states=h_hf,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )
        x_hf = res + attn_out_hf

        # sublayer 2 runs the mlp
        res2 = x_hf
        h2_hf = layer.post_attention_layernorm(x_hf)
        mlp_out_hf = layer.mlp(h2_hf)
        x_hf = res2 + mlp_out_hf

        # ── LONG RESIDUAL STREAM (matches Nim implementation) ────────────
        # step 1 runs attn_norm.forward_with_residual(x, residual)
        # use BF16 addition (not FP32), then HF's FP32 RMSNorm module
        if r_long is None:
            # first layer runs attn_norm(x), the residual stays x
            h_l = layer.input_layernorm(x_long)
            r_l = x_long.clone()
        else:
            # later layers run the fused norm(x + r), the addition stays BF16
            combined = x_long + r_long  # the BF16 addition
            r_l = combined.clone()
            h_l = layer.input_layernorm(combined)  # HF's FP32 RMSNorm

        # step 2 runs the attention
        attn_l, _ = layer.self_attn(
            hidden_states=h_l,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=None,
        )

        # step 3 runs mlp_norm.forward_with_residual(attn_out, residual)
        # BF16 addition again, then HF's FP32 RMSNorm
        combined2 = attn_l + r_l  # the BF16 addition
        r2_l = combined2.clone()
        h2_l = layer.post_attention_layernorm(combined2)  # HF's FP32 RMSNorm

        # step 4 runs the MLP
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

        if layer_idx < PREFIX_BLOCKS:
            # ── Save ALL intermediate values ──────────────────────────────
            fixture = {
                # block input:
                "layer_input": x_long.clone(),

                # after attn_norm.forward_with_residual(x, residual):
                "after_attn_norm": h_l.clone(),
                "after_attn_norm_residual": r_l.clone(),

                # after attention:
                "after_attn": attn_l.clone(),

                # after mlp_norm.forward_with_residual(attn_out, residual):
                "after_mlp_norm": h2_l.clone(),
                "after_mlp_norm_residual": r2_l.clone(),

                # after MLP:
                "mlp_out": mlp_l.clone(),

                # the HF local reference (the chain checkpoint of this block):
                "hf_layer_output": x_hf.clone(),

                # RoPE (for Nim test)
                "position_ids": pos_ids.clone(),
                "cos": cos.clone(),
                "sin": sin.clone(),
            }
            fixtures.append(fixture)
        elif layer_idx == num_layers - 1:
            # the tail checkpoint carries the decoder stack output taken pre-final-norm,
            # the layer output feeding the final RMSNorm and lm_head
            #
            # - the long stream sum equals the local stream at every layer,
            #   so both formulations hand the same value to the final norm
            fixtures.append({
                "pre_final_norm": x_hf.clone(),
            })

        # Update long stream for next layer
        x_long = mlp_l
        r_long = r2_l

    assert len(fixtures) == PREFIX_BLOCKS + 1, (
        f"expected {PREFIX_BLOCKS + 1} fixtures (8 block checkpoints + tail), "
        f"got {len(fixtures)}")

    # save the fixtures
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for idx, fixture in enumerate(fixtures):
        is_tail = idx == len(fixtures) - 1
        filename = "tail.safetensor" if is_tail else f"block-{idx:02d}.safetensor"
        filepath = FIXTURE_DIR / filename

        sorted_tensors = {
            name: tensor.detach().cpu().to(torch.bfloat16).contiguous()
            for name, tensor in sorted(fixture.items())
            if tensor is not None
        }

        serialized = st.save(sorted_tensors, metadata=None)
        with open(filepath, "wb") as f:
            f.write(serialized)

        if is_tail:
            metadata = {
                "case": f"chain_tail_seq{input_ids.shape[1]}",
                "layer": "model.norm (input side)",
                "model": MODEL_NAME,
                "note": "pre_final_norm is the decoder stack output taken "
                        "pre-final-norm; it feeds the final RMSNorm and lm_head",
                "num_hidden_layers": num_layers,
                "seq_len": int(input_ids.shape[1]),
            }
        else:
            metadata = {
                "case": f"chain_block_{idx}_seq{input_ids.shape[1]}",
                "layer": f"model.layers.{idx}",
                "model": MODEL_NAME,
                "note": "hf_layer_output is the local-residual checkpoint, "
                        "mlp_out/r2 the long residual stream",
                "seq_len": int(input_ids.shape[1]),
            }
        metadata_path = FIXTURE_DIR / f"{filename}.metadata.json.zst"
        write_text_zst(metadata_path,
                       json.dumps(metadata, indent=2, sort_keys=True)
                       .encode("utf-8") + b"\n")

        print(f"  Saved: {filepath}")

    print("\n✓ All invariants verified (EXACT equality, diff=0.0) at all "
          f"{num_layers} layer boundaries")
    print("  The invariant and the 9 checkpoints are tested in t_bf16_qwen3_02_first_8_layers_plus_final.nim")


if __name__ == "__main__":
    main()
