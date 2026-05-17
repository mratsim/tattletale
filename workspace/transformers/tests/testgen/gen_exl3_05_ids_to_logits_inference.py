#!/usr/bin/env python3
"""
Generate EXL3 layer-by-layer inference fixtures for Nim testing.

Captures layer_input and layer_output for each of the 28 layers, plus final_logits.

All core logic (decoder, forward, RoPE, RMS norm) lives in ``q_exl3_common.py``.
This file only handles model paths, tokenizer, data I/O, and the top-level
forward orchestration.

Usage:
    cd tattletale
    .venv/bin/python testgen/gen_exl3_05_ids_to_logits_inference.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save_file

# ── Add testgen dir to path for importing q_exl3_common ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from q_exl3_common import (
    # utilities
    get_exl3_tensors,
    get_in_features_out_features,
    derive_K,
    load_config,
    # orig (CUDA ground truth)
    reconstruct_orig_exl3,
    linear_forward_orig_exl3,
    # orig / reimpl (EXL3-anchored)
    rms_norm_orig_exl3,
    precompute_freqs_cis_reimpl_exl3,
    apply_rotary_pos_emb_reimpl_exl3,
)

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # tests/
MODEL_DIR = os.path.join(BASE_DIR, "hf_models", "Qwen3-0.6B-EXL3-5bpw")
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
OUTPUT_DIR = Path(BASE_DIR) / "fixtures" / "exl3-ids-inference" / "Qwen3-0.6B-EXL3-5bpw"
INPUT_TEXT = "Hello, how are you?"
LAYER_COUNT = 28
DTYPE = torch.float16
DEVICE = "cuda:0"

# ── Determinism ────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


# ── Helpers ──────────────────────────────────────────────────────────────


def reconstruct_and_cache(tensors: dict, config: dict) -> dict:
    """Reconstruct all EXL3 linear weights and cache them.

    Returns:
        { "model.layers.0.self_attn.q_proj": weight [out_features, in_features], ... }
    """
    cache: dict = {}
    for key, entry in tensors.items():
        if key.startswith("_") or entry.get("trellis") is None:
            continue
        trellis = entry["trellis"].to("cuda:0")
        K = derive_K(trellis)
        mcg = entry.get("mcg") is not None
        mul1 = entry.get("mul1") is not None
        in_f, out_f = get_in_features_out_features(key, trellis, config)
        w = reconstruct_orig_exl3(trellis, K, mcg, mul1, (in_f, out_f))
        # Transpose to [out_features, in_features] for F.linear
        cache[key] = w.contiguous()  # [in_features, out_features] — non-transposed, for ext.hgemm (differs from F.linear which needs [out_features, in_features])
        print(f"  Reconstructed {key}: [{in_f}, {out_f}] -> weight [{out_f}, {in_f}]")
    return cache


# ── Tokenizer helpers (minimal, no HF dependency) ──────────────────────


def load_tokenizer(model_dir: str) -> dict:
    """Load tokenizer config for vocab size and special tokens."""
    import json
    with open(os.path.join(model_dir, "tokenizer_config.json")) as f:
        tok_config = json.load(f)
    with open(os.path.join(model_dir, "vocab.json")) as f:
        vocab = json.load(f)
    return {
        "vocab": vocab,
        "bos_token_id": tok_config.get("bos_token_id", 151643),
        "eos_token_id": tok_config.get("eos_token_id", 151645),
    }


def encode_text(text: str, vocab: dict) -> list:
    """Simple byte-level encode for our fixed test input."""
    known = {"Hello, how are you?": [10161, 11, 1355, 527, 499, 30]}
    if text in known:
        return known[text]
    tokens = []
    for ch in text:
        if ch in vocab:
            tokens.append(vocab[ch])
        else:
            tokens.append(vocab.get(" ", 220))
    return tokens if tokens else [151643]


# ── Main forward pass ─────────────────────────────────────────────────


def run_exl3_forward(tensors: dict, weights: dict, config: dict,
                     input_ids: torch.Tensor) -> tuple:
    """Run full EXL3 model forward pass, capturing per-layer intermediates.

    Args:
        tensors: Raw EXL3 tensors dict (for norms, embeddings, lm_head entries).
        weights: Reconstructed linear weights dict.
        config: Model config dict.
        input_ids: [1, seq_len] token ids.

    Returns:
        (per_layer_intermediates, final_logits)
    """
    device = torch.device(DEVICE)
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    intermediate_size = config["intermediate_size"]
    rms_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 1000000.0)
    max_seq_len = config.get("max_position_embeddings", 40960)

    batch, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Precompute RoPE cos/sin
    cos, sin = precompute_freqs_cis_reimpl_exl3(head_dim, max_seq_len, theta=rope_theta)
    cos = cos.to(DTYPE).to(device)
    sin = sin.to(DTYPE).to(device)

    # Embedding
    embed_weight = tensors["_embeddings"]["model.embed_tokens.weight"].to(device)
    h = torch.nn.functional.embedding(input_ids.to(device), embed_weight)

    # Gather norm weights
    norms: dict = {}
    for k, v in tensors["_norms"].items():
        norms[k] = v.to(device)

    # Gather lm_head entry
    lm_head_key = "lm_head"

    intermediates_list: list = [None] * LAYER_COUNT

    for layer_idx in range(LAYER_COUNT):
        prefix = f"model.layers.{layer_idx}"
        layer_input = h.clone()

        # ── RMS Norm (input_layernorm) ──
        norm_key = f"{prefix}.input_layernorm.weight"
        ln_weight = norms.get(norm_key)
        if ln_weight is None:
            ln_weight = norms.get(f"{prefix}.input_layernorm.weight")
        h_norm = rms_norm_orig_exl3(h, ln_weight, rms_eps)

        # ── Self-Attention ──
        q_weight = weights[f"{prefix}.self_attn.q_proj"]
        k_weight = weights[f"{prefix}.self_attn.k_proj"]
        v_weight = weights[f"{prefix}.self_attn.v_proj"]
        o_weight = weights[f"{prefix}.self_attn.o_proj"]

        q_suh = tensors[f"{prefix}.self_attn.q_proj"]["suh"].to(device)
        q_svh = tensors[f"{prefix}.self_attn.q_proj"]["svh"].to(device)
        k_suh = tensors[f"{prefix}.self_attn.k_proj"]["suh"].to(device)
        k_svh = tensors[f"{prefix}.self_attn.k_proj"]["svh"].to(device)
        v_suh = tensors[f"{prefix}.self_attn.v_proj"]["suh"].to(device)
        v_svh = tensors[f"{prefix}.self_attn.v_proj"]["svh"].to(device)
        o_suh = tensors[f"{prefix}.self_attn.o_proj"]["suh"].to(device)
        o_svh = tensors[f"{prefix}.self_attn.o_proj"]["svh"].to(device)

        # Q/K/V
        q = linear_forward_orig_exl3(h_norm, q_weight, q_suh, q_svh)
        k = linear_forward_orig_exl3(h_norm, k_weight, k_suh, k_svh)
        v = linear_forward_orig_exl3(h_norm, v_weight, v_suh, v_svh)

        # Reshape to multi-head format
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # QK norm
        q_norm_w = norms.get(f"{prefix}.self_attn.q_norm.weight")
        k_norm_w = norms.get(f"{prefix}.self_attn.k_norm.weight")
        if q_norm_w is not None:
            q = rms_norm_orig_exl3(q, q_norm_w.to(device), rms_eps)
        if k_norm_w is not None:
            k = rms_norm_orig_exl3(k, k_norm_w.to(device), rms_eps)

        # RoPE
        q, k = apply_rotary_pos_emb_reimpl_exl3(q, k, cos, sin, position_ids)
        q = q.to(torch.float16)
        k = k.to(torch.float16)

        # GQA
        if num_kv_heads < num_heads:
            n_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        # SDPA
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            scale=head_dim ** -0.5)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, num_heads * head_dim)

        # O projection
        attn_output = linear_forward_orig_exl3(attn_output, o_weight, o_suh, o_svh)

        # Residual
        h = layer_input + attn_output
        after_attn_residual = h.clone()

        # ── MLP ──
        residual = h
        post_ln_weight = norms.get(f"{prefix}.post_attention_layernorm.weight")
        h_norm = rms_norm_orig_exl3(h, post_ln_weight, rms_eps)

        gate_weight = weights[f"{prefix}.mlp.gate_proj"]
        up_weight = weights[f"{prefix}.mlp.up_proj"]
        down_weight = weights[f"{prefix}.mlp.down_proj"]

        gate_suh = tensors[f"{prefix}.mlp.gate_proj"]["suh"].to(device)
        gate_svh = tensors[f"{prefix}.mlp.gate_proj"]["svh"].to(device)
        up_suh = tensors[f"{prefix}.mlp.up_proj"]["suh"].to(device)
        up_svh = tensors[f"{prefix}.mlp.up_proj"]["svh"].to(device)
        down_suh = tensors[f"{prefix}.mlp.down_proj"]["suh"].to(device)
        down_svh = tensors[f"{prefix}.mlp.down_proj"]["svh"].to(device)

        gate = linear_forward_orig_exl3(h_norm, gate_weight, gate_suh, gate_svh)
        up = linear_forward_orig_exl3(h_norm, up_weight, up_suh, up_svh)

        gate = torch.nn.functional.silu(gate)
        mlp_output = gate * up
        mlp_output = linear_forward_orig_exl3(mlp_output, down_weight, down_suh, down_svh)

        h = residual + mlp_output
        layer_output = h.clone()

        intermediates_list[layer_idx] = {
            "layer_input": layer_input.cpu().to(DTYPE).contiguous(),
            "layer_output": layer_output.cpu().to(DTYPE).contiguous(),
        }

    # ── Final norm ──
    final_norm_weight = norms.get("model.norm.weight")
    h = rms_norm_orig_exl3(h, final_norm_weight, rms_eps)

    # ── LM Head (EXL3-quantized) ──
    if lm_head_key in weights:
        lm_w = weights[lm_head_key]
        lm_suh = tensors[lm_head_key]["suh"].to(device)
        lm_svh = tensors[lm_head_key]["svh"].to(device)
        logits = linear_forward_orig_exl3(h, lm_w, lm_suh, lm_svh)
    else:
        # Fallback: use embed_tokens.weight (tie_word_embeddings)
        logits = torch.nn.functional.linear(h, embed_weight)

    return intermediates_list, logits


# ── Saving ──────────────────────────────────────────────────────────────


def save_fixture(output_dir: Path, layer_idx: int, metadata: dict, tensors_dict: dict) -> Path:
    """Save intermediate tensors as safetensor + metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"layer-{layer_idx:02d}.safetensor"
    filepath = output_dir / filename

    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().to(DTYPE).contiguous())
        for name, tensor in sorted(tensors_dict.items())
        if tensor is not None
    )

    st_save_file(sorted_tensors, str(filepath))

    metadata_path = filepath.with_suffix(".metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")

    return filepath


# ── Main ────────────────────────────────────────────────────────────────


def main():
    print(f"Model: {MODEL_DIR}")
    print(f"Input: {INPUT_TEXT}")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    config = load_config()
    print(f"Loading EXL3 tensors from {MODEL_PATH}...")
    tensors = get_exl3_tensors(MODEL_PATH)

    # Check if lm_head is in the main result dict
    lm_head_key = "lm_head"
    if lm_head_key in tensors:
        print(f"  Found lm_head in main EXL3 layers (has trellis: {tensors[lm_head_key].get('trellis') is not None})")
    else:
        print(f"  Warning: lm_head not found as EXL3 layer, will use embed_tokens.weight as fallback")

    print(f"\nReconstructing EXL3 weights...")
    weights = reconstruct_and_cache(tensors, config)

    print(f"\nRunning forward pass for {LAYER_COUNT} layers...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        input_ids = tokenizer(INPUT_TEXT, return_tensors="pt").input_ids
    except (ImportError, ModuleNotFoundError, OSError) as e:
        print(f"  [info] transformers not available ({e}), using known token IDs")
        input_ids = torch.tensor([[10161, 11, 1355, 527, 499, 30]])
    print(f"  Input IDs: {input_ids[0].tolist()}")
    print(f"  Sequence length: {input_ids.shape[1]}")

    intermediates_list, logits = run_exl3_forward(tensors, weights, config, input_ids)

    # Save per-layer fixtures
    print(f"\nSaving layer fixtures...")
    for i, intermediates in enumerate(intermediates_list):
        if intermediates is None:
            print(f"  Layer {i:02d}: SKIPPED")
            continue
        filepath = save_fixture(
            OUTPUT_DIR, i,
            metadata={
                "framework": "exl3",
                "model": "Qwen3-0.6B-EXL3-5bpw",
                "layer": i,
                "input_text": INPUT_TEXT,
                "input_tokens": input_ids[0].tolist(),
                "batch_size": 1,
                "seq_len": input_ids.shape[1],
                "dtype": "float16",
                "device": "cpu",
            },
            tensors_dict=intermediates,
        )
        print(f"  Layer {i:02d}: {filepath}")

    # Save final logits
    print(f"\nSaving final logits...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logits_file = OUTPUT_DIR / "final_logits.safetensor"
    st_save_file(
        {"logits": logits.detach().cpu().to(DTYPE).contiguous()},
        str(logits_file),
        metadata={"model": "Qwen3-0.6B-EXL3-5bpw", "input_text": INPUT_TEXT, "dtype": "float16"},
    )
    print(f"  Logits: {logits_file}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
