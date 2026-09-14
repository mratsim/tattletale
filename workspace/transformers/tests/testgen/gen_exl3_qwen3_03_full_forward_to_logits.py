#!/usr/bin/env python3
"""
EXL3 chain generator, aligned to the bf16-03 chain template.

- one 004 stats frame per layer boundary, the two boundary entries
  (layer_input, layer_output) on the fp16 storage grid
- one 005 decisions frame, one record per position over the top-32
  logits support, written by the canonical chains recording code
  imported from fixture_stats and gen_rerecord_chains_005
- statistics-only output, exactly the surface the suite consumes
- recordings come from the exl3_utils loading machinery, never out of the code under test
- backend exllamav3_cuda runs the production kernels, the CUDA extension required
- backend pytorch is the pure-torch fallback, the 00 codec family records the two decoders agreeing element for element
- the pre-write guard mirrors gen_rerecord_chains_005, the recorded argmax id
  reproduces exactly, the margin stays inside the band, a wider
  divergence aborts the write
- run from the tests/ dir, uv run python testgen/gen_exl3_qwen3_03_full_forward_to_logits.py
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import torch

# ── Script dir plus the tests/ tree on the import path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

# Both imports sit directly below the sys.path setup
from fixture_stats import argmax_record_from_row, grid_of, read_text_zst, stats_file_bytes, write_argmax_decisions, write_text_zst  # noqa
from gen_rerecord_chains_005 import NUM_POSITIONS  # noqa
from quant_utils.exl3_utils import get_exl3_tensors, get_in_features_out_features, derive_K, derive_cb, load_config, reconstruct_orig_exl3, reconstruct_reimpl_exl3, linear_forward_orig_exl3, linear_forward_reimpl_exl3, rms_norm_orig_exl3, precompute_freqs_cis_reimpl_exl3, apply_rotary_pos_emb_reimpl_exl3  # noqa

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # the tests/ directory
MODEL_NAME = "Qwen3-0.6B-EXL3-5bpw"
MODEL_DIR = os.path.join(BASE_DIR, "hf_models", MODEL_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
OUTPUT_DIR = Path(BASE_DIR) / "fixtures" / "exl3-03-full-forward-to-logits" / MODEL_NAME
INPUT_TEXT = "Hello, how are you?"
DTYPE = torch.float16

# Tokenizer ids of INPUT_TEXT, the ids the suite replays verbatim.
INPUT_IDS = [9707, 11, 1246, 525, 498, 30]

# ── Backend ─────────────────────────────────────────────────────────────
# Production ext kernels run CUDA-only, the pure-torch reimplementation
# stands in on the cpu reference box, the 01-layer generator fallback
try:
    from exllamav3.ext import exllamav3_ext as _ext  # noqa: F401
    USE_CUDA = torch.cuda.is_available()
except (ImportError, ModuleNotFoundError, OSError):
    USE_CUDA = False
BACKEND = "exllamav3_cuda" if USE_CUDA else "pytorch"
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

# ── Determinism ────────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ── EXL3 linear machinery ────────────────────────────────────────────────


def reconstruct_and_cache(tensors: dict, config: dict) -> dict:
    """Reconstruct every EXL3 linear weight through the loading machinery.

    - backend exllamav3_cuda, [in_features, out_features], the ext.hgemm layout
    - backend pytorch, [out_features, in_features], the F.linear layout
    - returns the reconstructed weights dict keyed like model.layers.0.self_attn.q_proj
    """
    cache: dict = {}
    for key, entry in tensors.items():
        if key.startswith("_") or entry.get("trellis") is None:
            continue
        trellis = entry["trellis"].to(DEVICE)
        K = derive_K(trellis)
        in_f, out_f = get_in_features_out_features(key, trellis, config)
        if BACKEND == "exllamav3_cuda":
            mcg = entry.get("mcg") is not None
            mul1 = entry.get("mul1") is not None
            w = reconstruct_orig_exl3(trellis, K, mcg, mul1, (in_f, out_f))
            cache[key] = w.contiguous()
        else:
            cb = derive_cb(entry)
            w = reconstruct_reimpl_exl3(trellis, K, cb, (in_f, out_f))
            cache[key] = w.t().contiguous()
        print(f"  Reconstructed {key}: [{in_f}, {out_f}] ({BACKEND})")
    return cache


def linear_forward(x: torch.Tensor, weight: torch.Tensor,
                   suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """One EXL3 linear pass on the selected backend, Hadamard in then GEMM then Hadamard out, the layouts from reconstruct_and_cache."""
    if BACKEND == "exllamav3_cuda":
        return linear_forward_orig_exl3(x, weight, suh, svh)
    return linear_forward_reimpl_exl3(x, weight, suh, svh)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm on the selected backend, fp32 accumulation, fp16 output."""
    if BACKEND == "exllamav3_cuda":
        return rms_norm_orig_exl3(x, weight, eps)
    orig_shape = x.shape
    x32 = x.reshape(-1, orig_shape[-1]).to(torch.float32)
    w32 = weight.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    y = x32 * torch.rsqrt(variance + eps) * w32
    return y.to(x.dtype).reshape(orig_shape)


# ── Forward pass ────────────────────────────────────────────────────────


def run_exl3_forward(tensors: dict, weights: dict, config: dict,
                     input_ids: list) -> tuple:
    """Run the full EXL3 model forward pass, capturing per-layer boundaries.

    Args:
    - tensors, the raw EXL3 tensor dict (norms, embeddings, lm_head entries)
    - weights, the reconstructed linear weights dict
    - config, the parsed model config
    - input_ids, the flat token id list

    Returns:
    - (per_layer_boundaries, logits), one (layer_input, layer_output) pair
      per layer as fp16 cpu tensors, the final logits on DEVICE
    """
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    rms_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 1000000.0)
    max_seq_len = config.get("max_position_embeddings", 40960)

    ids = torch.tensor([input_ids], dtype=torch.long)
    batch, seq_len = ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Precompute RoPE cos/sin
    cos, sin = precompute_freqs_cis_reimpl_exl3(head_dim, max_seq_len, theta=rope_theta)
    cos = cos.to(DTYPE).to(DEVICE)
    sin = sin.to(DTYPE).to(DEVICE)

    # Embedding lookup step
    embed_weight = tensors["_embeddings"]["model.embed_tokens.weight"].to(DEVICE)
    h = torch.nn.functional.embedding(ids.to(DEVICE), embed_weight)

    # Gather norm weights
    norms: dict = {}
    for k, v in tensors["_norms"].items():
        norms[k] = v.to(DEVICE)

    boundaries: list = [None] * int(config["num_hidden_layers"])

    for layer_idx in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{layer_idx}"
        layer_input = h.clone()

        # ── RMS Norm (input_layernorm) ──
        ln_weight = norms[f"{prefix}.input_layernorm.weight"]
        h_norm = rms_norm(h, ln_weight, rms_eps)

        # ── Self-Attention ──
        q_weight = weights[f"{prefix}.self_attn.q_proj"]
        k_weight = weights[f"{prefix}.self_attn.k_proj"]
        v_weight = weights[f"{prefix}.self_attn.v_proj"]
        o_weight = weights[f"{prefix}.self_attn.o_proj"]

        q_suh = tensors[f"{prefix}.self_attn.q_proj"]["suh"].to(DEVICE)
        q_svh = tensors[f"{prefix}.self_attn.q_proj"]["svh"].to(DEVICE)
        k_suh = tensors[f"{prefix}.self_attn.k_proj"]["suh"].to(DEVICE)
        k_svh = tensors[f"{prefix}.self_attn.k_proj"]["svh"].to(DEVICE)
        v_suh = tensors[f"{prefix}.self_attn.v_proj"]["suh"].to(DEVICE)
        v_svh = tensors[f"{prefix}.self_attn.v_proj"]["svh"].to(DEVICE)
        o_suh = tensors[f"{prefix}.self_attn.o_proj"]["suh"].to(DEVICE)
        o_svh = tensors[f"{prefix}.self_attn.o_proj"]["svh"].to(DEVICE)

        # Q/K/V projection passes
        q = linear_forward(h_norm, q_weight, q_suh, q_svh)
        k = linear_forward(h_norm, k_weight, k_suh, k_svh)
        v = linear_forward(h_norm, v_weight, v_suh, v_svh)

        # Reshape to multi-head format
        q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # QK norm application
        q_norm_w = norms.get(f"{prefix}.self_attn.q_norm.weight")
        k_norm_w = norms.get(f"{prefix}.self_attn.k_norm.weight")
        if q_norm_w is not None:
            q = rms_norm(q, q_norm_w, rms_eps)
        if k_norm_w is not None:
            k = rms_norm(k, k_norm_w, rms_eps)

        # RoPE application step
        q, k = apply_rotary_pos_emb_reimpl_exl3(q, k, cos, sin, position_ids)
        q = q.to(torch.float16)
        k = k.to(torch.float16)

        # GQA head expansion
        if num_kv_heads < num_heads:
            n_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        # SDPA attention call
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            scale=head_dim ** -0.5)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, num_heads * head_dim)

        # O projection pass
        attn_output = linear_forward(attn_output, o_weight, o_suh, o_svh)

        # Residual add step
        h = layer_input + attn_output
        residual = h

        # ── MLP ──
        h_norm = rms_norm(h, norms[f"{prefix}.post_attention_layernorm.weight"], rms_eps)

        gate_weight = weights[f"{prefix}.mlp.gate_proj"]
        up_weight = weights[f"{prefix}.mlp.up_proj"]
        down_weight = weights[f"{prefix}.mlp.down_proj"]

        gate_suh = tensors[f"{prefix}.mlp.gate_proj"]["suh"].to(DEVICE)
        gate_svh = tensors[f"{prefix}.mlp.gate_proj"]["svh"].to(DEVICE)
        up_suh = tensors[f"{prefix}.mlp.up_proj"]["suh"].to(DEVICE)
        up_svh = tensors[f"{prefix}.mlp.up_proj"]["svh"].to(DEVICE)
        down_suh = tensors[f"{prefix}.mlp.down_proj"]["suh"].to(DEVICE)
        down_svh = tensors[f"{prefix}.mlp.down_proj"]["svh"].to(DEVICE)

        gate = linear_forward(h_norm, gate_weight, gate_suh, gate_svh)
        up = linear_forward(h_norm, up_weight, up_suh, up_svh)

        gate = torch.nn.functional.silu(gate)
        mlp_output = gate * up
        mlp_output = linear_forward(mlp_output, down_weight, down_suh, down_svh)

        h = residual + mlp_output

        boundaries[layer_idx] = (
            layer_input.cpu().to(DTYPE).contiguous(),
            h.clone().cpu().to(DTYPE).contiguous(),
        )
        print(f"  Layer {layer_idx:02d}: boundary captured")

    # ── Final norm ──
    h = rms_norm(h, norms["model.norm.weight"], rms_eps)

    # ── LM Head (EXL3-quantized) ──
    if "lm_head" in weights:
        logits = linear_forward(h, weights["lm_head"],
                                tensors["lm_head"]["suh"].to(DEVICE),
                                tensors["lm_head"]["svh"].to(DEVICE))
    else:
        # Fallback path through the tied embed_tokens weight.
        logits = torch.nn.functional.linear(h, embed_weight)

    return boundaries, logits


# ── Saving ──────────────────────────────────────────────────────────────


def save_layer_stats(output_dir: Path, layer_idx: int, frame: bytes) -> Path:
    """Write the 004 uniform stats frame of one layer boundary, the sidecar carrying the layer boundary entries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"layer-{layer_idx:02d}.safetensor.stats.json.zst"
    write_text_zst(str(path), frame)
    return path


def _hex_f64_to_float(s: str) -> float:
    """One hex f64 bit pattern back to a float."""
    return struct.unpack("<d", struct.pack("<Q", int(s, 16)))[0]


def load_recorded_decisions():
    """Returns the already recorded decisions frame of this model dir or None."""
    path = OUTPUT_DIR / "final_logits.decisions.json.zst"
    if not path.exists():
        return None
    return json.loads(read_text_zst(str(path)))


def check_recorded_decisions(old: dict | None, records: list) -> None:
    """Pre-write guard against the already recorded decisions frame.

    - per position the recorded argmax id reproduces exactly
    - the margin sits inside the gen_rerecord_chains_005 band, 0.5
    - a wider divergence means the machine does not reproduce the committed recording, the write aborts instead of corrupting the chain truth
    """
    if old is None:
        print("  No recorded decisions frame, the guard passes trivially.")
        return
    schema = old.get("schema")
    print("  pos  argmax_old  argmax_new  match  margin_old  margin_new")
    for pos, rec in enumerate(records):
        old_step = old["steps"][pos]
        if schema == "ttt-tf-005-argmax-decisions":
            margin_old = _hex_f64_to_float(old_step["margin"])
        else:
            margin_old = old_step["argmax_margin"]
        margin_new = _hex_f64_to_float(rec["margin"])
        ok = rec["argmax_id"] == old_step["argmax_id"]
        print(f"  {pos:>3}  {old_step['argmax_id']:>10}  {rec['argmax_id']:>10}"
              f"  {str(ok):>5}  {margin_old:>10.6g}  {margin_new:>10.6g}")
        if not ok:
            raise SystemExit(
                f"argmax divergence at position {pos}: computed "
                f"{rec['argmax_id']} vs recorded {old_step['argmax_id']} "
                f"({BACKEND} on {DEVICE.type} does not reproduce the "
                "committed recording, refusing to corrupt the chain truth)")
        if abs(margin_new - margin_old) > 0.5:
            raise SystemExit(
                f"margin divergence at position {pos}: computed "
                f"{margin_new} vs recorded {margin_old}")


def compare_layer_stats(output_dir: Path, layer_bytes: list) -> None:
    """Fingerprint deltas of the new stats frames against the recorded frames, a drift report over the band inputs before the overwrite.

    Args:
    - layer_bytes, the new (layer_idx, frame_bytes) pairs in layer order
    """
    max_abs = 0.0
    max_rel = 0.0
    where = ""
    for layer_idx, frame in layer_bytes:
        obj = json.loads(frame.decode("utf-8"))
        old_path = output_dir / f"layer-{layer_idx:02d}.safetensor.stats.json.zst"
        if not old_path.exists():
            continue
        old = json.loads(read_text_zst(str(old_path)))
        for name, tensor in obj["tensors"].items():
            old_tensor = old["tensors"].get(name)
            if old_tensor is None:
                continue
            for key in ("mean_abs", "signed_mean", "tail_probability",
                        "max_magnitude"):
                new_v = _hex_f64_to_float(tensor[key])
                old_v = _hex_f64_to_float(old_tensor[key])
                if math.isnan(new_v) or math.isnan(old_v) or old_v == 0.0:
                    continue
                delta = abs(new_v - old_v)
                rel = delta / max(abs(old_v), 1e-30)
                if delta > max_abs:
                    max_abs, where = delta, f"layer {layer_idx:02d} {name} {key}"
                if rel > max_rel:
                    max_rel, where = rel, f"layer {layer_idx:02d} {name} {key}"
    print(f"  Incumbent-frame drift: max abs {max_abs:.3g}, max rel {max_rel:.3g}"
          f" (at {where})")


# ── Main ────────────────────────────────────────────────────────────────


def main():
    """Record the per-layer stats frames plus the final logits decisions."""
    print(f"Model: {MODEL_DIR}")
    print(f"Input: {INPUT_TEXT}")
    print(f"Input ids: {INPUT_IDS}")
    print(f"Device: {DEVICE}")
    print(f"Backend: {BACKEND}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    config = load_config(MODEL_DIR)
    print(f"Loading EXL3 tensors from {MODEL_PATH}...")
    tensors = get_exl3_tensors(MODEL_PATH)

    if "lm_head" in tensors:
        print(f"  Found lm_head in main EXL3 layers (has trellis: "
              f"{tensors['lm_head'].get('trellis') is not None})")
    else:
        print(f"  Warning: lm_head not found as EXL3 layer, "
              f"will use embed_tokens.weight as fallback")

    print(f"\nReconstructing EXL3 weights...")
    weights = reconstruct_and_cache(tensors, config)

    print(f"\nRunning forward pass for {int(config['num_hidden_layers'])} layers...")
    boundaries, logits = run_exl3_forward(tensors, weights, config, INPUT_IDS)

    # Per-layer 004 uniform stats frames, one per layer boundary.
    print(f"\nWriting per-layer stats frames...")
    layer_bytes = []
    for i, (layer_input, layer_output) in enumerate(boundaries):
        if layer_input is None:
            print(f"  Layer {i:02d}: SKIPPED")
            continue
        frame = stats_file_bytes(
            f"layer-{i:02d}",
            [("layer_input", layer_input), ("layer_output", layer_output)])
        layer_bytes.append((i, frame))
    compare_layer_stats(OUTPUT_DIR, layer_bytes)
    for i, frame in layer_bytes:
        print(f"  Layer {i:02d}: {save_layer_stats(OUTPUT_DIR, i, frame)}")

    # Final logits 005 argmax decisions frame, the canonical chains
    # recording code, one record per position over the top-32 support.
    print(f"\nSaving final logits argmax decisions...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logits_cpu = logits.detach().cpu()
    records = []
    for pos in range(min(NUM_POSITIONS, logits_cpu.shape[1])):
        row = logits_cpu[0, pos].to(torch.float32)
        records.append(argmax_record_from_row(row))
    check_recorded_decisions(load_recorded_decisions(), records)
    decisions_path = OUTPUT_DIR / "final_logits.decisions.json.zst"
    write_argmax_decisions(str(decisions_path), "final_logits.decisions",
                           records, grid_of(logits_cpu))
    print(f"  Decisions: {decisions_path} ({len(records)} records)")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
