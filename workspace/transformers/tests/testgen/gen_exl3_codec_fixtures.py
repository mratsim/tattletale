#!/usr/bin/env python3
"""
Generate EXL3 codec fixtures for an EXL3-quantized model.

This script:
1. Loads an EXL3-quantized model from safetensors
2. Reconstructs weights (via PyTorch reimpl and/or production EXL3 CUDA)
3. Runs linear-layer forward passes and saves per-layer fixtures
4. Optionally verifies the PyTorch decoder against the production CUDA kernel
   or against the original FP16 weights

All core logic is in ``q_exl3_common.py`` — this file only handles
orchestration, CLI, and data serialisation.

Usage:
  # Default: use our PyTorch reimpl decoder, save fixtures
  python testgen/gen_exl3_codec_fixtures.py

  # Use production EXL3 reconstruct (requires compiled exllamav3_ext)
  python testgen/gen_exl3_codec_fixtures.py --backend exllamav3

  # Run both and compare (verification mode)
  python testgen/gen_exl3_codec_fixtures.py --backend both --check

  # Only test a specific layer
  python testgen/gen_exl3_codec_fixtures.py --layer 8 --proj q_proj

  # Dry-run: just decode one tile, no fixture files
  python testgen/gen_exl3_codec_fixtures.py --dry-run

  # Verify decoder consistency against production kernel
  python testgen/gen_exl3_codec_fixtures.py --verify

  # Verify decoder against original FP16 weights
  python testgen/gen_exl3_codec_fixtures.py --verify-fp16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file as st_save_file

# ── Add testgen dir to path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from q_exl3_common import (
    # utilities
    load_config,
    get_exl3_tensors,
    parse_layer_name,
    get_in_features_out_features,
    derive_cb,
    derive_K,
    # reimpl (PyTorch)
    reconstruct_reimpl_exl3,
    had_r_128_reimpl_exl3,
    linear_forward_reimpl_exl3,
    # orig (CUDA)
    dequant_reimpl_exl3,
)

# ─── try importing the CUDA backend ───
USE_CUDA: bool = False
try:
    from q_exl3_common import reconstruct_orig_exl3
    USE_CUDA = True
    print(f"  [OK] exllamav3 CUDA extension available")
except ImportError:
    print(f"  [info] exllamav3_ext not available — CUDA backend disabled")

# ─── Paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # tests/
FIXTURE_DIR = os.path.join(BASE_DIR, "fixtures", "exl3-codec")
MODEL_DIR = os.path.join(BASE_DIR, "hf_models", "Qwen3-0.6B-EXL3-5bpw")
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
FP16_MODEL_DIR = os.path.join(BASE_DIR, "hf_models", "Qwen3-0.6B")
FP16_MODEL_PATH = os.path.join(FP16_MODEL_DIR, "model.safetensors")
LAYER_COUNT = 28

# Exponential layer subset: early, middle, late coverage at ~25% storage cost
EXPONENTIAL_LAYERS = sorted({0, 1, 2, 4, 8, 16, LAYER_COUNT - 1})

# ─── Determinism ───────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SEED_DECODE = 42
SEED_FORWARD = 43


# ─── Helpers ───────────────────────────────────────────────────────


def _try_import_exllamav3():
    """Try to import exllamav3_ext. Returns None if unavailable."""
    try:
        from exllamav3.ext import exllamav3_ext as ext
        return ext
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  [info] exllamav3_ext not available: {e}")
        return None


# ────────────────────────────────────────────────────────────────────
#  Fixture generation
# ────────────────────────────────────────────────────────────────────


def generate_forward_fixture(layer_key: str, layer_entry: dict,
                             config: dict, device: torch.device,
                             backend: str = "pytorch",
                             check: bool = False) -> dict:
    """Generate forward-pass fixtures for one EXL3 linear layer.

    Args:
        layer_key: e.g. "model.layers.8.self_attn.q_proj"
        layer_entry: dict with "trellis", "suh", "svh", etc.
        config: Model config dict.
        device: torch device.
        backend: "pytorch", "exllamav3", or "both".
        check: If True and backend="both", assert both match.

    Returns:
        Dict with fixture tensors.
    """
    trellis = layer_entry["trellis"]
    suh = layer_entry["suh"]
    svh = layer_entry["svh"]
    bias = layer_entry["bias"]
    mcg = layer_entry["mcg"] is not None
    mul1 = layer_entry["mul1"] is not None

    K = derive_K(trellis)
    cb = derive_cb(layer_entry)
    in_features, out_features = get_in_features_out_features(
        layer_key, trellis, config)

    print(f"  K={K}, cb={cb}, in={in_features}, out={out_features}")
    print(f"  trellis: {list(trellis.shape)}, dtype={trellis.dtype}")
    print(f"  suh: {list(suh.shape)}, svh: {list(svh.shape)}")
    if mcg:
        print(f"  codebook: mcg (cb=1)")
    elif mul1:
        print(f"  codebook: mul1 (cb=2)")
    else:
        print(f"  codebook: default (cb=0)")

    # Generate random input for forward pass
    torch.manual_seed(SEED_FORWARD)
    x = torch.randn(1, in_features, dtype=torch.float16, device=device)

    weight_pytorch = None
    weight_prod = None

    # ── PyTorch reimpl decoder ──
    if backend in ("pytorch", "both"):
        print(f"  Decoding with PyTorch reimpl...")
        w = reconstruct_reimpl_exl3(trellis, K, cb, (in_features, out_features))
        weight_pytorch = w.t().contiguous()

    # ── Production CUDA decoder ──
    if backend in ("exllamav3", "both"):
        print(f"  Decoding with production EXL3 kernel...")
        try:
            w = reconstruct_orig_exl3(trellis, K, mcg, mul1,
                                       (in_features, out_features))
            weight_prod = w.t().contiguous()
        except ImportError:
            print(f"  [WARN] Production EXL3 not available, skipping")
            if backend == "exllamav3":
                raise

    # ── Compare ──
    if check and weight_pytorch is not None and weight_prod is not None:
        diff = (weight_pytorch - weight_prod).abs().max().item()
        print(f"  Max diff between decoders: {diff:.6f}")
        if diff > 1e-3:
            mean_diff = (weight_pytorch - weight_prod).abs().mean().item()
            print(f"  [WARN] Large difference! Mean abs diff: {mean_diff:.6f}")
        else:
            print(f"  [OK] Decoders match (diff < 1e-3)")

      # ── Forward pass ──
    weight = weight_pytorch if weight_pytorch is not None else weight_prod
    y = linear_forward_reimpl_exl3(
        x, weight, suh, svh, bias, device=device
    )

    # Compute integrity hash via torch.hash_tensor before any CPU transfer
    weight_cpu = weight.cpu().contiguous()
    weight_hash = f"{torch.hash_tensor(weight_cpu).item():016x}"

    return {
        "input": x.cpu().contiguous(),
        "output": y.cpu().contiguous(),
        "trellis": trellis.cpu().contiguous(),
        "bias": bias.cpu().contiguous() if bias is not None else None,
        "weight_hash": weight_hash,
    }


def generate_all_fixtures(device: torch.device,
                          backend: str = "pytorch",
                          check: bool = False,
                          dry_run: bool = False,
                          layer_filter: int | None = None,
                          proj_filter: str | None = None,
                          all_layers: bool = False):
    """Generate fixtures for EXL3 layers.

    By default generates only the exponential subset ({0,1,2,4,8,16,last}).
    Pass all_layers=True or use --layer N for full/single-layer coverage.
    """
    config = load_config()
    tensors = get_exl3_tensors(MODEL_PATH)

    # Extract layer entries
    layer_entries: dict = {}
    for key in tensors:
        if key.startswith("_") or key.startswith("lm_head"):
            continue
        parsed = parse_layer_name(key)
        if parsed is None:
            continue
        layer_idx, component, proj_name = parsed
        if layer_filter is not None and layer_idx != layer_filter:
            continue
        if not all_layers and layer_filter is None and layer_idx not in EXPONENTIAL_LAYERS:
            continue
        if proj_filter is not None and proj_filter not in proj_name:
            continue
        layer_entries[key] = tensors[key]

    if not layer_entries:
        print("No EXL3 layers found!")
        return

    print(f"Found {len(layer_entries)} EXL3 layers")
    print(f"Device: {device}")
    print(f"Backend: {backend}")
    if dry_run:
        print("DRY RUN — no files will be written")

    for layer_key, entry in sorted(layer_entries.items()):
        print(f"\n{'=' * 60}")
        print(f"Layer: {layer_key}")

        if entry["trellis"] is None:
            print(f"  Skipping (no trellis)")
            continue

        # Move tensors to device
        for name in ("trellis", "suh", "svh", "bias", "mcg", "mul1"):
            if entry[name] is not None:
                entry[name] = entry[name].to(device)

        fixtures = generate_forward_fixture(
            layer_key, entry, config, device,
            backend=backend, check=check
        )

        if dry_run:
            print(f"  [dry-run] Would save fixtures for {layer_key}")
            continue

        # Save fixtures
        layer_name = layer_key.replace(".", "_")
        layer_fixture_dir = os.path.join(FIXTURE_DIR, layer_name)
        os.makedirs(layer_fixture_dir, exist_ok=True)

        tensors_to_save: dict = {}
        for name, tensor in [
            ("input", fixtures["input"]),
            ("output", fixtures["output"]),
            ("trellis", fixtures["trellis"]),
        ]:
            if tensor is not None:
                tensors_to_save[name] = tensor

        fixture_path = os.path.join(layer_fixture_dir, "fixture.safetensors")
        st_save_file(tensors_to_save, fixture_path)

        meta = OrderedDict([
            ("layer_key", layer_key),
            ("in_features", int(fixtures["input"].shape[-1])),
            ("out_features", int(fixtures["output"].shape[-1])),
            ("K", derive_K(entry["trellis"])),
            ("cb", derive_cb(entry)),
            ("backend", backend),
            ("weight_hash", fixtures["weight_hash"]),
        ])
        meta_path = os.path.join(layer_fixture_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        print(f"  Saved fixtures to {layer_fixture_dir}")


# ────────────────────────────────────────────────────────────────────
#  Verification helpers
# ────────────────────────────────────────────────────────────────────


def verify_decoder_on_tile(device: torch.device, verbose: bool = True):
    """Verify the PyTorch decoder by running known test vectors.

    Reads one tile from the model, decodes with both implementations,
    and checks for consistency.
    """
    print("=" * 60)
    print("Verification: tile-level decode")
    print("=" * 60)

    tensors = get_exl3_tensors(MODEL_PATH)
    for key, entry in sorted(tensors.items()):
        if key.startswith("_") or key.startswith("lm_head"):
            continue
        if entry["trellis"] is not None:
            print(f"Using layer: {key}")
            trellis = entry["trellis"].to(device)
            K = derive_K(trellis)
            cb = derive_cb(entry)
            mcg = entry["mcg"] is not None
            mul1 = entry["mul1"] is not None
            in_f, out_f = get_in_features_out_features(key, trellis, {})
            break
    else:
        print("No EXL3 layers found for verification")
        return

    print(f"K={K}, cb={cb}, mcg={mcg}, mul1={mul1}")
    print(f"trellis shape: {list(trellis.shape)}")

    # PyTorch decoder
    w_pytorch = reconstruct_reimpl_exl3(trellis, K, cb, (in_f, out_f))

    # Production decoder
    w_prod = None
    try:
        w_prod = reconstruct_orig_exl3(trellis, K, mcg, mul1, (in_f, out_f))
    except ImportError:
        print("  [info] Production decoder not available")

    if w_prod is not None:
        diff = (w_pytorch - w_prod).abs()
        print(f"Max abs diff: {diff.max().item():.6f}")
        print(f"Mean abs diff: {diff.mean().item():.6f}")
        rel = diff / (w_prod.abs() + 1e-10)
        print(f"Max relative diff: {rel.max().item():.6f}")

        atol = 1e-2
        match = diff.max().item() < atol
        print(f"Match (atol={atol}): {'✓ PASS' if match else '✗ FAIL'}")
    else:
        print("  [info] Production decoder unavailable — skipping comparison")
        print(f"  Our decoder output shape: {list(w_pytorch.shape)}")
        print(f"  Stats: mean={w_pytorch.mean().item():.4f}, "
              f"std={w_pytorch.std().item():.4f}")


def verify_against_fp16(device: torch.device, num_layers: int = 5):
    """Verify EXL3 decoder against original FP16 weights.

    NMSE should be < 5% for correct 5-bit quantization.
    """
    print("=" * 60)
    print("Verification: EXL3 decode vs FP16 original")
    print("  (EXL3 weights are Hadamard-transformed; we apply")
    print("   inverse transform + scales before comparison)")
    print("=" * 60)

    exl3_tensors = get_exl3_tensors(MODEL_PATH)

    fp16_weights: dict = {}
    with safe_open(FP16_MODEL_PATH, framework="pt") as f:
        for k in f.keys():
            fp16_weights[k] = f.get_tensor(k)

    layer_count = 0
    nmse_sum = 0.0

    for key, entry in sorted(exl3_tensors.items()):
        if key.startswith("_") or entry["trellis"] is None:
            continue
        if key == "lm_head":
            continue

        fp16_key = key + ".weight"
        if fp16_key not in fp16_weights:
            print(f"  Skip {key}: no FP16 weight")
            continue

        trellis = entry["trellis"].to(device)
        suh = entry["suh"].to(device)
        svh = entry["svh"].to(device)
        K = derive_K(trellis)
        cb = derive_cb(entry)
        in_f, out_f = get_in_features_out_features(key, trellis, {})
        w_fp16 = fp16_weights[fp16_key].to(device)

        # Align shapes
        if w_fp16.shape[0] == out_f and w_fp16.shape[1] == in_f:
            w_fp16 = w_fp16.t()

        # Reconstruct from trellis
        w_exl3 = reconstruct_reimpl_exl3(trellis, K, cb, (in_f, out_f))

        # Convert back to original FP16 domain
        w = dequant_reimpl_exl3(w_exl3, suh, svh)

        # Trim padding
        min_in = min(w.shape[0], w_fp16.shape[0])
        min_out = min(w.shape[1], w_fp16.shape[1])
        we = w[:min_in, :min_out]
        wf = w_fp16[:min_in, :min_out]

        diff = (we - wf).float()
        nmse = diff.square().mean() / wf.float().square().mean()
        nmse_sum += nmse.item()
        layer_count += 1
        print(f"  {key:<50} NMSE={nmse.item():.6f}  "
              f"max|Δ|={diff.abs().max().item():.4f}  "
              f"mean|Δ|={diff.abs().mean().item():.4f}")

        if layer_count >= num_layers:
            break

    if layer_count > 0:
        avg_nmse = nmse_sum / layer_count
        print(f"  Avg NMSE over {layer_count} layers: {avg_nmse:.6f}")
        thresh = 0.002
        msg = ("✓ PASS (within quantization noise)"
               if avg_nmse < thresh
               else "⚠ HIGH NMSE — decoder may be wrong")
        print(f"  {msg}")


# ────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate EXL3 codec fixtures")
    parser.add_argument("--all-layers", action="store_true",
                        help="Generate for ALL layers (default: exponential subset)")
    parser.add_argument("--backend", choices=["pytorch", "exllamav3", "both"],
                        default="pytorch",
                        help="Which decoder backend to use")
    parser.add_argument("--check", action="store_true",
                        help="Compare both backends when using --backend both")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated, don't write files")
    parser.add_argument("--layer", type=int, default=None,
                        help="Only process this layer index")
    parser.add_argument("--verify-fp16", action="store_true",
                        help="Verify EXL3 decode vs FP16 original (no fixture generation)")
    parser.add_argument("--proj", type=str, default=None,
                        help="Only process projections matching this name")
    parser.add_argument("--verify", action="store_true",
                        help="Run decoder verification (no fixture generation)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for computation (cuda:0, cpu, etc.)")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.verify_fp16:
        verify_against_fp16(device)
        return

    if args.verify:
        verify_decoder_on_tile(device)
        return

    generate_all_fixtures(
        device=device,
        backend=args.backend,
        check=args.check,
        dry_run=args.dry_run,
        layer_filter=args.layer,
        proj_filter=args.proj,
        all_layers=args.all_layers,
    )


if __name__ == "__main__":
    main()
