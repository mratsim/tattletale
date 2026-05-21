#!/usr/bin/env python3
"""
Generate EXL3 layer fixtures for Qwen3-0.6B using exllamav3 CUDA backend.

This script:
1. Loads the EXL3-quantized model (trellis + suh + svh per linear layer)
2. Reconstructs weights via q_exl3_common
3. Runs EXL3 linear forward (Hadamard + GEMM + Hadamard) on CUDA
4. Runs attention and transformer block forward with long residual stream pattern
5. Saves per-layer fixtures for Nim testing (test_qwen3_03_layers_exl3.nim)

Space-saving: All weights come from the EXL3 model file. Only inputs/outputs are saved.

Determinism:
- Each generator calls torch.manual_seed with its own seed constant
- CUDA determinism: cudnn.deterministic=True, cudnn.benchmark=False
- Fixture files are fully deterministic across separate process invocations

Usage:
    cd workspace/transformers
    CUDA_HOME=... PATH=... python tests/testgen/gen_exl3_03_layer_fixtures_Qwen3-0.6B.py
    python tests/testgen/gen_exl3_03_layer_fixtures_Qwen3-0.6B.py --only linear
    python tests/testgen/gen_exl3_03_layer_fixtures_Qwen3-0.6B.py --only attn
    python tests/testgen/gen_exl3_03_layer_fixtures_Qwen3-0.6B.py --only block
"""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors.torch import save_file as st_save_file

# ── Setup CUDA_HOME and PATH before importing exllamav3 ──
_venv_python = os.path.dirname(sys.executable)
_venv_bin = os.path.dirname(_venv_python)
if _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + ':' + os.environ.get("PATH", "")
if "CUDA_HOME" not in os.environ:
    import glob
    sp_base = os.path.join(os.path.dirname(_venv_python), '..', 'lib')
    for d in glob.glob(os.path.join(sp_base, 'python*', 'site-packages', 'nvidia', 'cu*')):
        if os.path.exists(os.path.join(d, 'bin', 'nvcc')):
            os.environ["CUDA_HOME"] = os.path.abspath(d)
            break
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# ── Add testgen dir to path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from q_exl3_common import (
    get_exl3_tensors,
    get_in_features_out_features,
    derive_K,
    derive_cb,
    load_config,
    linear_forward_orig_exl3,
    had_r_128_orig_exl3,
    rms_norm_orig_exl3,
    precompute_freqs_cis_reimpl_exl3,
    apply_rotary_pos_emb_reimpl_exl3,
)

# ─── Try CUDA backend ───
USE_CUDA: bool = False
try:
    from q_exl3_common import reconstruct_orig_exl3
    USE_CUDA = True
    print(f"  [OK] exllamav3 CUDA extension loaded")
except Exception as e:
    print(f"  [WARN] exllamav3 CUDA extension not available: {e}")
    print(f"  [WARN] Falling back to PyTorch reimpl")
    from q_exl3_common import reconstruct_reimpl_exl3

# ─── Determinism ──────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "Qwen3-0.6B-EXL3-5bpw"
LAYER_IDX = 0  # Test layer 0 (first layer, simplest residual state)
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "exl3-layers", f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
MODEL_DIR = os.path.join(GRANDPARENT_DIR, "hf_models", MODEL_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

# Per-generator seeds — independent, order-agnostic.
SEED_LINEAR = 42
SEED_BLOCK = 43
SEED_ATTN = 44

# EXL3 operates in float16 on CUDA
DTYPE = torch.float16
DEVICE = torch.device("cuda:0") if USE_CUDA and torch.cuda.is_available() else torch.device("cpu")
print(f"  [info] Using device: {DEVICE}")


# ─── Helpers ──────────────────────────────────────────────────────────


def ensure_fixture_dir() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(layer_name: str, case_num: int, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors format with separate metadata file."""
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    sorted_tensors = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in sorted(tensors.items())
        if tensor is not None
    )
    st_save_file(sorted_tensors, filepath)

    metadata_path = filepath + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")

    return filepath


def load_norm_weights(tensors: dict) -> dict:
    """Extract norm weights from EXL3 model tensors."""
    norms: dict = {}
    for k, v in tensors.get("_norms", {}).items():
        norms[k] = v
    return norms


def _build_linear_layer(layer_key: str, entry: dict, config: dict):
    """Reconstruct weight for one EXL3 linear layer using CUDA or PyTorch."""
    trellis = entry["trellis"].to(DEVICE)
    suh = entry["suh"].to(DEVICE)
    svh = entry["svh"].to(DEVICE)
    bias = entry.get("bias")
    if bias is not None:
        bias = bias.to(DEVICE)

    K = derive_K(trellis)
    cb = derive_cb(entry)
    in_f, out_f = get_in_features_out_features(layer_key, trellis, config)
    mcg = entry.get("mcg") is not None
    mul1 = entry.get("mul1") is not None

    if USE_CUDA:
        w = reconstruct_orig_exl3(trellis, K, mcg, mul1, (in_f, out_f))
    else:
        w = reconstruct_reimpl_exl3(trellis, K, cb, (in_f, out_f))

    weight = w.contiguous()  # [in_features, out_features] for ext.hgemm

    return {
        "weight": weight,
        "suh": suh,
        "svh": svh,
        "bias": bias,
        "in_features": in_f,
        "out_features": out_f,
        "K": K,
        "cb": cb,
    }


# ─── Generators ──────────────────────────────────────────────────────


def generate_linear_fixtures(tensors: dict, config: dict) -> None:
    """Generate fixtures for EXL3 linear layers."""
    torch.manual_seed(SEED_LINEAR)

    prefix = f"model.layers.{LAYER_IDX}"
    proj_names = [
        ("self_attn.q_proj", f"{prefix}.self_attn.q_proj"),
        ("self_attn.k_proj", f"{prefix}.self_attn.k_proj"),
        ("self_attn.v_proj", f"{prefix}.self_attn.v_proj"),
        ("self_attn.o_proj", f"{prefix}.self_attn.o_proj"),
        ("mlp.gate_proj", f"{prefix}.mlp.gate_proj"),
        ("mlp.up_proj", f"{prefix}.mlp.up_proj"),
        ("mlp.down_proj", f"{prefix}.mlp.down_proj"),
    ]

    for proj_short, proj_key in proj_names:
        if proj_key not in tensors:
            print(f"  Skipping {proj_short}: not in tensors")
            continue

        entry = tensors[proj_key]
        if entry["trellis"] is None:
            print(f"  Skipping {proj_short}: no trellis")
            continue

        layer_info = _build_linear_layer(proj_key, entry, config)
        in_f = layer_info["in_features"]

        test_shapes = [
            (2, 4),   # 00: batch=2, seq=4 (flattened)
            (1, 1),   # 01: single token
            (1, 8),   # 02: short sequence
            (2, 4),   # 03: zeros (same shape as 00)
        ]

        for case_num, (batch, seq) in enumerate(test_shapes):
            total = batch * seq

            if case_num == 3:  # zeros
                x = torch.zeros(total, in_f, dtype=DTYPE, device=DEVICE)
            else:
                x = torch.randn(total, in_f, dtype=DTYPE, device=DEVICE)

            y = linear_forward_orig_exl3(
                x, layer_info["weight"], layer_info["suh"], layer_info["svh"],
                layer_info["bias"], device=DEVICE
            )

            layer_name = f"linear-{proj_short}"
            save_fixture(
                layer_name, case_num,
                {
                    "model": MODEL_NAME,
                    "layer": proj_key,
                    "case": ["normal_forward", "single_token", "short_sequence", "zeros_input"][case_num],
                    "in_features": in_f,
                    "out_features": layer_info["out_features"],
                    "K": layer_info["K"],
                    "cb": layer_info["cb"],
                    "backend": "exllamav3_cuda" if USE_CUDA else "pytorch",
                },
                {
                    "input": x.cpu(),
                    "output": y.cpu(),
                },
            )

        print(f"  Generated {proj_short} fixtures (4 cases, backend={'cuda' if USE_CUDA else 'cpu'})")


def generate_attn_fixtures(tensors: dict, config: dict) -> None:
    """Generate attention fixtures for RopeGQAttention with EXL3 linear layers."""
    torch.manual_seed(SEED_ATTN)
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    rms_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 1000000.0)
    max_seq_len = config.get("max_position_embeddings", 40960)
    prefix = f"model.layers.{LAYER_IDX}"
    norms = load_norm_weights(tensors)

    cos, sin = precompute_freqs_cis_reimpl_exl3(head_dim, max_seq_len, theta=rope_theta)
    cos = cos.to(DTYPE).to(DEVICE)
    sin = sin.to(DTYPE).to(DEVICE)

    proj_names = {
        "q_proj": f"{prefix}.self_attn.q_proj",
        "k_proj": f"{prefix}.self_attn.k_proj",
        "v_proj": f"{prefix}.self_attn.v_proj",
        "o_proj": f"{prefix}.self_attn.o_proj",
    }
    linear_layers: dict = {}
    for name, key in proj_names.items():
        if key in tensors and tensors[key]["trellis"] is not None:
            linear_layers[name] = _build_linear_layer(key, tensors[key], config)
        else:
            print(f"  Warning: {name} not available, skipping attn fixtures")
            return

    q_norm_weight = norms.get(f"{prefix}.self_attn.q_norm.weight").to(DEVICE)
    k_norm_weight = norms.get(f"{prefix}.self_attn.k_norm.weight").to(DEVICE)

    for case_num, (batch, seq) in enumerate([(1, 1), (2, 8)]):
        print(f"  Generating attn case {case_num}: batch={batch}, seq={seq}")
        hidden_states = torch.randn(batch, seq, hidden_size, dtype=DTYPE, device=DEVICE)
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0).expand(batch, -1).contiguous()

        q = linear_forward_orig_exl3(
            hidden_states, linear_layers["q_proj"]["weight"],
            linear_layers["q_proj"]["suh"], linear_layers["q_proj"]["svh"],
            linear_layers["q_proj"]["bias"], device=DEVICE)
        k = linear_forward_orig_exl3(
            hidden_states, linear_layers["k_proj"]["weight"],
            linear_layers["k_proj"]["suh"], linear_layers["k_proj"]["svh"],
            linear_layers["k_proj"]["bias"], device=DEVICE)
        v = linear_forward_orig_exl3(
            hidden_states, linear_layers["v_proj"]["weight"],
            linear_layers["v_proj"]["suh"], linear_layers["v_proj"]["svh"],
            linear_layers["v_proj"]["bias"], device=DEVICE)

        q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

        if q_norm_weight is not None:
            q = rms_norm_orig_exl3(q, q_norm_weight, rms_eps)
        if k_norm_weight is not None:
            k = rms_norm_orig_exl3(k, k_norm_weight, rms_eps)

        q, k = apply_rotary_pos_emb_reimpl_exl3(q, k, cos, sin, position_ids)
        q = q.to(DTYPE)
        k = k.to(DTYPE)

        # GQA: repeat K and V to match Q heads before SDPA
        if num_kv_heads < num_heads:
            n_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            scale=head_dim ** -0.5)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, num_heads * head_dim)

        out = linear_forward_orig_exl3(
            attn_out, linear_layers["o_proj"]["weight"],
            linear_layers["o_proj"]["suh"], linear_layers["o_proj"]["svh"],
            linear_layers["o_proj"]["bias"], device=DEVICE)

        cos_ref = cos[position_ids[0]].unsqueeze(0).expand(batch, -1, -1).to(DTYPE).cpu()
        sin_ref = sin[position_ids[0]].unsqueeze(0).expand(batch, -1, -1).to(DTYPE).cpu()

        save_fixture("attn", case_num, {
            "model": MODEL_NAME, "layer": f"{prefix}.self_attn",
            "case": ["single_token", "batch2_seq8"][case_num],
            "backend": "exllamav3_cuda" if USE_CUDA else "pytorch",
        }, {
            "hidden_states": hidden_states.cpu(),
            "cos": cos_ref, "sin": sin_ref,
            "position_ids": position_ids.cpu(),
            "output": out.cpu(),
        })
    print(f"  Generated attn fixtures (2 cases, backend={'cuda' if USE_CUDA else 'cpu'})")


def generate_block_fixtures(tensors: dict, config: dict) -> None:
    """Generate fixtures for full transformer block with EXL3 linear layers
    and long residual stream pattern.
    """
    torch.manual_seed(SEED_BLOCK)

    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    rms_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 1000000.0)
    max_seq_len = config.get("max_position_embeddings", 40960)

    prefix = f"model.layers.{LAYER_IDX}"
    norms = load_norm_weights(tensors)

    # Precompute RoPE cos/sin
    cos, sin = precompute_freqs_cis_reimpl_exl3(head_dim, max_seq_len, theta=rope_theta)
    cos = cos.to(DTYPE).to(DEVICE)
    sin = sin.to(DTYPE).to(DEVICE)

    # Build EXL3 linear layers for this transformer block
    proj_keys = {
        "q_proj": f"{prefix}.self_attn.q_proj",
        "k_proj": f"{prefix}.self_attn.k_proj",
        "v_proj": f"{prefix}.self_attn.v_proj",
        "o_proj": f"{prefix}.self_attn.o_proj",
        "gate_proj": f"{prefix}.mlp.gate_proj",
        "up_proj": f"{prefix}.mlp.up_proj",
        "down_proj": f"{prefix}.mlp.down_proj",
    }

    linear_layers: dict = {}
    for name, key in proj_keys.items():
        if key in tensors and tensors[key]["trellis"] is not None:
            linear_layers[name] = _build_linear_layer(key, tensors[key], config)
        else:
            print(f"  Warning: {name} not available, skipping block fixtures")
            return

    # Norm weights on device
    input_ln_weight = norms.get(f"{prefix}.input_layernorm.weight").to(DEVICE)
    post_attn_ln_weight = norms.get(f"{prefix}.post_attention_layernorm.weight").to(DEVICE)
    q_norm_weight = norms.get(f"{prefix}.self_attn.q_norm.weight").to(DEVICE)
    k_norm_weight = norms.get(f"{prefix}.self_attn.k_norm.weight").to(DEVICE)

    if input_ln_weight is None or post_attn_ln_weight is None:
        print("  Warning: norm weights not available, skipping block fixtures")
        return

    # Test cases: (batch, seq, with_residual)
    test_cases = [
        (1, 1, False),  # 00: single token, no residual (first block, decode)
        (2, 8, False),  # 01: short sequence, no residual (first block, prefill)
        (1, 1, True),   # 02: single token, with residual (middle block, decode)
        (2, 8, True),   # 03: short sequence, with residual (middle block, prefill)
    ]

    for case_num, (batch, seq, with_residual) in enumerate(test_cases):
        print(f"  Generating block case {case_num}: batch={batch}, seq={seq}, with_residual={with_residual}")

        # Input
        input_hidden_states = torch.randn(batch, seq, hidden_size, dtype=DTYPE, device=DEVICE)
        residual = torch.randn(batch, seq, hidden_size, dtype=DTYPE, device=DEVICE) if with_residual else None
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0).expand(batch, -1).contiguous()

        # Long residual stream pattern (matches Nim TransformerBlock)
        if residual is None:
            residual = input_hidden_states.clone()

        # Step 2: attn_norm.forward_with_residual(x, residual)
        attn_norm_input = input_hidden_states + residual
        attn_norm_out = rms_norm_orig_exl3(attn_norm_input, input_ln_weight, rms_eps)
        r_after_attn_norm = attn_norm_input

        # Step 3: Attention forward with EXL3 linear layers
        q = linear_forward_orig_exl3(
            attn_norm_out, linear_layers["q_proj"]["weight"],
            linear_layers["q_proj"]["suh"], linear_layers["q_proj"]["svh"],
            linear_layers["q_proj"]["bias"], device=DEVICE)
        k = linear_forward_orig_exl3(
            attn_norm_out, linear_layers["k_proj"]["weight"],
            linear_layers["k_proj"]["suh"], linear_layers["k_proj"]["svh"],
            linear_layers["k_proj"]["bias"], device=DEVICE)
        v = linear_forward_orig_exl3(
            attn_norm_out, linear_layers["v_proj"]["weight"],
            linear_layers["v_proj"]["suh"], linear_layers["v_proj"]["svh"],
            linear_layers["v_proj"]["bias"], device=DEVICE)

        # Reshape to multi-head format
        q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

        # QK norm
        if q_norm_weight is not None:
            q = rms_norm_orig_exl3(q, q_norm_weight, rms_eps)
        if k_norm_weight is not None:
            k = rms_norm_orig_exl3(k, k_norm_weight, rms_eps)

        # RoPE
        q, k = apply_rotary_pos_emb_reimpl_exl3(q, k, cos, sin, position_ids)
        q = q.to(DTYPE)
        k = k.to(DTYPE)

        # GQA: repeat K and V to match Q heads before SDPA
        if num_kv_heads < num_heads:
            n_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            scale=head_dim ** -0.5)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq, num_heads * head_dim)

        # O projection
        attn_output = linear_forward_orig_exl3(
            attn_output, linear_layers["o_proj"]["weight"],
            linear_layers["o_proj"]["suh"], linear_layers["o_proj"]["svh"],
            linear_layers["o_proj"]["bias"], device=DEVICE)

        # Long residual: h = attn_norm_input + attn_output
        h = r_after_attn_norm + attn_output

        # Step 4: mlp_norm.forward_with_residual(h, r)
        mlp_norm_out = rms_norm_orig_exl3(h, post_attn_ln_weight, rms_eps)
        output_residual = h

        # Step 5: MLP forward with EXL3 linear layers
        gate = linear_forward_orig_exl3(
            mlp_norm_out, linear_layers["gate_proj"]["weight"],
            linear_layers["gate_proj"]["suh"], linear_layers["gate_proj"]["svh"],
            linear_layers["gate_proj"]["bias"], device=DEVICE)
        up = linear_forward_orig_exl3(
            mlp_norm_out, linear_layers["up_proj"]["weight"],
            linear_layers["up_proj"]["suh"], linear_layers["up_proj"]["svh"],
            linear_layers["up_proj"]["bias"], device=DEVICE)
        gate = torch.nn.functional.silu(gate)
        mlp_inter = gate * up
        mlp_out = linear_forward_orig_exl3(
            mlp_inter, linear_layers["down_proj"]["weight"],
            linear_layers["down_proj"]["suh"], linear_layers["down_proj"]["svh"],
            linear_layers["down_proj"]["bias"], device=DEVICE)

        output = mlp_out

        save_fixture(
            "transformer-block", case_num,
            {
                "model": MODEL_NAME,
                "layer": f"{prefix}",
                "case": [
                    "single_token_no_residual",
                    "seq_no_residual",
                    "single_token_with_residual",
                    "seq_with_residual",
                ][case_num],
                "with_residual": str(with_residual),
                "framework": "exl3",
                "backend": "exllamav3_cuda" if USE_CUDA else "pytorch",
            },
            {
                "input_hidden_states": input_hidden_states.cpu(),
                "residual": residual.cpu() if residual is not None else None,
                "position_ids": position_ids.cpu(),
                "output": output.cpu(),
                "output_residual": output_residual.cpu(),
            },
        )

    print(f"  Generated block fixtures (4 cases, backend={'cuda' if USE_CUDA else 'cpu'})")


# ─── Generators registry (for --only) ────────────────────────────────

GENERATORS = {
    "linear": lambda tensors, config: generate_linear_fixtures(tensors, config),
    "attn":   lambda tensors, config: generate_attn_fixtures(tensors, config),
    "block":  lambda tensors, config: generate_block_fixtures(tensors, config),
}


# ─── Main ────────────────────────────────────────────────────────────

def main():
    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "--only":
        if len(sys.argv) < 3:
            print(f"Usage: python {sys.argv[0]} [--only linear|attn|block]")
            sys.exit(1)
        only = sys.argv[2]
        if only not in GENERATORS:
            print(f"Unknown: {only}. Available: {', '.join(sorted(GENERATORS))}")
            sys.exit(1)

    print(f"Generating {MODEL_NAME} layer {LAYER_IDX} EXL3 fixtures")
    print(f"  Backend: {'exllamav3 CUDA' if USE_CUDA else 'PyTorch reimpl'}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    ensure_fixture_dir()

    config = load_config()
    print(f"Loading EXL3 tensors from {MODEL_PATH}...")
    tensors = get_exl3_tensors(MODEL_PATH)

    if only is None:
        print("\nGenerating linear layer fixtures...")
        generate_linear_fixtures(tensors, config)
        print("\nGenerating attention fixtures...")
        generate_attn_fixtures(tensors, config)
        print("\nGenerating transformer block fixtures...")
        generate_block_fixtures(tensors, config)
    else:
        GENERATORS[only](tensors, config)

    print("=" * 60)
    print(f"Fixture generation complete!")
    print(f"Fixtures saved to: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
