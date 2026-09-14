#!/usr/bin/env python3
"""
Generate EXL3 layer-internal fixtures for the Nim layer-internals suite.

1. load the EXL3-quantized model (trellis, suh, svh per linear layer)
2. reconstruct the weights through ``quant_utils/exl3_utils.py``
3. run the EXL3 linear forward (Hadamard + GEMM + Hadamard) on CUDA
4. run the attention and transformer block forward, the long residual
   stream pattern of the suite
5. save per-layer fixtures for Nim testing (t_exl3_qwen3_01_layer_internals.nim)

All weights come from the EXL3 model file, the payloads carry
only the suite-read driving tensors, the 004 stats frames keep
the full recorded tensor set (the fingerprint surface of the dieted family).

Determinism contract:

- each generator seeds through torch.manual_seed with its own constant
- CUDA determinism flags, cudnn.deterministic=True, cudnn.benchmark=False
- fixture files reproduce exactly across separate process invocations

Run with python tests/testgen/gen_exl3_qwen3_01_layer_internals.py [--only {linear,attn,block}]
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

# ── Script dir plus the tests/ tree on the import path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

from fixture_stats import (  # path insert precedes the import (noqa E402)
    write_stats_file, write_text_zst)
from quant_utils.exl3_utils import (  # path insert precedes the import (noqa E402)
    get_exl3_tensors,
    get_in_features_out_features,
    derive_K,
    derive_cb,
    load_config,
    reconstruct_orig_exl3,
    reconstruct_reimpl_exl3,
    linear_forward_orig_exl3,
    linear_forward_reimpl_exl3,
    had_r_128_orig_exl3,
    rms_norm_orig_exl3,
    precompute_freqs_cis_reimpl_exl3,
    apply_rotary_pos_emb_reimpl_exl3,
)

# ─── Try CUDA backend ───
# Dispatch shape, matching the 03-chain generator:
# - exllamav3_cuda when exllamav3_ext imports and a CUDA device exists
# - pytorch otherwise, the pure-torch fallback
# Ext import presence is the real availability check, the wrapper
# functions import lazily and fail only at the first reconstruct call.
USE_CUDA: bool = False
try:
    from exllamav3.ext import exllamav3_ext as _ext  # noqa: F401
    USE_CUDA = torch.cuda.is_available()
    print(f"  [OK] exllamav3 CUDA extension loaded")
except Exception as e:
    print(f"  [WARN] exllamav3 CUDA extension not available: {e}")
    print(f"  [WARN] Falling back to PyTorch reimpl")

def linear_forward(x: torch.Tensor, weight: torch.Tensor,
                   suh: torch.Tensor, svh: torch.Tensor,
                   bias: torch.Tensor | None = None) -> torch.Tensor:
    """One EXL3 linear pass on the selected backend, Hadamard in, GEMM, Hadamard out."""
    if USE_CUDA:
        return linear_forward_orig_exl3(x, weight, suh, svh, bias, device=DEVICE)
    return linear_forward_reimpl_exl3(x, weight, suh, svh, bias, device=DEVICE)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm on the selected backend, fp32 accumulation, fp16 output."""
    if USE_CUDA:
        return rms_norm_orig_exl3(x, weight, eps)
    orig_shape = x.shape
    x32 = x.reshape(-1, orig_shape[-1]).to(torch.float32)
    w32 = weight.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    y = x32 * torch.rsqrt(variance + eps) * w32
    return y.to(x.dtype).reshape(orig_shape)


# ─── Determinism ──────────────────────────────────────────────────────
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── Config ───────────────────────────────────────────────────────────
MODEL_NAME = "Qwen3-0.6B-EXL3-5bpw"
LAYER_IDX = 0  # Test layer 0 (first layer, simplest residual state)
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "exl3-01-layer-internals", f"{MODEL_NAME}-layer-{LAYER_IDX}"
)
MODEL_DIR = os.path.join(GRANDPARENT_DIR, "hf_models", MODEL_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")

# Per-generator seeds, independent and order-agnostic.
SEED_LINEAR = 42
SEED_BLOCK = 43
SEED_ATTN = 44

# EXL3 operates in float16 on CUDA
DTYPE = torch.float16
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print(f"  [info] Using device: {DEVICE}")


# ─── Helpers ──────────────────────────────────────────────────────────


def ensure_fixture_dir() -> None:
    """Creates FIXTURE_DIR when absent."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)


def save_fixture(layer_name: str, case_num: int, metadata: dict,
                 tensors: dict, payload: dict) -> str:
    """Save one fixture payload with a separate metadata frame.

    Args:
    - layer_name, case_num, naming the output file `<layer>-<model>-<case>.safetensor`
    - metadata, the descriptor dict written beside the payload
    - tensors, the full recorded tensor set, the uniform stats frame
      covers every float-dtype tensor of it so the dieted-out tensors
      keep a distribution check
    - payload, the suite-read driving tensors saved into the safetensor file

    Returns the fixture file path.
    """
    filename = f"{layer_name}-{MODEL_NAME}-{case_num:02d}.safetensor"
    filepath = os.path.join(FIXTURE_DIR, filename)

    sorted_payload = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in sorted(payload.items())
        if tensor is not None
    )
    st_save_file(sorted_payload, filepath)

    metadata_path = filepath + ".metadata.json.zst"
    write_text_zst(metadata_path,
                   json.dumps(metadata, sort_keys=True, indent=2)
                   .encode("utf-8") + b"\n")

    entries = [(name, tensor) for name, tensor in sorted(tensors.items())
               if tensor.dtype.is_floating_point]
    write_stats_file(filepath + ".stats.json.zst",
                     os.path.basename(filepath), entries)

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
        weight = w.contiguous()  # [in_features, out_features] for ext.hgemm
    else:
        w = reconstruct_reimpl_exl3(trellis, K, cb, (in_f, out_f))
        weight = w.t().contiguous()  # [out_features, in_features] for F.linear

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
            (2, 4),   # 00 has batch=2, seq=4 (flattened)
            (1, 1),   # 01 has a single token
            (1, 8),   # 02 has a short sequence
            (2, 4),   # 03 has zeros (same shape as 00)
        ]

        for case_num, (batch, seq) in enumerate(test_shapes):
            total = batch * seq

            if case_num == 3:  # all-zero input case
                x = torch.zeros(total, in_f, dtype=DTYPE, device=DEVICE)
            else:
                x = torch.randn(total, in_f, dtype=DTYPE, device=DEVICE)

            y = linear_forward(
                x, layer_info["weight"], layer_info["suh"], layer_info["svh"],
                layer_info["bias"]
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
                {
                    # the suite-read driving input, the output
                    # stays on the 004 stats frame only
                    "input": x.cpu(),
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

    q_norm_weight_t = norms.get(f"{prefix}.self_attn.q_norm.weight")
    k_norm_weight_t = norms.get(f"{prefix}.self_attn.k_norm.weight")
    q_norm_weight = q_norm_weight_t.to(DEVICE) if q_norm_weight_t is not None else None
    k_norm_weight = k_norm_weight_t.to(DEVICE) if k_norm_weight_t is not None else None

    for case_num, (batch, seq) in enumerate([(1, 1), (2, 8)]):
        print(f"  Generating attn case {case_num}: batch={batch}, seq={seq}")
        hidden_states = torch.randn(batch, seq, hidden_size, dtype=DTYPE, device=DEVICE)
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0).expand(batch, -1).contiguous()

        q = linear_forward(
            hidden_states, linear_layers["q_proj"]["weight"],
                linear_layers["q_proj"]["suh"], linear_layers["q_proj"]["svh"],
                linear_layers["q_proj"]["bias"])
        k = linear_forward(
            hidden_states, linear_layers["k_proj"]["weight"],
                linear_layers["k_proj"]["suh"], linear_layers["k_proj"]["svh"],
                linear_layers["k_proj"]["bias"])
        v = linear_forward(
            hidden_states, linear_layers["v_proj"]["weight"],
                linear_layers["v_proj"]["suh"], linear_layers["v_proj"]["svh"],
                linear_layers["v_proj"]["bias"])

        q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

        if q_norm_weight is not None:
            q = rms_norm(q, q_norm_weight, rms_eps)
        if k_norm_weight is not None:
            k = rms_norm(k, k_norm_weight, rms_eps)

        q, k = apply_rotary_pos_emb_reimpl_exl3(q, k, cos, sin, position_ids)
        q = q.to(DTYPE)
        k = k.to(DTYPE)

        # GQA path, repeat K and V to match Q heads before SDPA
        if num_kv_heads < num_heads:
            n_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            scale=head_dim ** -0.5)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, num_heads * head_dim)

        out = linear_forward(
            attn_out, linear_layers["o_proj"]["weight"],
                linear_layers["o_proj"]["suh"], linear_layers["o_proj"]["svh"],
                linear_layers["o_proj"]["bias"])

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
        }, {
            # the suite-read driving tensors, the rope tables and the output
            # stay on the 004 stats frame only
            "hidden_states": hidden_states.cpu(),
            "position_ids": position_ids.cpu(),
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
    input_ln_weight_t = norms.get(f"{prefix}.input_layernorm.weight")
    post_attn_ln_weight_t = norms.get(f"{prefix}.post_attention_layernorm.weight")
    q_norm_weight_t = norms.get(f"{prefix}.self_attn.q_norm.weight")
    k_norm_weight_t = norms.get(f"{prefix}.self_attn.k_norm.weight")
    input_ln_weight = input_ln_weight_t.to(DEVICE) if input_ln_weight_t is not None else None
    post_attn_ln_weight = post_attn_ln_weight_t.to(DEVICE) if post_attn_ln_weight_t is not None else None
    q_norm_weight = q_norm_weight_t.to(DEVICE) if q_norm_weight_t is not None else None
    k_norm_weight = k_norm_weight_t.to(DEVICE) if k_norm_weight_t is not None else None

    if input_ln_weight is None or post_attn_ln_weight is None:
        print("  Warning: norm weights not available, skipping block fixtures")
        return

    # Test cases carry (batch, seq, with_residual)
    test_cases = [
        (1, 1, False),  # 00, single token, no residual (first block, decode)
        (2, 8, False),  # 01, short sequence, no residual (first block, prefill)
        (1, 1, True),   # 02, single token, with residual (middle block, decode)
        (2, 8, True),   # 03, short sequence, with residual (middle block, prefill)
    ]

    for case_num, (batch, seq, with_residual) in enumerate(test_cases):
        print(f"  Generating block case {case_num}: batch={batch}, seq={seq}, with_residual={with_residual}")

        # Input hidden states
        input_hidden_states = torch.randn(batch, seq, hidden_size, dtype=DTYPE, device=DEVICE)
        residual = torch.randn(batch, seq, hidden_size, dtype=DTYPE, device=DEVICE) if with_residual else None
        position_ids = torch.arange(seq, device=DEVICE).unsqueeze(0).expand(batch, -1).contiguous()

        # Long residual stream pattern (matches Nim TransformerBlock)
        if residual is None:
            residual = input_hidden_states.clone()

        # Step 2 runs attn_norm.forward_with_residual(x, residual)
        attn_norm_input = input_hidden_states + residual
        attn_norm_out = rms_norm(attn_norm_input, input_ln_weight, rms_eps)
        r_after_attn_norm = attn_norm_input

        # Step 3, the attention forward with EXL3 linear layers
        q = linear_forward(
            attn_norm_out, linear_layers["q_proj"]["weight"],
                linear_layers["q_proj"]["suh"], linear_layers["q_proj"]["svh"],
                linear_layers["q_proj"]["bias"])
        k = linear_forward(
            attn_norm_out, linear_layers["k_proj"]["weight"],
                linear_layers["k_proj"]["suh"], linear_layers["k_proj"]["svh"],
                linear_layers["k_proj"]["bias"])
        v = linear_forward(
            attn_norm_out, linear_layers["v_proj"]["weight"],
                linear_layers["v_proj"]["suh"], linear_layers["v_proj"]["svh"],
                linear_layers["v_proj"]["bias"])

        # Reshape to multi-head format
        q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

        # QK norm application
        if q_norm_weight is not None:
            q = rms_norm(q, q_norm_weight, rms_eps)
        if k_norm_weight is not None:
            k = rms_norm(k, k_norm_weight, rms_eps)

        # RoPE application step
        q, k = apply_rotary_pos_emb_reimpl_exl3(q, k, cos, sin, position_ids)
        q = q.to(DTYPE)
        k = k.to(DTYPE)

        # GQA path, repeat K and V to match Q heads before SDPA
        if num_kv_heads < num_heads:
            n_repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            scale=head_dim ** -0.5)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq, num_heads * head_dim)

        # O projection pass
        attn_output = linear_forward(
            attn_output, linear_layers["o_proj"]["weight"],
                linear_layers["o_proj"]["suh"], linear_layers["o_proj"]["svh"],
                linear_layers["o_proj"]["bias"])

        # Long residual, h = attn_norm_input + attn_output
        h = r_after_attn_norm + attn_output

        # Step 4 runs mlp_norm.forward_with_residual(h, r)
        mlp_norm_out = rms_norm(h, post_attn_ln_weight, rms_eps)
        output_residual = h

        # Step 5, the MLP forward with EXL3 linear layers
        gate = linear_forward(
            mlp_norm_out, linear_layers["gate_proj"]["weight"],
                linear_layers["gate_proj"]["suh"], linear_layers["gate_proj"]["svh"],
                linear_layers["gate_proj"]["bias"])
        up = linear_forward(
            mlp_norm_out, linear_layers["up_proj"]["weight"],
                linear_layers["up_proj"]["suh"], linear_layers["up_proj"]["svh"],
                linear_layers["up_proj"]["bias"])
        gate = torch.nn.functional.silu(gate)
        mlp_inter = gate * up
        mlp_out = linear_forward(
            mlp_inter, linear_layers["down_proj"]["weight"],
                linear_layers["down_proj"]["suh"], linear_layers["down_proj"]["svh"],
                linear_layers["down_proj"]["bias"])

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
            {
                # the suite-read driving tensors, the two
                # outputs stay on the 004 stats frame only
                "input_hidden_states": input_hidden_states.cpu(),
                "residual": residual.cpu() if residual is not None else None,
                "position_ids": position_ids.cpu(),
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
    """Parse the --only switch and run the selected fixture generator."""
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
