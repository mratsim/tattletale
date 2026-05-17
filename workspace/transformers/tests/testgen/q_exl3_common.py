"""
q_exl3_common — Shared logic for EXL3 fixture generation.

Suffix conventions:
    _orig_exl3      = thin shim around EXL3 production kernel (ext.reconstruct)
    _reimpl_exl3    = our reimplementation of EXL3-internal behavior.
                      Risk: decoder / forward may be wrong if the reimpl diverges.
    (no suffix)     = pure utility (loading, parsing, shape derivation).

## On LOP3

The LOP3 codebook decode formula (``(x & M1) ^ M2`` instead of the PTX-spec
``a ^ (b & c)``) is due to a PTX ↔ SASS index reversal in inline asm on sm_120.
See:
- https://github.com/turboderp-org/exllamav3/blob/v0.0.34/exllamav3/exllamav3_ext/quant/codebook.cuh#L28-L44
- workspace/positron/docs/dev/cuda_lop3.md
- https://github.com/Cornell-RelaxML/qtip/blob/main/lib/codebook/bitshift.py
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file as st_save_file

# ────────────────────────────────────────────────────────────────────
#  Utility functions (no suffix)
# ────────────────────────────────────────────────────────────────────


def load_config(model_dir: str | None = None) -> dict:
    """Load model config.json from the EXL3 model directory."""
    if model_dir is None:
        from os.path import dirname as _d
        _SCRIPT_DIR = _d(__file__)
        BASE_DIR = _d(_SCRIPT_DIR)  # tests/
        model_dir = os.path.join(BASE_DIR, "hf_models", "Qwen3-0.6B-EXL3-5bpw")
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def get_exl3_tensors(model_path: str) -> dict:
    """Load all tensors from safetensors, grouped by layer.

    Returns:
        {
            "model.layers.0.self_attn.q_proj": {
                "trellis": Tensor, "suh": Tensor, "svh": Tensor,
                "mcg": Tensor|None, "mul1": Tensor|None, "bias": Tensor|None,
            },
            ...
            "_norms": { "model.layers.0.input_layernorm.weight": Tensor, ... },
            "_embeddings": { ... },
            "_lm_head": { ... },
            "_others": { ... },
        }
    """
    result: dict = {}
    norms: dict = {}
    embeddings: dict = {}
    lm_head: dict = {}
    others: dict = {}

    with safe_open(model_path, framework="pt") as f:
        keys = list(f.keys())

        # Identify EXL3 layers by grouping tensors by common prefix
        exl3_layers: set = set()
        for k in keys:
            for suffix in (".trellis", ".suh", ".svh", ".mcg", ".mul1", ".bias"):
                if k.endswith(suffix):
                    exl3_layers.add(k[: -len(suffix)])

        for layer_key in sorted(exl3_layers):
            entry: dict = {}
            for suffix in ("trellis", "suh", "svh", "mcg", "mul1", "bias"):
                tensor_key = f"{layer_key}.{suffix}"
                if tensor_key in keys:
                    entry[suffix] = f.get_tensor(tensor_key)
                else:
                    entry[suffix] = None
            result[layer_key] = entry

        # Collect non-EXL3 tensors
        exl3_keys: set = set()
        for layer_key in exl3_layers:
            for suffix in ("trellis", "suh", "svh", "mcg", "mul1", "bias"):
                exl3_keys.add(f"{layer_key}.{suffix}")

        for k in keys:
            if k in exl3_keys:
                continue
            t = f.get_tensor(k)
            if "norm" in k.lower() or "layernorm" in k.lower():
                norms[k] = t
            elif "embed" in k.lower():
                embeddings[k] = t
            elif "lm_head" in k.lower():
                lm_head[k] = t
            else:
                others[k] = t

    result["_norms"] = norms
    result["_embeddings"] = embeddings
    result["_lm_head"] = lm_head
    result["_others"] = others
    return result


def parse_layer_name(layer_key: str) -> tuple | None:
    """Parse 'model.layers.8.self_attn.q_proj' → (8, 'self_attn', 'q_proj')."""
    parts = layer_key.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        try:
            layer_idx = int(parts[2])
        except ValueError:
            return None
        return layer_idx, parts[3], ".".join(parts[4:]) if len(parts) > 4 else ""
    return None


def get_in_features_out_features(layer_key: str, trellis: torch.Tensor,
                                 config: dict) -> tuple:
    """Determine in_features and out_features from trellis shape."""
    tiles_k, tiles_n, _ = trellis.shape
    return tiles_k * 16, tiles_n * 16


def derive_cb(layer_entry: dict) -> int:
    """Determine codebook variant from mcg/mul1 tensors."""
    if layer_entry.get("mcg") is not None:
        return 1
    if layer_entry.get("mul1") is not None:
        return 2
    return 0


def derive_K(trellis: torch.Tensor) -> int:
    """Derive bitrate K from trellis shape."""
    return trellis.shape[2] * 16 // 256


# ────────────────────────────────────────────────────────────────────
#  EXL3 production kernel shim (_orig_exl3)
#  These call into the compiled exllamav3_ext CUDA extension.
#  They require CUDA + a working exllamav3_ext JIT build.
# ────────────────────────────────────────────────────────────────────


def reconstruct_orig_exl3(trellis: torch.Tensor, K: int, mcg: bool, mul1: bool,
                          out_shape: tuple) -> torch.Tensor:
    """Reconstruct weights using the production EXL3 CUDA kernel.

    Requires compiled ``exllamav3_ext``.
    Returns [in_features, out_features] float16 on CUDA.
    """
    from exllamav3.ext import exllamav3_ext as ext
    in_features, out_features = out_shape
    w = torch.empty((in_features, out_features), dtype=torch.half,
                    device=trellis.device)
    ext.reconstruct(w, trellis.contiguous(), K, mcg, mul1)
    return w


def had_r_128_orig_exl3(x: torch.Tensor,
                         scale: Optional[torch.Tensor] = None,
                         norm: float = 1.0) -> torch.Tensor:
    """Apply 128-block Walsh-Hadamard transform via production EXL3 CUDA kernel.

    Calls ``ext.had_r_128(input, output, scale, None, norm)``.
    Requires compiled ``exllamav3_ext``.
    Returns a new tensor.
    """
    from exllamav3.ext import exllamav3_ext as ext
    x = x.contiguous()
    out = torch.empty_like(x)
    ext.had_r_128(x, out, scale, None, norm)
    return out


def linear_forward_orig_exl3(x: torch.Tensor, weight: torch.Tensor,
                               suh: torch.Tensor, svh: torch.Tensor,
                               bias: Optional[torch.Tensor] = None,
                               device: torch.device = None) -> torch.Tensor:
    """Forward pass for an EXL3 linear layer using production CUDA kernels.

    Matches exactly what ``LinearEXL3.forward()`` does internally:
      Hadamard(input, suh, norm=1) -> hgemm -> Hadamard(output, svh, norm=1) -> bias

    **Weight layout**: ``weight`` must be ``[in_features, out_features]``
    (non-transposed, native output of ``reconstruct_orig_exl3``).
    Differs from ``F.linear`` which expects ``[out_features, in_features]``.

    Requires compiled ``exllamav3_ext``.
    """
    from exllamav3.ext import exllamav3_ext as ext
    if device is not None:
        x = x.to(device)
        weight = weight.to(device)
        suh = suh.to(device)
        svh = svh.to(device)
        if bias is not None:
            bias = bias.to(device)

    x = x.contiguous()
    weight = weight.contiguous()

    orig_shape = x.shape
    in_f = x.shape[-1]
    out_f = weight.shape[1]  # weight is [in_features, out_features]
    x_2d = x.reshape(-1, in_f)

    # Input Hadamard (adds suh scale, no extra norm — scale absorbs 1/sqrt(128))
    xh = torch.empty_like(x_2d)
    ext.had_r_128(x_2d, xh, suh, None, 1.0)

    # GEMM via ext.hgemm (weight is [in_features, out_features])
    y_2d = torch.empty(x_2d.shape[0], out_f, dtype=x.dtype, device=x.device)
    ext.hgemm(xh, weight, y_2d)

    # Output Hadamard (adds svh scale)
    ext.had_r_128(y_2d, y_2d, None, svh, 1.0)

    y = y_2d.reshape(*orig_shape[:-1], out_f)

    if bias is not None:
        y = y + bias
    return y


def rms_norm_orig_exl3(x: torch.Tensor, weight: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """RMS norm using the production EXL3 CUDA kernel.

    Wraps ``ext.rms_norm(x, w, y, eps, 0.0, 1.0, False, False)``.
    Requires compiled ``exllamav3_ext``.
    """
    from exllamav3.ext import exllamav3_ext as ext
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    y = torch.empty_like(x_2d)
    ext.rms_norm(x_2d, weight, y, eps, 0.0, 1.0, False, False)
    return y.reshape(orig_shape)


def precompute_freqs_cis_reimpl_exl3(head_dim: int, max_position: int,
                                      theta: float = 1000000.0) -> tuple:
    """Precompute cos/sin matching exllamav3's RoPE convention (float32, NEOX cat).

    Uses float32 and ``cat([half, half], -1)`` for NEOX duplication, matching
    exllamav3's ``RoPE.compute_sincos()``.

    Returns (cos, sin) each [max_position, head_dim] in float16.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos.to(torch.float16), sin.to(torch.float16)


def apply_rotary_pos_emb_reimpl_exl3(q: torch.Tensor, k: torch.Tensor,
                                       cos: torch.Tensor, sin: torch.Tensor,
                                       position_ids: torch.Tensor) -> tuple:
    """Apply RoPE using the standard NEOX formula.

    cos/sin: [max_seq_len, head_dim] precomputed table.
    q/k: [batch, num_heads, seq, head_dim] (head-major).
    """
    cos_sliced = cos[position_ids]
    sin_sliced = sin[position_ids]
    cos_sliced = cos_sliced.unsqueeze(1)
    sin_sliced = sin_sliced.unsqueeze(1)
    x1 = q[..., :q.shape[-1] // 2]
    x2 = q[..., q.shape[-1] // 2:]
    q_rot = q * cos_sliced + torch.cat((-x2, x1), dim=-1) * sin_sliced
    x1 = k[..., :k.shape[-1] // 2]
    x2 = k[..., k.shape[-1] // 2:]
    k_rot = k * cos_sliced + torch.cat((-x2, x1), dim=-1) * sin_sliced
    return q_rot, k_rot

# ────────────────────────────────────────────────────────────────────
#  EXL3 reimplementations (_reimpl_exl3)
# ────────────────────────────────────────────────────────────────────


def _funnel_shift_reimpl_exl3(b: torch.Tensor, a: torch.Tensor,
                               shift: torch.Tensor) -> torch.Tensor:
    """Funnel shift: extract 16-bit window from 64-bit ``(a<<32|b)`` pair.

    All tensors uint32, shift is int tensor. Returns uint16 tensor.
    """
    a64 = a.to(torch.int64)
    b64 = b.to(torch.int64)
    shift64 = shift.to(torch.int64)
    merged = (a64 << 32) | b64
    shifted = (merged >> shift64) & 0xFFFF
    return shifted.to(torch.uint16)


def decode_codebook_reimpl_exl3(words: torch.Tensor, cb: int) -> torch.Tensor:
    """Decode packed uint16 words to float16 values.

    Uses int32 throughout to match CUDA uint32 overflow semantics.

    **LOP3 note**: The CUDA source says ``lop3.b32 ... 0x6a`` but inline-asm
    on sm_120 executes LUT 0x78 (PTX ↔ SASS index reversal).  The correct
    formula ``(x & M1) ^ M2`` was verified empirically against the GPU kernel.
    See ``workspace/positron/docs/dev/cuda_lop3.md`` for details.

    Args:
        words: [..., N] uint16 tensor of packed indices.
        cb: Codebook variant (0, 1, or 2).

    Returns:
        [..., N] float16 tensor of decoded values.
    """
    x = words.to(torch.int32)  # promote to 32-bit (matches CUDA uint32 overflow)

    if cb == 0:
        x = x * 89226354 + 64248484
    elif cb == 1:
        x = x * 0xCBAC1FED
    elif cb == 2:
        x = x * 0x83DCD12D
        x = x & 0xFFFFFFFF
        b0 = (x >> 0) & 0xFF
        b1 = (x >> 8) & 0xFF
        b2 = (x >> 16) & 0xFF
        b3 = (x >> 24) & 0xFF
        x = b0 + b1 + b2 + b3 + 0x6400
        k_inv = torch.tensor(0.00677, dtype=torch.float16, device=x.device)
        k_bias = torch.tensor(-10.39, dtype=torch.float16, device=x.device)
        return x.to(torch.float16) * k_inv + k_bias
    else:
        raise ValueError(f"Unknown codebook: {cb}")

    # LOP3: Blackwell sm_120 executes 0x6a source as LUT 0x78.
    # Verified empirically with isolated GPU kernel.
    # Formula: result = (x & M1) ^ M2
    x = (x & 0x8fff8fff) ^ 0x3b603b60

    # Reinterpret lower/upper 16 bits as float16 and sum
    lo = (x & 0xFFFF).to(torch.uint16).view(torch.float16)
    hi = ((x >> 16) & 0xFFFF).to(torch.uint16).view(torch.float16)
    return lo + hi


def _shuffle_tiles_batch_reimpl_exl3(decoded_flat: torch.Tensor,
                                      tiles_k: int, tiles_n: int) -> torch.Tensor:
    """Apply the CUDA kernel's tensor-core → row-major tile shuffle (batched).

    Inverse permutation: for output position j, which input element?
    """
    _inv = torch.tensor([
         0,  32,  64,  96, 128, 160, 192, 224,   4,  36,  68, 100, 132, 164, 196, 228,
         1,  33,  65,  97, 129, 161, 193, 225,   5,  37,  69, 101, 133, 165, 197, 229,
         8,  40,  72, 104, 136, 168, 200, 232,  12,  44,  76, 108, 140, 172, 204, 236,
         9,  41,  73, 105, 137, 169, 201, 233,  13,  45,  77, 109, 141, 173, 205, 237,
        16,  48,  80, 112, 144, 176, 208, 240,  20,  52,  84, 116, 148, 180, 212, 244,
        17,  49,  81, 113, 145, 177, 209, 241,  21,  53,  85, 117, 149, 181, 213, 245,
        24,  56,  88, 120, 152, 184, 216, 248,  28,  60,  92, 124, 156, 188, 220, 252,
        25,  57,  89, 121, 153, 185, 217, 249,  29,  61,  93, 125, 157, 189, 221, 253,
         2,  34,  66,  98, 130, 162, 194, 226,   6,  38,  70, 102, 134, 166, 198, 230,
         3,  35,  67,  99, 131, 163, 195, 227,   7,  39,  71, 103, 135, 167, 199, 231,
        10,  42,  74, 106, 138, 170, 202, 234,  14,  46,  78, 110, 142, 174, 206, 238,
        11,  43,  75, 107, 139, 171, 203, 235,  15,  47,  79, 111, 143, 175, 207, 239,
        18,  50,  82, 114, 146, 178, 210, 242,  22,  54,  86, 118, 150, 182, 214, 246,
        19,  51,  83, 115, 147, 179, 211, 243,  23,  55,  87, 119, 151, 183, 215, 247,
        26,  58,  90, 122, 154, 186, 218, 250,  30,  62,  94, 126, 158, 190, 222, 254,
        27,  59,  91, 123, 155, 187, 219, 251,  31,  63,  95, 127, 159, 191, 223, 255,
    ], dtype=torch.int64, device=decoded_flat.device)
    shuffled = decoded_flat[:, _inv]
    return shuffled.view(tiles_k, tiles_n, 16, 16)


def reconstruct_reimpl_exl3(trellis: torch.Tensor, K: int, cb: int,
                             out_shape: tuple) -> torch.Tensor:
    """Reconstruct weights using pure-PyTorch tensor ops.

    Args:
        trellis: [tiles_k, tiles_n, 256*K//16] int16 packed tensor.
        K: Bitrate.
        cb: Codebook variant (0, 1, 2).
        out_shape: (in_features, out_features).

    Returns:
        [in_features, out_features] float16 weight matrix.
    """
    device = trellis.device
    tiles_k, tiles_n, _ = trellis.shape
    in_features, out_features = out_shape

    tile_size = 256
    packed_words = tile_size * K // 32  # in uint32s

    offsets = torch.arange(tile_size, device=device)

    # b0 = t*K + K - 16 + 256*K  (+256*K ensures positivity for %)
    b0 = offsets * K + K - 16 + tile_size * K
    b1 = b0 + 16
    i0 = b0 // 32
    i1 = (b1 - 1) // 32
    s0 = (i1 + 1) * 32 - b1

    i0 = i0 % packed_words
    i1 = i1 % packed_words

    # Pair consecutive int16 values into uint32 (matches CUDA kernel)
    u16 = trellis.to(torch.uint16)
    packed = u16[:, :, 0::2].to(torch.int64) | (u16[:, :, 1::2].to(torch.int64) << 16)

    i0_exp = i0.unsqueeze(0).unsqueeze(0).expand(tiles_k, tiles_n, -1)
    i1_exp = i1.unsqueeze(0).unsqueeze(0).expand(tiles_k, tiles_n, -1)
    a = packed.gather(2, i0_exp.to(torch.int64)) & 0xFFFFFFFF
    b = packed.gather(2, i1_exp.to(torch.int64)) & 0xFFFFFFFF

    s0_exp = s0.unsqueeze(0).unsqueeze(0).expand(tiles_k, tiles_n, -1)
    words = _funnel_shift_reimpl_exl3(b, a, s0_exp)

    vals = decode_codebook_reimpl_exl3(words, cb)

    total_tiles = tiles_k * tiles_n
    vals_flat = vals.reshape(total_tiles, -1)
    w = _shuffle_tiles_batch_reimpl_exl3(vals_flat, tiles_k, tiles_n)

    w = w.permute(0, 2, 1, 3).reshape(in_features, out_features)
    return w


def fwht_128_reimpl_exl3(x: torch.Tensor) -> torch.Tensor:
    """In-place fast Walsh-Hadamard transform, length 128 on last dim."""
    n = 128
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                a = x[..., j].clone()
                b = x[..., j + step].clone()
                x[..., j] = a + b
                x[..., j + step] = a - b
        step *= 2
    return x


def had_r_128_reimpl_exl3(x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                           norm: float = 1.0,
                           pre_fwht: bool = True) -> torch.Tensor:
    """Apply 128-block Walsh-Hadamard transform on last dimension.

    Matches ``ext.had_r_128(input, output, scale_pre, scale_post, norm)``.
    CUDA kernel: ``output = FWHT(input * scale_pre) * scale_post / sqrt(128)``.
    - Input Hadamard: scale=suh, pre_fwht=True (scale BEFORE FWHT)
    - Output Hadamard: scale=svh, pre_fwht=False (scale AFTER FWHT)

    Args:
        x: [batch, dim] where dim is multiple of 128.
        scale: [dim] optional element-wise scale.
        norm: Post-transform normalization factor.
        pre_fwht: If True, apply scale before FWHT; if False, after.
    """
    x = x.clone()
    dim = x.shape[-1]
    with torch.no_grad():
        for blk_start in range(0, dim, 128):
            blk_slice = x[..., blk_start:blk_start + 128]
            blk_f32 = blk_slice.float()
            if scale is not None and pre_fwht:
                s = scale[blk_start:blk_start + 128]
                blk_f32 *= s.float()
            fwht_128_reimpl_exl3(blk_f32)
            if scale is not None and not pre_fwht:
                s = scale[blk_start:blk_start + 128]
                blk_f32 *= s.float()
            blk_f32 *= norm
            blk_slice.copy_(blk_f32.to(x.dtype))
    return x
    return x


def linear_forward_reimpl_exl3(x: torch.Tensor, weight: torch.Tensor,
                                suh: torch.Tensor, svh: torch.Tensor,
                                bias: Optional[torch.Tensor] = None,
                                device: torch.device = None) -> torch.Tensor:
    """Forward pass for an EXL3 linear layer (PyTorch reimpl).

    Applies: Input Hadamard → GEMM → Output Hadamard → (bias).

    Args:
        x: [batch, in_features] float16 input.
        weight: [out_features, in_features] float16 (transposed for F.linear).
        suh: [in_features] float16 scale.
        svh: [out_features] float16 scale.
        bias: Optional [out_features] float16 bias.

    Returns:
        [batch, out_features] float16 output.
    """
    if device is not None:
        x = x.to(device)
        weight = weight.to(device)
        suh = suh.to(device)
        svh = svh.to(device)
        if bias is not None:
            bias = bias.to(device)

    _norm = 0.088388347648  # 1/sqrt(128) — CUDA kernel always applies this internally
    xh = had_r_128_reimpl_exl3(x, suh, _norm)  # Input Hadamard: pre_fwht=True
    y = torch.nn.functional.linear(xh, weight)
    y = had_r_128_reimpl_exl3(y, svh, _norm, pre_fwht=False)  # Output Hadamard

    if bias is not None:
        y = y + bias
    return y


def dequant_reimpl_exl3(w_reconstructed: torch.Tensor,
                         suh: torch.Tensor,
                         svh: torch.Tensor,
                         hadamard_dim: int = 128) -> torch.Tensor:
    """Recover the original FP16-domain weight from an EXL3-reconstructed weight.

    EXL3 forward is::

        y = svh * H @ W_trellis @ H * suh / sqrt(128) @ x

    So the original FP16 weight is approximately::

        W_fp16 ≈ svh * H @ W_reconstructed @ H * suh / sqrt(128)

    where ``H`` is the **unnormalized** Sylvester Hadamard matrix.

    Args:
        w_reconstructed: [in_features, out_features] float16 —
            output of ``reconstruct_reimpl_exl3``.
        suh: [in_features] float16 input scale.
        svh: [out_features] float16 output scale.
        hadamard_dim: Block size for the Hadamard transform (default 128).

    Returns:
        [in_features, out_features] float16 weight approximating the
        original FP16 linear weight.
    """
    k, n = w_reconstructed.shape
    HAD = hadamard_dim

    # Build unnormalized Sylvester Hadamard matrix H_{HAD}
    had = torch.ones(1, 1, dtype=torch.float)
    while had.shape[0] < HAD:
        had = torch.cat([torch.cat([had, had], 1),
                         torch.cat([had, -had], 1)], 0)

    had_norm = 1.0 / math.sqrt(HAD)  # 1/sqrt(128) = 0.088388347648
    had = had.to(w_reconstructed.device)
    w = w_reconstructed.float()

    # Build normalized Sylvester Hadamard H_norm = H / sqrt(HAD)
    had = torch.ones(1, 1, dtype=torch.float)
    while had.shape[0] < HAD:
        had = torch.cat([torch.cat([had, had], 1),
                         torch.cat([had, -had], 1)], 0)
    had = had.to(w_reconstructed.device)
    had_norm = had / math.sqrt(HAD)  # matches exl3 get_hadamard_dt

    w = w_reconstructed.float()

    # Left: H_norm @ W (block-wise on rows) — matching exl3 preapply_had_l
    for blk in range(0, k, HAD):
        w[blk:blk + HAD] = had_norm @ w[blk:blk + HAD]

    w *= suh.unsqueeze(1).float()  # suh row scale — matching exl3 order

    # Right: W @ H_norm (block-wise on cols) — matching exl3 preapply_had_r
    for blk in range(0, n, HAD):
        w[:, blk:blk + HAD] = w[:, blk:blk + HAD] @ had_norm

    w *= svh.unsqueeze(0).float()  # svh column scale — matching exl3 order

    return w.half()
