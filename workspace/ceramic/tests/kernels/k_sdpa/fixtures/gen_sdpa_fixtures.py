#!/usr/bin/env python3
"""SDPA fixture generator: python + numpy, libtorch-free, raw binary.

One directory per fixture under this script's directory:

- meta.json with name, dtype, shape fields, scale, causal, mask kind,
  and the float64 checksum of the expected output
- q.bin / k.bin / v.bin, the inputs, raw little-endian, the meta dtype,
  contiguous row-major over (B, H, S, D)
- mask.bin, the additive mask, raw float32 over (B, Hm, S, S)
- o.bin, the expected output, raw float32

The reference runs in float64 end to end for a tight truth.

- bf16 inputs are float32 values rounded to bf16 bits with
  round-to-nearest-even, stored as raw uint16
- fixtures keep at least one unmasked key per query row
- the fully-masked-row NaN contract stays documented, it is not
  fixture-tested

Run:  python3 gen_sdpa_fixtures.py [name ...]   (default: all fixtures)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def f32_to_bf16_bits(x):
    """float32 array -> uint16 bf16 bits, round-to-nearest-even."""
    u = x.astype(np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def sdpa_reference(q, k, v, mask, causal, scale):
    """float64 reference: o = softmax(scale * q @ k^T + mask) @ v."""
    q = q.astype(np.float64)
    k = k.astype(np.float64)
    v = v.astype(np.float64)
    B, Hq, L, Dk = q.shape
    _, Hkv, S, _ = k.shape
    Dv = v.shape[3]
    g = Hq // Hkv
    # GQA: kv head h // g serves the query heads [hkv*g, (hkv+1)*g).
    # Reshaping the flat head axis as (Hkv, g) pairs h = hkv*g + g_idx
    # with kv head hkv, the transpose brings the group axis next to batch.
    q5 = q.reshape(B, Hkv, g, L, Dk).transpose(0, 2, 1, 3, 4)       # (B, g, Hkv, L, Dk)
    k5 = np.broadcast_to(k.reshape(B, 1, Hkv, S, Dk), (B, g, Hkv, S, Dk))
    v5 = np.broadcast_to(v.reshape(B, 1, Hkv, S, Dv), (B, g, Hkv, S, Dv))
    s = np.einsum("bghqd,bghkd->bghqk", q5, k5) * scale
    if causal:
        qIdx = np.arange(L).reshape(1, 1, 1, L, 1)
        kIdx = np.arange(S).reshape(1, 1, 1, 1, S)
        s = np.where(kIdx > qIdx, -np.inf, s)
    if mask is not None:
        m = mask.astype(np.float64)  # (B, Hm, S, S), Hm in {1, Hkv}
        m = m.reshape(B, 1, m.shape[1], S, S)
        if m.shape[2] == 1 and Hkv > 1:
            m = np.broadcast_to(m, (B, 1, Hkv, S, S))
        m = np.broadcast_to(m, (B, g, Hkv, S, S))
        s = s + m
    mMax = s.max(axis=-1, keepdims=True)
    p = np.exp(s - mMax)
    l = p.sum(axis=-1, keepdims=True)
    o5 = np.einsum("bghqk,bghkd->bghqd", p, v5) / l
    return o5.transpose(0, 2, 1, 3, 4).reshape(B, Hq, L, Dv)


def write_bin(path, arr):
    arr.tofile(path)


def make_fixture(name, dtype, B, Hq, Hkv, L, S, Dk, Dv, scale,
                 causal=False, mask_kind="none", mask_heads=0, seed=0):
    """Generates one fixture directory, prints its shape line."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, Hq, L, Dk)).astype(np.float32)
    k = rng.standard_normal((B, Hkv, S, Dk)).astype(np.float32)
    v = rng.standard_normal((B, Hkv, S, Dv)).astype(np.float32)

    mask = None
    if mask_kind == "additive":
        # (B, Hm, S, S) with Hm in {1, Hkv}. A few -inf blocks, each
        # query row keeps at least one unmasked key
        Hm = mask_heads if mask_heads else Hkv
        mask = np.where(rng.standard_normal((B, Hm, S, S)) > 1.5,
                        np.float32(-np.inf), np.float32(0.0))
        # guarantee one open key per row: clear the diagonal
        diag = np.arange(S)
        mask[:, :, diag, diag] = 0.0

    o = sdpa_reference(q, k, v, mask, causal, scale)

    out = os.path.join(HERE, name)
    os.makedirs(out, exist_ok=True)
    meta = {
        "name": name, "dtype": dtype,
        "B": B, "Hq": Hq, "Hkv": Hkv, "L": L, "S": S, "Dk": Dk, "Dv": Dv,
        "scale": scale, "causal": causal, "mask": mask_kind,
        "checksum": float(o.sum()),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    qOut, kOut, vOut = q, k, v
    if dtype == "float16":
        qOut, kOut, vOut = q.astype(np.float16), k.astype(np.float16), v.astype(np.float16)
    elif dtype == "bfloat16":
        qOut, kOut, vOut = f32_to_bf16_bits(q), f32_to_bf16_bits(k), f32_to_bf16_bits(v)
    write_bin(os.path.join(out, "q.bin"), qOut)
    write_bin(os.path.join(out, "k.bin"), kOut)
    write_bin(os.path.join(out, "v.bin"), vOut)
    if mask is not None:
        write_bin(os.path.join(out, "mask.bin"), mask.astype(np.float32))
    write_bin(os.path.join(out, "o.bin"), o.astype(np.float32))
    print(f"{name}: dtype={dtype} B={B} Hq={Hq} Hkv={Hkv} L={L} S={S} "
          f"Dk={Dk} Dv={Dv} scale={scale} causal={causal} "
          f"mask={mask_kind} checksum={o.sum():.6f}")


FIXTURES = [
    dict(name="tiny_f32", dtype="float32", B=1, Hq=1, Hkv=1, L=32, S=32, Dk=16, Dv=16, scale=None, seed=1),
    dict(name="gqa2_f32", dtype="float32", B=1, Hq=2, Hkv=1, L=32, S=32, Dk=32, Dv=32, scale=None, seed=2),
    dict(name="batched_gqa_f32", dtype="float32", B=2, Hq=4, Hkv=2, L=64, S=64, Dk=32, Dv=32, scale=None, seed=3),
    dict(name="causal_f32", dtype="float32", B=1, Hq=1, Hkv=1, L=64, S=64, Dk=32, Dv=32, scale=None, causal=True, seed=4),
    dict(name="mask_f32", dtype="float32", B=1, Hq=2, Hkv=2, L=48, S=48, Dk=32, Dv=32, scale=None, mask_kind="additive", seed=5),
    dict(name="causal_mask_gqa4_f32", dtype="float32", B=1, Hq=4, Hkv=1, L=64, S=64, Dk=64, Dv=64, scale=None, causal=True, mask_kind="additive", seed=6),
    dict(name="tiny_f16", dtype="float16", B=1, Hq=1, Hkv=1, L=32, S=32, Dk=16, Dv=16, scale=None, seed=7),
    dict(name="causal_gqa2_f16", dtype="float16", B=1, Hq=2, Hkv=1, L=64, S=64, Dk=32, Dv=32, scale=None, causal=True, seed=8),
    dict(name="tiny_bf16", dtype="bfloat16", B=1, Hq=1, Hkv=1, L=32, S=32, Dk=16, Dv=16, scale=None, seed=9),
    dict(name="d128_causal_f32", dtype="float32", B=1, Hq=1, Hkv=1, L=128, S=128, Dk=128, Dv=128, scale=None, causal=True, seed=10),
    dict(name="decode_gqa8_f16", dtype="float16", B=1, Hq=8, Hkv=1, L=32, S=256, Dk=64, Dv=64, scale=None, causal=True, seed=11),
    dict(name="scale_f32", dtype="float32", B=1, Hq=1, Hkv=1, L=32, S=32, Dk=32, Dv=32, scale=0.7, seed=12),
    dict(name="mask_broadcast_f32", dtype="float32", B=1, Hq=4, Hkv=2, L=64, S=64, Dk=32, Dv=32, scale=None, mask_kind="additive", mask_heads=1, seed=13),
]


def main():
    wanted = set(sys.argv[1:])
    for spec in FIXTURES:
        if wanted and spec["name"] not in wanted:
            continue
        spec = dict(spec)
        if spec.get("scale") is None:
            spec["scale"] = 1.0 / np.sqrt(spec["Dk"])
        make_fixture(**spec)


if __name__ == "__main__":
    main()
