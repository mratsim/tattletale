#!/usr/bin/env python3
"""Grouped GEMM fixture generator: numpy float64 reference over the exact
widened element bits, raw binary.

One directory per fixture under fixtures/ next to this script:

| file      | contents                                             |
| --------- | ---------------------------------------------------- |
| a.bin     | (total_m, K) packed lines, element bits              |
| b.bin     | (E, K, N) expert-stacked weights, element bits       |
| ids.bin   | (total_m) int32 per-line expert id, -1 marks padding |
| d.bin     | (total_m, N) expected output, raw float32            |
| meta.json | geometry, per-file checksums                         |

Recorded d contract, computed in float64 over the widened inputs:

| line                                      | recorded d value                                          |
| ----------------------------------------- | --------------------------------------------------------- |
| id g >= 0                                 | a(line)·b(g), the f64 result rounded to f32               |
| padding line, its m tile has a valid line | 0.0, the accumulator's zero-init                          |
| padding line, its m tile is all padding   | -999.0, the destination's pre-fill sentinel, never stored |

Input rules:

| input     | contents                                           |
| --------- | -------------------------------------------------- |
| a, b      | standard_normal, rounded once to the element dtype |
| ids       | exact ints from the fixture table                  |
| rng order | per fixture: a, then b                             |

Run:  uv run --with numpy python3 gen_grouped_gemm_fixtures.py [name ...]
      (default: all fixtures)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def xorhash_checksum(arr):
    """Xor over the raw bit patterns, order-independent, exact.

    - 2-byte arrays reduce their uint16 bit patterns
    - 4-byte arrays reduce their uint32 bit patterns
    - the Nim side xors the same bits, equality is exact
    """
    return int(np.bitwise_xor.reduce(arr.ravel().view(
        np.uint16 if arr.dtype.itemsize == 2 else np.uint32)))


def to_bits(values, dtype):
    """Rounds to the element dtype, returns the raw bit patterns."""
    if dtype == "float16":
        return values.astype(np.float16).view(np.uint16)
    assert dtype == "float32", f"unsupported dtype {dtype}"
    return values.astype(np.float32).view(np.uint32)


def widen(bits, dtype):
    """Widens element bit patterns to f64 exactly."""
    if dtype == "float16":
        return bits.view(np.float16).astype(np.float64)
    return bits.view(np.float32).astype(np.float64)


def grouped_gemm_reference(a, b, ids, mt):
    """float64 reference over the exact widened element values.

    Shapes:
    - a: (total_m, K) f64
    - b: (E, K, N) f64
    - ids: (total_m) int
    - mt: the kernel's m tile, deciding which padding lines share a stored tile

    Returns:
      The expected d as (total_m, N) float64.
    """
    total_m, k = a.shape
    n = b.shape[2]
    ids = np.asarray(ids)
    d = np.zeros((total_m, n), dtype=np.float64)
    for i in range(total_m):
        g = ids[i]
        if g >= 0:
            d[i] = a[i] @ b[g]
    for i in range(total_m):
        if ids[i] < 0:
            m0 = (i // mt) * mt
            if np.any(ids[m0:min(m0 + mt, total_m)] >= 0):
                d[i] = 0.0
            else:
                d[i] = -999.0
    return d


def build_ids(counts, pads):
    """Per-line expert ids from per-expert valid counts and -1 tail pads."""
    ids = []
    for g, (count, pad) in enumerate(zip(counts, pads)):
        ids.extend([g] * count)
        ids.extend([-1] * pad)
    return ids


def make_fixture(name, dtype, e, k, n, counts, pads, mt, seed):
    """Generates one fixture directory, prints its shape line.

    Preconditions:
    - counts[g] + pads[g] lines per expert block, one (count, pad) pair per expert
    - every id in ids is below e, and ids total to sum(counts) + sum(pads)
    """
    ids = build_ids(counts, pads)
    total_m = len(ids)
    assert len(counts) == e and len(pads) == e, f"{name}: one (count, pad) pair per expert"
    assert max(ids) < e and min(ids) >= -1, f"{name}: ids outside [-1, {e - 1}]"

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((total_m, k))
    b = rng.standard_normal((e, k, n))

    aBits = to_bits(a, dtype)
    bBits = to_bits(b, dtype)
    d = grouped_gemm_reference(
        widen(aBits, dtype).reshape(total_m, k),
        widen(bBits, dtype).reshape(e, k, n),
        ids, mt)
    dOut = d.astype(np.float32)

    idsArr = np.asarray(ids, dtype=np.int32)
    out = os.path.join(HERE, "fixtures", name)
    os.makedirs(out, exist_ok=True)
    files = {
        "a.bin": aBits,
        "b.bin": bBits,
        "ids.bin": idsArr,
        "d.bin": dOut,
    }
    for fname, arr in files.items():
        arr.tofile(os.path.join(out, fname))
    meta = {
        "name": name,
        "dtype": dtype,
        "E": e, "K": k, "N": n, "total_m": total_m, "m_tile": mt,
        "counts": counts, "pads": pads,
        "seed": seed,
        "xorhash_checksums": {fname: xorhash_checksum(np.ascontiguousarray(arr))
                              for fname, arr in files.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: dtype={dtype} E={e} K={k} N={n} total_m={total_m} "
          f"counts={counts} pads={pads}")


FIXTURES = [
    # aligned packed lines, no padding: every m tile sits inside one expert
    dict(name="grouped_aligned_packed_f32", dtype="float32", e=8, k=64, n=64,
         counts=[64] * 8, pads=[0] * 8, mt=32, seed=0xC18D1301),
    # counts straddling the 32-line m tiles, K crosses the 16-line depth tile,
    # N crosses the 32-column tile, -1 tails on some experts
    dict(name="grouped_boundary_cross_f32", dtype="float32", e=8, k=40, n=48,
         counts=[1, 3, 0, 33, 1, 0, 3, 33], pads=[2, 0, 5, 0, 31, 4, 1, 0],
         mt=32, seed=0xC18D1302),
    # every expert block padded to a 32 multiple, one expert empty so one full
    # m tile is all padding and must not be stored
    dict(name="grouped_padding_heavy_f32", dtype="float32", e=8, k=64, n=64,
         counts=[5, 17, 32, 1, 33, 0, 7, 20], pads=[27, 15, 0, 31, 31, 32, 25, 12],
         mt=32, seed=0xC18D1303),
    # wide: 512 experts with small cycling counts and -1 tails
    dict(name="grouped_wide_f32", dtype="float32", e=512, k=64, n=64,
         counts=[1, 3, 0, 33] * 128, pads=[1, 0, 2, 0] * 128,
         mt=32, seed=0xC18D1304),
    # float16 elements over the boundary-crossing geometry
    dict(name="grouped_boundary_cross_f16", dtype="float16", e=8, k=40, n=48,
         counts=[1, 3, 0, 33, 1, 0, 3, 33], pads=[2, 0, 5, 0, 31, 4, 1, 0],
         mt=32, seed=0xC18D1305),
]


def main():
    wanted = set(sys.argv[1:])
    for spec in FIXTURES:
        if wanted and spec["name"] not in wanted:
            continue
        make_fixture(**spec)


if __name__ == "__main__":
    main()
