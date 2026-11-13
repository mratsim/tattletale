#!/usr/bin/env python3
"""Grouped GEMV fixture generator: numpy float64 reference over the exact
widened element bits, raw binary.

One directory per fixture under fixtures/gemv/ next to this script:

| file        | contents                                            |
| ----------- | --------------------------------------------------- |
| x.bin       | (T, K) token activations, element bits              |
| b.bin       | (E, K, N) expert-stacked weights, element bits      |
| ids.bin     | (T, topk) int32 expert ids, -1 marks an unused slot |
| weights.bin | (T, topk) float32 routing weights                   |
| d.bin       | (T, N) expected output, raw float32                 |
| meta.json   | geometry, per-file checksums                        |

Recorded d contract, computed in float64 over the widened inputs:

| token slot state        | recorded d value                                              |
| ----------------------- | ------------------------------------------------------------- |
| at least one valid slot | sum over ascending i of w_i · (x(t)·b(ids(t,i))), f32-rounded |
| all slots -1            | 0.0, the kernel's zero sums, the kernel still stores it       |

Input rules:

| input     | contents                                           |
| --------- | -------------------------------------------------- |
| x, b      | standard_normal, rounded once to the element dtype |
| ids       | exact ints from the fixture table (or the seeds)   |
| weights   | seeded uniform reads, float32                      |
| rng order | per fixture: x, then b, then weights               |

Run:  uv run --with numpy python3 gen_grouped_gemv_fixtures.py [name ...]
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


def grouped_gemv_reference(x, b, ids, weights):
    """float64 reference over the exact widened element values.

    Shapes:
    - x: (T, K) f64
    - b: (E, K, N) f64
    - ids: (T, topk) int
    - weights: (T, topk) f64

    Slot sums run in ascending i, each finished slot dot scales by its routing weight,
    matching the kernel's combine contract.

    Returns:
      The expected d as (T, N) float64.
    """
    t, k = x.shape
    n = b.shape[2]
    topk = ids.shape[1]
    d = np.zeros((t, n), dtype=np.float64)
    for ti in range(t):
        acc = np.zeros(n, dtype=np.float64)
        for i in range(topk):
            g = ids[ti, i]
            if g >= 0:
                acc += weights[ti, i] * (x[ti] @ b[g])
        d[ti] = acc
    return d


def make_fixture(name, dtype, t, topk, e, k, n, ids, seed):
    """Generates one fixture directory, prints its shape line.

    Preconditions:
    - ids is the (t, topk) int table, every entry in [-1, e)
    - weights come from the fixture seed's uniform stream
    """
    ids = np.asarray(ids, dtype=np.int64)
    assert ids.shape == (t, topk), f"{name}: ids shape {ids.shape} != ({t}, {topk})"
    assert ids.min() >= -1 and ids.max() < e, f"{name}: ids outside [-1, {e - 1}]"

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t, k))
    b = rng.standard_normal((e, k, n))
    weights = rng.uniform(0.1, 1.0, size=(t, topk))

    xBits = to_bits(x, dtype)
    bBits = to_bits(b, dtype)
    weightsArr = weights.astype(np.float32)
    d = grouped_gemv_reference(
        widen(xBits, dtype).reshape(t, k),
        widen(bBits, dtype).reshape(e, k, n),
        ids, weightsArr.astype(np.float64))
    dOut = d.astype(np.float32)

    idsArr = ids.astype(np.int32)
    out = os.path.join(HERE, "fixtures", "gemv", name)
    os.makedirs(out, exist_ok=True)
    files = {
        "x.bin": xBits,
        "b.bin": bBits,
        "ids.bin": idsArr,
        "weights.bin": weightsArr,
        "d.bin": dOut,
    }
    for fname, arr in files.items():
        arr.tofile(os.path.join(out, fname))
    meta = {
        "name": name,
        "dtype": dtype,
        "T": t, "topk": topk, "E": e, "K": k, "N": n,
        "seed": seed,
        "xorhash_checksums": {fname: xorhash_checksum(np.ascontiguousarray(arr))
                              for fname, arr in files.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: dtype={dtype} T={t} topk={topk} E={e} K={k} N={n}")


FIXTURES = [
    # pure decode cell: one token, one slot
    dict(name="grouped_decode_f32", dtype="float32", t=1, topk=1, e=8, k=64, n=64,
         ids=[[3]], seed=0xC18D1401),
    # -1 sentinels in ids (invalid slots contribute 0), ragged N, one token whose
    # six slots are all invalid (its d must be exactly 0), one expert repeated
    # within a token
    dict(name="grouped_sentinel_ragged_f32", dtype="float32", t=8, topk=6, e=64, k=40, n=48,
         ids=[[5, 17, -1, 3, -1, 42],
              [0, 1, 2, 3, 4, 5],
              [-1, -1, -1, -1, -1, -1],
              [63, 63, 63, -1, -1, -1],
              [7, -1, 9, -1, 11, -1],
              [22, 33, 44, 55, 6, 8],
              [-1, 12, -1, 13, -1, 14],
              [31, 32, 33, 34, 35, 36]],
         seed=0xC18D1402),
    # wide: 512 experts, hot experts repeated across tokens, a hot 32-expert subset
    # supplies every slot id, no sentinels
    dict(name="grouped_wide_f32", dtype="float32", t=64, topk=6, e=512, k=64, n=64,
         ids="wide_hot", seed=0xC18D1403),
    # float16 elements over the sentinel geometry, ids identical to the f32 table
    dict(name="grouped_sentinel_ragged_f16", dtype="float16", t=8, topk=6, e=64, k=40, n=48,
         ids=[[5, 17, -1, 3, -1, 42],
              [0, 1, 2, 3, 4, 5],
              [-1, -1, -1, -1, -1, -1],
              [63, 63, 63, -1, -1, -1],
              [7, -1, 9, -1, 11, -1],
              [22, 33, 44, 55, 6, 8],
              [-1, 12, -1, 13, -1, 14],
              [31, 32, 33, 34, 35, 36]],
         seed=0xC18D1404),
    # one token, all six slots valid over few experts: the folded combine visible
    dict(name="grouped_single_token_f32", dtype="float32", t=1, topk=6, e=8, k=64, n=64,
         ids=[[0, 1, 0, 2, 1, 0]], seed=0xC18D1405),
]


def wide_hot_ids(t, topk, e):
    """ids over a hot 32-expert subset, repeating across tokens, seeded."""
    rng = np.random.default_rng(0xC18D1403)
    hot = rng.permutation(e)[:32]
    return [[int(hot[rng.integers(0, 32)]) for _ in range(topk)] for _ in range(t)]


def main():
    wanted = set(sys.argv[1:])
    for spec in FIXTURES:
        if wanted and spec["name"] not in wanted:
            continue
        spec = dict(spec)
        if spec["ids"] == "wide_hot":
            spec["ids"] = wide_hot_ids(spec["t"], spec["topk"], spec["e"])
        make_fixture(**spec)


if __name__ == "__main__":
    main()
