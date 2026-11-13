#!/usr/bin/env python3
"""MoE frontend fixture generator: numpy float64 references over the exact
widened element bits, raw binary.

One directory per fixture under fixtures/ next to this script. Three families,
one meta.json each with geometry, variant and per-file checksums:

router fixtures (scores are fp32 end to end):

| file        | contents                                   |
| ----------- | ------------------------------------------ |
| scores.bin  | (T, E) fp32 router logits                  |
| bias.bin    | (E,) fp32 selection bias, zeros = no bias  |
| ids.bin     | (T, topk) int32 selected experts, -1 excess|
| weights.bin | (T, topk) float32 routing weights          |

dispatch fixtures (integer-exact metadata):

| file            | contents                                        |
| --------------- | ----------------------------------------------- |
| ids.bin         | (T, topk) int32 expert ids, -1 unused slots     |
| x.bin           | (T, K) float32 activation rows                  |
| packedIds.bin   | (capacity,) int32 per-slot expert id, -1 pads   |
| packedOffsets.bin | (E,) int32 cumulative padded block ends       |
| sortedPairs.bin | (capacity,) int32 token per slot, -1 padding    |
| slotOfPair.bin  | (T, topk) int32 slot per pair, -1 unused        |
| packedX.bin     | (capacity, K) float32 gathered rows             |

end-to-end fixtures (the whole layer through moeForward):

| file       | contents                                    |
| ---------- | ------------------------------------------- |
| x.bin      | (T, K) element bits                         |
| b.bin      | (E, K, N) element bits                      |
| scores.bin | (T, E) fp32                                 |
| bias.bin   | (E,) fp32                                   |
| d.bin      | (T, N) float32 expected output              |

References, computed in float64 over the widened inputs:

- router: selection on scores + bias (lower id wins ties), weights from the
  UNBIASED scores (softmax over all E, max-shifted, or sigmoid), renorm over
  the selected set with the 1e-20 floor, ·scaling_factor, -1 excess slots
- dispatch: row-major (token, slot) enumeration, sort key (expert, enumeration
  index), 32-slot padded blocks, -1 padding
- end to end: d(t) = sum_i w(t, i)·(x(t)·b(ids(t, i))), the weights from the
  router reference (the dispatch is an exact integer permutation)

Run:  uv run --with numpy python3 gen_moe_fixtures.py [name ...]
      (default: all fixtures)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANT_SOFTMAX = 0
VARIANT_SIGMOID = 1


def xorhash_checksum(arr):
    """Xor over the raw bit patterns, order-independent, exact."""
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


def router_reference(scores, bias, topk, variant, scaling):
    """float64 router reference over the exact fp32 scores.

    Shapes: scores (T, E) f64, bias (E,) f64.

    Returns:
      (ids (T, topk) int, weights (T, topk) f64) per the frozen contract.
    """
    t, e = scores.shape
    ids = np.full((t, topk), -1, dtype=np.int64)
    weights = np.zeros((t, topk), dtype=np.float64)
    for ti in range(t):
        sel_val = np.zeros(topk, dtype=np.float64)
        sel_idx = np.full(topk, -1, dtype=np.int64)
        n_sel = 0
        for i in range(e):
            s = scores[ti, i] + bias[i]
            insert = n_sel < topk or s > sel_val[topk - 1]
            if insert:
                pos = min(n_sel, topk - 1)
                while pos > 0 and sel_val[pos - 1] < s:
                    sel_val[pos] = sel_val[pos - 1]
                    sel_idx[pos] = sel_idx[pos - 1]
                    pos -= 1
                sel_val[pos] = s
                sel_idx[pos] = i
                if n_sel < topk:
                    n_sel += 1
        w = np.zeros(n_sel, dtype=np.float64)
        if variant == VARIANT_SOFTMAX:
            m = scores[ti].max()
            z = np.exp(scores[ti] - m).sum()
            for i in range(n_sel):
                w[i] = np.exp(scores[ti, sel_idx[i]] - m) / z
        else:
            for i in range(n_sel):
                w[i] = 1.0 / (1.0 + np.exp(-scores[ti, sel_idx[i]]))
        denom = max(w.sum(), 1e-20)
        for i in range(n_sel):
            ids[ti, i] = sel_idx[i]
            weights[ti, i] = w[i] / denom * scaling
    return ids, weights


def dispatch_reference(ids, e):
    """Integer-exact dispatch metadata reference.

    ids: (T, topk) int, -1 unused slots.

    Returns:
      (packed_ids, packed_offsets, sorted_pairs, slot_of_pair) as int lists.
    """
    t, topk = ids.shape
    counts = np.zeros(e, dtype=np.int64)
    ranks = np.zeros((t, topk), dtype=np.int64)
    for ti in range(t):
        for ip in range(topk):
            g = ids[ti, ip]
            if g >= 0:
                ranks[ti, ip] = counts[g]
                counts[g] += 1
    offsets = np.zeros(e, dtype=np.int64)
    acc = 0
    for g in range(e):
        acc += (counts[g] + 31) // 32 * 32
        offsets[g] = acc
    cap = t * topk + 31 * e
    sorted_pairs = np.full(cap, -1, dtype=np.int64)
    slot_of_pair = np.full((t, topk), -1, dtype=np.int64)
    for ti in range(t):
        for ip in range(topk):
            g = ids[ti, ip]
            if g >= 0:
                start = offsets[g] - (counts[g] + 31) // 32 * 32
                slot = start + ranks[ti, ip]
                sorted_pairs[slot] = ti
                slot_of_pair[ti, ip] = slot
        for ip in range(topk):
            g = ids[ti, ip]
            if g >= 0:
                start = offsets[g] - (counts[g] + 31) // 32 * 32
                sorted_pairs[start + counts[g]:offsets[g]] = -1
    packed_ids = np.full(cap, -1, dtype=np.int64)
    for slot in range(cap):
        if sorted_pairs[slot] >= 0:
            expert = int(np.searchsorted(offsets, slot, side="right"))
            packed_ids[slot] = expert
    return packed_ids, offsets, sorted_pairs, slot_of_pair


def make_router_fixture(name, t, e, topk, variant, scaling, bias_scale, seed):
    """Router fixture: fp32 scores and bias, f64 reference, fp32 weights."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-2.0, 2.0, size=(t, e)).astype(np.float32)
    bias = (rng.uniform(-bias_scale, bias_scale, size=e)).astype(np.float32)
    ids, weights = router_reference(
        scores.astype(np.float64), bias.astype(np.float64), topk, variant, scaling)
    ids_arr = ids.astype(np.int32)
    weights_arr = weights.astype(np.float32)
    out = os.path.join(HERE, "fixtures", name)
    os.makedirs(out, exist_ok=True)
    files = {
        "scores.bin": scores,
        "bias.bin": bias,
        "ids.bin": ids_arr,
        "weights.bin": weights_arr,
    }
    for fname, arr in files.items():
        arr.tofile(os.path.join(out, fname))
    meta = {
        "name": name, "family": "router", "T": t, "E": e, "topk": topk,
        "variant": variant, "scaling": scaling,
        "xorhash_checksums": {fname: xorhash_checksum(np.ascontiguousarray(arr))
                              for fname, arr in files.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: router T={t} E={e} topk={topk} variant={variant} scaling={scaling}")


def make_dispatch_fixture(name, t, topk, e, k, ids, seed):
    """Dispatch fixture: exact integer metadata plus the gathered f32 rows."""
    ids = np.asarray(ids, dtype=np.int64)
    assert ids.shape == (t, topk), f"{name}: ids shape {ids.shape} != ({t}, {topk})"
    assert ids.min() >= -1 and ids.max() < e, f"{name}: ids outside [-1, {e - 1}]"
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(t, k))
    packed_ids, offsets, sorted_pairs, slot_of_pair = dispatch_reference(ids, e)
    cap = sorted_pairs.shape[0]
    packed_x = np.zeros((cap, k), dtype=np.float32)
    for slot in range(cap):
        src = sorted_pairs[slot]
        if src >= 0:
            packed_x[slot] = x[src]
    ids_arr = ids.astype(np.int32)
    files = {
        "ids.bin": ids_arr,
        "x.bin": x.astype(np.float32),
        "packedIds.bin": packed_ids.astype(np.int32),
        "packedOffsets.bin": offsets.astype(np.int32),
        "sortedPairs.bin": sorted_pairs.astype(np.int32),
        "slotOfPair.bin": slot_of_pair.astype(np.int32),
        "packedX.bin": packed_x,
    }
    out = os.path.join(HERE, "fixtures", name)
    os.makedirs(out, exist_ok=True)
    for fname, arr in files.items():
        arr.tofile(os.path.join(out, fname))
    meta = {
        "name": name, "family": "dispatch", "T": t, "topk": topk, "E": e, "K": k,
        "xorhash_checksums": {fname: xorhash_checksum(np.ascontiguousarray(arr))
                              for fname, arr in files.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: dispatch T={t} topk={topk} E={e} K={k} cap={cap}")


def make_e2e_fixture(name, dtype, t, topk, e, k, n, variant, scaling, bias_scale, seed):
    """End-to-end fixture: the whole layer's expected d in float32."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(t, k))
    b = rng.uniform(-2.0, 2.0, size=(e, k, n))
    scores = rng.uniform(-2.0, 2.0, size=(t, e)).astype(np.float32)
    bias = (rng.uniform(-bias_scale, bias_scale, size=e)).astype(np.float32)
    ids, weights = router_reference(
        scores.astype(np.float64), bias.astype(np.float64), topk, variant, scaling)
    xw = widen(to_bits(x, dtype), dtype)
    bw = widen(to_bits(b, dtype), dtype)
    d = np.zeros((t, n), dtype=np.float64)
    for ti in range(t):
        for i in range(topk):
            g = ids[ti, i]
            if g >= 0:
                d[ti] += weights[ti, i] * (xw[ti] @ bw[g])
    d_out = d.astype(np.float32)
    files = {
        "x.bin": to_bits(x, dtype),
        "b.bin": to_bits(b, dtype),
        "scores.bin": scores,
        "bias.bin": bias,
        "d.bin": d_out,
    }
    out = os.path.join(HERE, "fixtures", name)
    os.makedirs(out, exist_ok=True)
    for fname, arr in files.items():
        arr.tofile(os.path.join(out, fname))
    meta = {
        "name": name, "family": "e2e", "dtype": dtype, "T": t, "topk": topk,
        "E": e, "K": k, "N": n, "variant": variant, "scaling": scaling,
        "xorhash_checksums": {fname: xorhash_checksum(np.ascontiguousarray(arr))
                              for fname, arr in files.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: e2e {dtype} T={t} topk={topk} E={e} K={k} N={n} "
          f"variant={variant} scaling={scaling}")


FIXTURES = [
    # router: softmax and sigmoid, bias on and off, scaling on one, topk over E
    dict(name="router_softmax_nobias", t=4, e=16, topk=6, variant=VARIANT_SOFTMAX,
         scaling=1.0, bias_scale=0.0, seed=0xC18D1501),
    dict(name="router_softmax_bias_scale", t=4, e=16, topk=6, variant=VARIANT_SOFTMAX,
         scaling=2.5, bias_scale=0.5, seed=0xC18D1502),
    dict(name="router_sigmoid_bias", t=4, e=16, topk=6, variant=VARIANT_SIGMOID,
         scaling=1.0, bias_scale=0.5, seed=0xC18D1503),
    dict(name="router_topk_over_e", t=2, e=3, topk=6, variant=VARIANT_SOFTMAX,
         scaling=1.0, bias_scale=0.0, seed=0xC18D1504),
    # dispatch: T=1, empty experts and duplicates, topk over E
    dict(name="dispatch_single_token", t=1, topk=1, e=8, k=40,
         ids=[[3]], seed=0xC18D1511),
    dict(name="dispatch_empty_experts", t=8, topk=6, e=16, k=24,
         ids=[[0, 0, 5, -1, 5, 12],
              [15, -1, -1, 2, 2, 2],
              [1, 1, 1, 1, 1, 1],
              [7, 8, 9, 10, 11, 12],
              [-1, -1, 3, -1, 3, -1],
              [4, 4, 4, 4, 4, 4],
              [13, 14, 15, 0, 1, 2],
              [6, -1, 6, -1, 6, -1]], seed=0xC18D1512),
    dict(name="dispatch_topk_over_e", t=4, topk=6, e=3, k=12,
         ids=[[0, 1, 2, 0, 1, 2],
              [2, 2, 2, 1, 1, 1],
              [1, 0, -1, 2, -1, 0],
              [0, 0, 1, 1, 2, 2]], seed=0xC18D1513),
    # end to end: gemm tier f32, gemv tier f32, gemm tier f16
    dict(name="e2e_gemm_f32", dtype="float32", t=32, topk=6, e=8, k=24, n=32,
         variant=VARIANT_SOFTMAX, scaling=1.0, bias_scale=0.5, seed=0xC18D1521),
    dict(name="e2e_gemv_f32", dtype="float32", t=4, topk=6, e=8, k=24, n=32,
         variant=VARIANT_SIGMOID, scaling=2.5, bias_scale=0.5, seed=0xC18D1522),
    dict(name="e2e_gemm_f16", dtype="float16", t=32, topk=6, e=8, k=24, n=32,
         variant=VARIANT_SOFTMAX, scaling=1.0, bias_scale=0.5, seed=0xC18D1523),
]


def main():
    wanted = set(sys.argv[1:])
    for spec in FIXTURES:
        if wanted and spec["name"] not in wanted:
            continue
        spec = dict(spec)
        if spec["name"].startswith("router"):
            make_router_fixture(**spec)
        elif spec["name"].startswith("dispatch"):
            make_dispatch_fixture(**spec)
        else:
            make_e2e_fixture(**spec)


if __name__ == "__main__":
    main()
