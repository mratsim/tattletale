#!/usr/bin/env python3
"""Jagged attention fixture generator: numpy float64 reference, raw binary.

One directory per fixture under fixtures/ next to this script:

| file           | contents                                                          |
| -------------- | ----------------------------------------------------------------- |
| q.bin          | (num_qo_tokens, H, D) queries, element bits                       |
| k.bin / v.bin  | (num_pages, num_layers, page_size, Nkv, D) KV pools, element bits |
| o.bin          | (num_qo_tokens, H, D) expected output, raw float32                |
| q_offsets.bin  | (num_seqs+1) int32, per-segment q token ranges                    |
| kv_offsets.bin | (num_seqs+1) int32, per-segment kv_indices ranges                 |
| kv_indices.bin | (num_indices) int32, page ids per segment                         |
| kv_seqlens.bin | (num_seqs) int32, KV token lengths per segment                    |
| meta.json      | geometry, scale, layer, per-file checksums                        |

The reference runs in float64, widened exactly from the recorded float16 and float32 element bits,
reproducing the semantics of the v1 kernel at workspace/positron/src/kernels/ceramic_v1/attn_ssm/grouped_query_attention_paged.nim

- per segment s with q_len = q_offsets[s+1] - q_offsets[s] and kv_len = kv_seqlens[s], decode (q_len == 1) attends the first kv_len keys,
  prefill (q_len >= 2) query token j attends keys 0 through kv_len - q_len + j,
  flash-attn varlen with the causal band per query token and the GQA
  mapping kv_head = q_head div (H div Nkv)
- KV addressing: token t of segment s is stored at page id
  kv_indices[kv_offsets[s] + t div page_size] and in-page position
  t mod page_size, so k = k_pool[page, layer, t mod page_size, kv_head, :]
- softmax in f64 computes exp(s - s_max) directly, one log2(e) factor away
  from the kernel's exp2 shape exp2((s - m) * log2(e)) with q_mul = scale * log2(e) folded into q
- expected o is recorded once as float32 bits with the f64 result rounded round-to-nearest-even

Pool garbage: every pool entry outside the union of the segments' covered
token positions at the attended layer is NaN. The reference gathers only
attended tokens so the recorded output is finite, and a kernel that reads
one garbage token poisons its output elements and fails the band.

Input rules:

| file      | contents                                                                                                        |
| --------- | --------------------------------------------------------------------------------------------------------------- |
| q         | standard_normal, rounded once to the element dtype                                                              |
| k, v pool | standard_normal everywhere, covered entries at the attended layer overwritten with fresh standard_normal values |
| garbage   | NaN at the attended layer outside the union coverage, other layers untouched                                    |
| metadata  | exact ints from the fixture table                                                                               |
| rng order | per fixture: q, k pool fill, v pool fill, then per-entry k then v overwrites in (page, position) order          |

Run:  uv run --with numpy python3 gen_jagged_attention_fixtures.py [name ...]
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


def write_bin(path, arr):
    arr.tofile(path)


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


def jagged_attention_reference(q, k, v, q_offsets, kv_offsets, kv_indices,
                               kv_seqlens, H, Nkv, D, page_size, layer, scale):
    """float64 reference over the exact widened element values.

    Shapes:
    - q: (num_qo_tokens, H, D) f64
    - k, v: (num_pages, num_layers, page_size, Nkv, D) f64
    - q_offsets, kv_offsets: (num_seqs+1) int
    - kv_indices: (num_indices) int, kv_seqlens: (num_seqs) int

    Returns:
      The expected o as (num_qo_tokens, H, D) float64.
    """
    num_pages = k.shape[0]
    num_qo = q.shape[0]
    num_seqs = len(kv_seqlens)
    kflat = k[:, layer].reshape(num_pages * page_size, Nkv, D)
    vflat = v[:, layer].reshape(num_pages * page_size, Nkv, D)
    o = np.zeros((num_qo, H, D), dtype=np.float64)
    for s in range(num_seqs):
        q0, q1 = q_offsets[s], q_offsets[s + 1]
        q_len = q1 - q0
        kv_len = kv_seqlens[s]
        pos = [int(kv_indices[kv_offsets[s] + t // page_size]) * page_size
               + t % page_size for t in range(kv_len)]
        kg = kflat[pos]
        vg = vflat[pos]
        for h in range(H):
            hkv = h // (H // Nkv)
            for j in range(q_len):
                hi = kv_len - 1 if q_len == 1 else kv_len - q_len + j
                scores = (kg[:hi + 1, hkv] @ q[q0 + j, h]) * scale
                p = np.exp(scores - scores.max())
                o[q0 + j, h] = (p @ vg[:hi + 1, hkv]) / p.sum()
    return o


def make_fixture(name, dtype, H, Nkv, D, page_size, num_layers, layer,
                 segments, seed):
    """Generates one fixture directory, prints its shape line.

    Preconditions:
    - H is a multiple of Nkv (GQA head mapping)
    - segments: one (q_len, kv_len) pair per segment or (q_len, kv_len, pages) with an explicit page-id list for shared pages
    - page ids are assigned sequentially across segments unless explicit
    """
    assert H % Nkv == 0, f"{name}: H {H} is not a multiple of Nkv {Nkv}"
    assert D == 64 and page_size == 32 and num_layers == 2, \
        f"{name}: the suite fixes D=64, page_size=32, num_layers=2"
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(D)

    q_offsets = [0]
    kv_offsets = [0]
    indices = []
    next_page = 0
    for seg in segments:
        q_len, kv_len = seg[0], seg[1]
        pages = list(seg[2]) if len(seg) == 3 else \
            list(range(next_page, next_page + (kv_len + page_size - 1) // page_size))
        assert kv_len <= len(pages) * page_size, \
            f"{name}: segment kv_len {kv_len} exceeds its {len(pages)} pages"
        next_page = max(next_page, max(pages) + 1)
        q_offsets.append(q_offsets[-1] + q_len)
        kv_offsets.append(kv_offsets[-1] + len(pages))
        indices.extend(pages)
    num_pages = next_page
    num_qo = q_offsets[-1]
    num_seqs = len(segments)

    q = rng.standard_normal((num_qo, H, D))
    kpool = rng.standard_normal((num_pages, num_layers, page_size, Nkv, D))
    vpool = rng.standard_normal((num_pages, num_layers, page_size, Nkv, D))

    # union coverage of the segments' token positions, then fresh standard_normal
    # values on the covered entries of the attended layer and NaN outside
    covered = np.zeros((num_pages, page_size), dtype=bool)
    for s, (q_len, kv_len) in enumerate((seg[0], seg[1]) for seg in segments):
        for t in range(kv_len):
            page = indices[kv_offsets[s] + t // page_size]
            covered[page, t % page_size] = True
    for page in range(num_pages):
        for pos in range(page_size):
            if covered[page, pos]:
                kpool[page, layer, pos] = rng.standard_normal((Nkv, D))
                vpool[page, layer, pos] = rng.standard_normal((Nkv, D))
            else:
                kpool[page, layer, pos] = np.nan
                vpool[page, layer, pos] = np.nan

    # round inputs to the storage dtype so the reference replays the kernel's values
    qBits = to_bits(q, dtype)
    kBits = to_bits(kpool, dtype).reshape(num_pages, num_layers, page_size, Nkv, D)
    vBits = to_bits(vpool, dtype).reshape(num_pages, num_layers, page_size, Nkv, D)
    o = jagged_attention_reference(
        widen(qBits, dtype).reshape(num_qo, H, D),
        widen(kBits, dtype), widen(vBits, dtype),
        q_offsets, kv_offsets, indices, [seg[1] for seg in segments],
        H, Nkv, D, page_size, layer, scale)
    oOut = o.astype(np.float32)

    qOff = np.asarray(q_offsets, dtype=np.int32)
    kvOff = np.asarray(kv_offsets, dtype=np.int32)
    kvIdx = np.asarray(indices, dtype=np.int32)
    kvSeq = np.asarray([seg[1] for seg in segments], dtype=np.int32)

    out = os.path.join(HERE, "fixtures", name)
    os.makedirs(out, exist_ok=True)
    files = {
        "q.bin": qBits,
        "k.bin": kBits,
        "v.bin": vBits,
        "o.bin": oOut,
        "q_offsets.bin": qOff,
        "kv_offsets.bin": kvOff,
        "kv_indices.bin": kvIdx,
        "kv_seqlens.bin": kvSeq,
    }
    for fname, arr in files.items():
        write_bin(os.path.join(out, fname), np.ascontiguousarray(arr))
    meta = {
        "name": name,
        "dtype": dtype,
        "num_seqs": num_seqs, "H": H, "Nkv": Nkv, "D": D,
        "page_size": page_size, "num_layers": num_layers, "layer": layer,
        "num_pages": num_pages, "num_qo_tokens": num_qo,
        "num_indices": len(indices),
        "scale": scale,
        "q_offsets": q_offsets, "kv_offsets": kv_offsets,
        "kv_seqlens": [seg[1] for seg in segments],
        "seed": seed,
        "xorhash_checksums": {fname: xorhash_checksum(np.ascontiguousarray(arr))
                              for fname, arr in files.items()},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: dtype={dtype} num_seqs={num_seqs} H={H} Nkv={Nkv} D={D} "
          f"page_size={page_size} num_pages={num_pages} num_qo_tokens={num_qo} "
          f"kv_seqlens={[seg[1] for seg in segments]}")


FIXTURES = [
    # decode-only, ragged cached lengths crossing page boundaries and one
    # multi-page sequence, GQA H=8 over Nkv=2, partial last pages
    dict(name="jagged_decode_gqa_f16", dtype="float16", H=8, Nkv=2,
         segments=[(1, 1), (1, 33), (1, 64), (1, 97)], seed=0xC17D4101),
    # mixed decode and prefill with ragged q and cached lengths, causal-banded prefill over cached and new tokens
    dict(name="jagged_prefill_gqa_f16", dtype="float16", H=8, Nkv=2,
         segments=[(1, 1), (5, 35), (9, 79)], seed=0xC17D4102),
    # decode over shared page ids across segments with prefix sharing,
    # Nkv=1, the extreme GQA ratio
    dict(name="jagged_prefix_share_f16", dtype="float16", H=8, Nkv=1,
         segments=[(1, 33, [0, 1]), (1, 65, [0, 1, 2]), (1, 17, [0])],
         seed=0xC17D4103),
    # identity page ids with the constant-stride dense block-table layout,
    # prefill regime and the cross-check basis against a dense layout
    dict(name="jagged_dense_degenerate_f16", dtype="float16", H=8, Nkv=4,
         segments=[(4, 36), (2, 5)], seed=0xC17D4104),
    # f32 elements, decode + prefill over GQA, judged at the f32 band
    dict(name="jagged_prefill_f32", dtype="float32", H=8, Nkv=2,
         segments=[(1, 50), (8, 28)], seed=0xC17D4105),
]


def main():
    wanted = set(sys.argv[1:])
    for spec in FIXTURES:
        if wanted and spec["name"] not in wanted:
            continue
        make_fixture(D=64, page_size=32, num_layers=2, layer=1, **spec)


if __name__ == "__main__":
    main()
