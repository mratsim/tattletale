#!/usr/bin/env python3
"""Paged SDPA fixture generator: python + numpy, libtorch-free, raw binary.

One directory per fixture under fixtures/ next to this script:

| file              | contents                                                         |
| ----------------- | ---------------------------------------------------------------- |
| q.bin             | (num_seqs, Hq, 1, Dk) queries, raw little-endian, the meta dtype |
| k.bin / v.bin     | (num_pages, num_layers, page_size, Nkv, D) KV pools, same layout |
| block_table.bin   | (num_seqs, max_pages) int32 page ids, -1 padding                 |
| cache_seqlens.bin | (num_seqs,) int32 lengths                                        |
| o.bin             | expected output, raw float32                                     |
| meta.json         | name, dtype, geometry, layer, scale, cache_seqlens, checksums    |

The reference runs in float64 end to end for a tight truth.
Per sequence, the attended layer's pool entries are gathered through the block table,
entry t mapping to `block_table[s, t div page_size] * page_size + t mod page_size`
while t < cache_seqlens[s].

Attention then computes o = softmax(scale * q @ K^T) @ V over the gathered
entries with no causal mask, kv head = h div (Hq div Nkv).

A sliding window narrows the attended range to [cache_seqlens[s] - window_size, cache_seqlens[s]).

Pool entries beyond an attended range default to finite random values, the nan_tail option writes NaN there, checked against the recorded output.

Run:  uv run --with numpy python3 gen_sdpa_paged_fixtures.py [name ...]
      (default: all fixtures)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def xorhash_checksum(arr):
    """Xor over the raw bit patterns, order-independent, exact.

    - 4-byte arrays reduce their uint32 bit patterns
    - 2-byte arrays reduce their uint16 bit patterns
    - the Nim side xors the same bits, equality is exact
    """
    return int(np.bitwise_xor.reduce(arr.ravel().view(
        np.uint16 if arr.dtype.itemsize == 2 else np.uint32)))


def write_bin(path, arr):
    arr.tofile(path)


def sdpa_paged_reference(q, kpool, vpool, block_table, cache_seqlens, layer, scale, window_size=None):
    """float64 reference: per sequence, a block-table gather feeding
    softmax(scale * q @ K^T) @ V, causal off, over the gathered entries."""
    num_seqs, Hq, _, Dk = q.shape
    num_pages, _, page_size, Nkv, _ = kpool.shape
    Dv = vpool.shape[-1]
    g = Hq // Nkv
    kflat = kpool.astype(np.float64)[:, layer].reshape(num_pages * page_size, Nkv, Dk)
    vflat = vpool.astype(np.float64)[:, layer].reshape(num_pages * page_size, Nkv, Dv)
    o = np.zeros((num_seqs, Hq, 1, Dv), dtype=np.float64)
    for s in range(num_seqs):
        covered = int(cache_seqlens[s])
        lo = 0 if window_size is None else max(0, covered - int(window_size))
        rows = [int(block_table[s, t // page_size]) * page_size + t % page_size
                for t in range(lo, covered)]
        kg = kflat[rows]
        vg = vflat[rows]
        for h in range(Hq):
            hkv = h // g
            qh = q[s, h, 0, :].astype(np.float64)
            scores = (kg[:, hkv] @ qh) * scale
            p = np.exp(scores - scores.max())
            o[s, h, 0, :] = (p @ vg[:, hkv]) / p.sum()
    return o


def make_fixture(name, dtype, num_seqs, Hq, Hkv, Dk, Dv, page_size, max_pages,
                 num_layers, layer, num_pages, cache_seqlens, seed, window_size=None,
                 nan_tail=False):
    """Generates one fixture directory, prints its shape line."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(Dk)

    q = rng.standard_normal((num_seqs, Hq, 1, Dk))
    # random slab values fill the pool everywhere, the attended layer's
    # entries a sequence covers are overwritten below
    kpool = rng.standard_normal((num_pages, num_layers, page_size, Hkv, Dk))
    vpool = rng.standard_normal((num_pages, num_layers, page_size, Hkv, Dv))

    block_table = np.full((num_seqs, max_pages), -1, dtype=np.int32)
    cache_seqlens = np.asarray(cache_seqlens, dtype=np.int32)
    next_page = 0
    for s, seqlen in enumerate(cache_seqlens):
        pages = (int(seqlen) + page_size - 1) // page_size
        for p in range(pages):
            block_table[s, p] = next_page
            next_page += 1
        for t in range(int(seqlen)):
            page = int(block_table[s, t // page_size])
            kpool[page, layer, t % page_size] = rng.standard_normal((Hkv, Dk))
            vpool[page, layer, t % page_size] = rng.standard_normal((Hkv, Dv))
    assert next_page <= num_pages, f"{name}: {next_page} pages needed, pool holds {num_pages}"
    if nan_tail:
        covered = np.zeros((num_pages, page_size), dtype=bool)
        for s, seqlen in enumerate(cache_seqlens):
            for t in range(int(seqlen)):
                covered[int(block_table[s, t // page_size]), t % page_size] = True
        for page in range(num_pages):
            for pr in range(page_size):
                if not covered[page, pr]:
                    kpool[page, layer, pr] = np.nan
                    vpool[page, layer, pr] = np.nan
    for s, seqlen in enumerate(cache_seqlens):
        assert int(seqlen) <= max_pages * page_size, \
            f"{name}: seq {s} length {int(seqlen)} exceeds the table capacity"

    # round inputs to the storage dtype so the reference replays the kernel's values
    q = q.astype(np.float16).astype(np.float32)
    kpool = kpool.astype(np.float16).astype(np.float32)
    vpool = vpool.astype(np.float16).astype(np.float32)

    o = sdpa_paged_reference(q, kpool, vpool, block_table, cache_seqlens, layer, scale,
                             window_size)

    out = os.path.join(HERE, "fixtures", name)
    os.makedirs(out, exist_ok=True)
    qOut, kOut, vOut = q.astype(np.float16), kpool.astype(np.float16), vpool.astype(np.float16)
    oOut = o.astype(np.float32)
    write_bin(os.path.join(out, "q.bin"), qOut)
    write_bin(os.path.join(out, "k.bin"), kOut)
    write_bin(os.path.join(out, "v.bin"), vOut)
    write_bin(os.path.join(out, "block_table.bin"), block_table)
    write_bin(os.path.join(out, "cache_seqlens.bin"), cache_seqlens)
    write_bin(os.path.join(out, "o.bin"), oOut)

    # per-file xorhash checksums let the Nim roundtrip verify every file, tables included
    meta = {
        "name": name, "dtype": dtype,
        "num_seqs": num_seqs, "Hq": Hq, "Hkv": Hkv, "Dk": Dk, "Dv": Dv,
        "page_size": page_size, "max_pages": max_pages,
        "num_layers": num_layers, "layer": layer, "num_pages": num_pages,
        "scale": scale, "window_size": int(window_size or 0),
        "nan_tail": bool(nan_tail),
        "cache_seqlens": [int(x) for x in cache_seqlens],
        "checksum": xorhash_checksum(oOut),
        "xorhash_checksums": {
            "q.bin": xorhash_checksum(qOut),
            "k.bin": xorhash_checksum(kOut),
            "v.bin": xorhash_checksum(vOut),
            "block_table.bin": xorhash_checksum(block_table),
            "cache_seqlens.bin": xorhash_checksum(cache_seqlens),
        },
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name}: dtype={dtype} num_seqs={num_seqs} Hq={Hq} Hkv={Hkv} "
          f"Dk={Dk} Dv={Dv} page_size={page_size} max_pages={max_pages} "
          f"num_layers={num_layers} layer={layer} num_pages={num_pages} "
          f"cache_seqlens={cache_seqlens.tolist()} checksum={o.sum():.6f}")


FIXTURES = [
    # multi-page, one-page + partial, length-1 sequences, Hq=4 over Hkv=2
    dict(name="paged_decode_gqa_f16", dtype="float16", num_seqs=3, Hq=4, Hkv=2,
         Dk=64, Dv=64, page_size=32, max_pages=4, num_layers=2, layer=1,
         num_pages=6, cache_seqlens=[70, 33, 1], seed=101),
    # a sub-tile length and an exact tile multiple, pages 3..5 stay garbage
    dict(name="paged_decode_small_f16", dtype="float16", num_seqs=2, Hq=2, Hkv=1,
         Dk=64, Dv=64, page_size=32, max_pages=4, num_layers=2, layer=1,
         num_pages=6, cache_seqlens=[17, 64], seed=102),
    # NaN covers every pool entry no sequence attends: unassigned pages 3..5 and seq 0's tail entries 17..31, the bounded loads must zero-pad them
    dict(name="paged_nan_tail_decode_f16", dtype="float16", num_seqs=2, Hq=2, Hkv=1,
         Dk=64, Dv=64, page_size=32, max_pages=4, num_layers=2, layer=1,
         num_pages=6, cache_seqlens=[17, 64], seed=104, nan_tail=True),
    # a mid-page window left edge (70 - 40 = 30) and a window starting past
    # page 0 (100 - 40 = 60, kv tile 1), the mask cuts the skipped entries
    dict(name="paged_window_decode_f16", dtype="float16", num_seqs=2, Hq=4, Hkv=2,
         Dk=64, Dv=64, page_size=32, max_pages=4, num_layers=2, layer=1,
         num_pages=7, cache_seqlens=[70, 100], seed=103, window_size=40),
]


def main():
    wanted = set(sys.argv[1:])
    for spec in FIXTURES:
        if wanted and spec["name"] not in wanted:
            continue
        make_fixture(**spec)


if __name__ == "__main__":
    main()
