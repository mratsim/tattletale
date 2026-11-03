# Paged SDPA benchmark: MLX block-table gather + mx.fast.scaled_dot_product_attention.
# Same case list and flop model as bench_paged_sdpa_decode_metal.nim,
# GQA Hq=32 over Hkv=8, Dk=Dv=128, page 32, pool layer 1 of 2.
# Dtype from argv: f32 (default) or bf16.
#
# No fused paged attention exists in MLX: the bench gathers each sequence's K/V
# through the block table, then runs mx.fast.scaled_dot_product_attention.
# Fused arguments: L=1, causal off, per-sequence validity masking.
# MLX pays one block-table gather per timed call: the Nim kernel reads pool pages in place.
#
# Effective flops, identical to bench_paged_sdpa_decode_metal.nim and the non-paged bench:
# - flops: 2 * sum over sequences of attended_seqlen * Hq * (Dk + Dv)
# - attended_seqlen = seqlen - windowLo, windowLo = max(0, seqlen - window)
#
# Methodology: batched eval, one mx.eval per batch calls, median of 3 timed rounds.
# Nim amortizes launch overhead with iters attention passes inside one kernel
# launch and reports the median of 3 launches. Batched eval gives the same amortization here.
#
# Run from the worktree root:
#   uv run python workspace/ceramic/bench/sdpa_gpu/bench_paged_sdpa_decode_mlx.py [bf16]

import math
import sys
import time

import mlx.core as mx

DTYPE = mx.bfloat16 if (len(sys.argv) > 1 and sys.argv[1] == "bf16") else mx.float32

HQ, HKV, DK, DV = 32, 8, 128, 128
PAGE, LAYERS, LAYER = 32, 2, 1

# Case list of bench_paged_sdpa_decode_metal.nim:
# (label, num_seqs, seqlen, window, batch, rounds). Window 0 attends the full sequence.
# batch and rounds size each timed round into the tens of ms.
# Large cases keep the gathered K+V per call above L2, timing measures DRAM streaming.
CASES = [
    ("l2-resident", 8, 1024, 0, 20, 10),
    ("dram", 32, 8192, 0, 2, 5),
    ("dram", 64, 8192, 0, 1, 3),
    ("dram-window", 32, 8192, 1024, 8, 5),
]


def make(shape):
    # logical index mod 7, times 0.25 plus 0.5, matching the Nim fill pattern
    n = math.prod(shape)
    return (mx.arange(n, dtype=mx.float32) % 7 * 0.25 + 0.5).reshape(shape).astype(DTYPE)


def block_table(num_seqs, pages_per_seq):
    # sequential page ids across sequences, bench tables are exactly full, no -1 padding
    bt = mx.zeros((num_seqs, pages_per_seq), dtype=mx.int32)
    next_page = 0
    for s in range(num_seqs):
        for p in range(pages_per_seq):
            bt[s, p] = next_page
            next_page += 1
    return bt


def gather_kv(kpool, vpool, bt, page_lo, page_hi):
    # Per-sequence K/V gathered through the block table: the overhead MLX
    # pays for the paged representation.
    # Input pools: (num_pages, LAYERS, PAGE, Hkv, D), layer LAYER served.
    # Output: (num_seqs, Hkv, (page_hi - page_lo) * PAGE, D) per tensor.
    idx = bt[:, page_lo:page_hi]
    k_l1, v_l1 = kpool[:, LAYER], vpool[:, LAYER]
    kg = k_l1[idx]
    vg = v_l1[idx]
    kg = kg.reshape(kg.shape[0], -1, kg.shape[3], kg.shape[4]).transpose(0, 2, 1, 3)
    vg = vg.reshape(vg.shape[0], -1, vg.shape[3], vg.shape[4]).transpose(0, 2, 1, 3)
    return kg, vg


def validity_mask(gathered_s, key_lo, cache_seqlens, window):
    # Per-sequence validity over the gathered key range: gathered position j is
    # global key key_lo + j, valid in [max(0, seqlen - window), seqlen).
    # Shape (num_seqs, 1, 1, gathered_s), broadcast over heads.
    g = key_lo + mx.arange(gathered_s)
    lo = mx.array([max(0, sl - window) if window else 0 for sl in cache_seqlens])
    hi = mx.array(cache_seqlens)
    mask = (g[None, :] >= lo[:, None]) & (g[None, :] < hi[:, None])
    return mask[:, None, None, :]


def ref_paged(q, kpool, vpool, bt, cache_seqlens, page, window, scale):
    # check_math reference: explicit per-sequence page walk and masked softmax,
    # no vectorized gather, no fused sdpa
    k_l1, v_l1 = kpool[:, LAYER], vpool[:, LAYER]
    reps = q.shape[1] // k_l1.shape[2]
    outs = []
    for s, seqlen in enumerate(cache_seqlens):
        lo = max(0, seqlen - window) if window else 0
        pages = range(lo // page, -(-seqlen // page))
        k = mx.concatenate([k_l1[int(bt[s, p])] for p in pages], axis=0)
        v = mx.concatenate([v_l1[int(bt[s, p])] for p in pages], axis=0)
        k = mx.repeat(k.transpose(1, 0, 2), reps, axis=0)  # (hq, S, d)
        v = mx.repeat(v.transpose(1, 0, 2), reps, axis=0)
        scores = (q[s] @ k.transpose(0, 2, 1)) * scale  # (hq, 1, S)
        j = mx.arange(k.shape[1]) + lo // page * page
        valid = (j >= lo) & (j < seqlen)
        scores = mx.where(valid[None, :], scores, mx.array(-mx.inf))
        outs.append(mx.softmax(scores, axis=-1) @ v)  # (hq, 1, d)
    return mx.stack(outs)


def paged_decode_call(q, kpool, vpool, bt, cache_seqlens, page, window, scale):
    # Timed path: gather through the block table, then fused sdpa.
    # Windowed cases gather only the window's pages, window bounds stay
    # page-aligned, so gathered position j maps to global key page_lo * page + j.
    seqlen = cache_seqlens[0]
    lo = max(0, seqlen - window) if window else 0
    page_lo, page_hi = lo // page, -(-seqlen // page)
    kg, vg = gather_kv(kpool, vpool, bt, page_lo, page_hi)
    mask = validity_mask(kg.shape[2], page_lo * page, cache_seqlens, window)
    return mx.fast.scaled_dot_product_attention(q, kg, vg, scale=scale, mask=mask)


def check_math():
    # Fused gather+sdpa path against the explicit reference, before any timing:
    # - validates the block-table gather
    # - validates the validity and window masks
    # - validates the installed MLX bool-mask handling
    # Gather ranges are uniform across sequences, so the ragged case checks
    # full-sequence validity while the windowed case runs on uniform seqlens.
    num_seqs, hq, hkv, d, page = 3, 4, 2, 16, 8
    scale = d ** -0.5
    # 2 pages per sequence, every referenced slot real: a -1 slot would
    # silently alias the last page, MLX negative indexing wraps
    kpool = make((7, LAYERS, page, hkv, d))
    vpool = make((7, LAYERS, page, hkv, d))
    bt = block_table(num_seqs, 2)
    q = make((num_seqs, hq, 1, d))
    tol = 5e-2 if DTYPE == mx.bfloat16 else 1e-4
    checks = [([16, 9, 3], 0), ([16, 16, 16], 5)]
    for cache_seqlens, window in checks:
        fused = paged_decode_call(q, kpool, vpool, bt, cache_seqlens, page, window, scale)
        want = ref_paged(q, kpool, vpool, bt, cache_seqlens, page, window, scale)
        err = mx.abs(fused - want).max().item()
        assert err < tol, f"gather+sdpa mismatch (window {window}), max err {err}"


def run_case(label, num_seqs, seqlen, window, batch, rounds):
    pages_per_seq = -(-seqlen // PAGE)
    num_pages = num_seqs * pages_per_seq + 1  # +1 spare page, never referenced
    cache_seqlens = [seqlen] * num_seqs
    scale = DK ** -0.5
    kpool = make((num_pages, LAYERS, PAGE, HKV, DK))
    vpool = make((num_pages, LAYERS, PAGE, HKV, DV))
    bt = block_table(num_seqs, pages_per_seq)
    q = make((num_seqs, HQ, 1, DK))

    def call():
        return paged_decode_call(q, kpool, vpool, bt, cache_seqlens, PAGE, window, scale)

    mx.eval(call())  # warmup
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(rounds):
            mx.eval([call() for _ in range(batch)])
        timings.append(time.perf_counter() - t0)
    med = sorted(timings)[1] / (rounds * batch)
    attended = seqlen - max(0, seqlen - window) if window else seqlen
    flops = 2 * num_seqs * attended * HQ * (DK + DV)
    tflops = flops / med / 1e12
    win_text = str(window) if window else "-"
    print(f"{label:>12} seqs={num_seqs:>3} seqlen={seqlen:>5} win={win_text:>5}  "
          f"median {med * 1e6:>10.1f} us  effective {tflops:>6.2f} TFLOP/s")


check_math()
print(f"Paged SDPA decode MLX bench, B=1 Hq=32 Hkv=8 Dk=Dv=128 page=32 layer=1 of 2 "
      f"{'bf16' if DTYPE == mx.bfloat16 else 'f32'} {mx.device_info()['device_name']},"
      f" batched eval (per-case batch)")
for case in CASES:
    run_case(*case)
