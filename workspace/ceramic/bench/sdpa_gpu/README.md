# SDPA GPU bench, Metal Nim vs MLX python

Nim side: `bench_scaled_dot_product_attention_metal.nim`, tile SDPA kernel over Metal.
Python side: `bench_scaled_dot_product_attention_mlx.py`, `mx.fast.scaled_dot_product_attention`.

## Run

    nim c -r -d:release \
      --outdir:build/bench workspace/ceramic/bench/sdpa_gpu/bench_scaled_dot_product_attention_metal.nim

    uv run python workspace/ceramic/bench/sdpa_gpu/bench_scaled_dot_product_attention_mlx.py

## Comparison procedure

- identical case list on both sides: (L, S, causal) pairs, B=1 Hq=32 Hkv=8 Dk=Dv=128 f32
- Nim: iters attention passes inside one kernel launch, median of 3 launches, per-pass time = total / iters
- MLX: batched eval, 20 calls per mx.eval, median of 3 timed rounds, per-call time = round / 20
- launch overhead sits inside the measured window on both sides, the python
  script asserts its fused path against explicit masked math on two small
  shapes before timing

## TFLOPS convention

Effective flops count only the key tiles a query tile visits, the model
`bench_scaled_dot_product_attention_metal.nim` reports:

- kv tile grid: ceil(S / NT), tile shape per dtype: f32 (MT, NT) = (32, 16),
  bf16 (16, 32)
- causal: the kernel's bottom-right band with offset = S - L. A query tile
  at qBase = q*MT + (S - L) visits max(0, min(kvTiles, ceil((qBase + MT) / NT))) tiles
- full: every query tile visits kvTiles
- per visited tile: QK costs 2*MT*NT*Dk, PV costs 2*MT*NT*Dv, totals scale
  by Hq * B, GQA head sharing does not discount per-head query work

## Causal alignment for L < S

- the kernel contract is hi(q) = min(S, q + offset + 1) with offset = S - L
  (bottom-right band), so a query row touches keys up to q + S - L
- both benches time full S keys with the bottom-right band, the same op
- the "causal" mask string in MLX is lower-right aligned for L != S.
  check_math verifies the installed MLX against the explicit lower-right
  mask and fails loudly if that ever changes


# Paged decode bench, Metal Nim vs MLX python

Nim side: `bench_paged_sdpa_decode_metal.nim`, paged decode over Metal through `k_tile_sdpa_paged.nim`.
Python side: `bench_paged_sdpa_decode_mlx.py`, a block-table gather plus `mx.fast.scaled_dot_product_attention`.

## Run

    nim c -r -d:release \
      -d:sdpaDtype=f32 \
      --outdir:build/bench workspace/ceramic/bench/sdpa_gpu/bench_paged_sdpa_decode_metal.nim

`sdpaDtype` also takes `bf16`, pass one value per run.

    uv run python workspace/ceramic/bench/sdpa_gpu/bench_paged_sdpa_decode_mlx.py [bf16]

## Shape and cases

- B=1 Hq=32 Hkv=8 Dk=Dv=128, page size 32, schedule tile (MT, NT) = (32, 32)
- one kv tile per page (the paged test suite's schedule), pool layer 1 of 2
- cases: 8 seqs x 1024 (L2-resident contrast), 32 x 8192, 64 x 8192, and 32 x 8192 with window 1024
- block tables fill page ids sequentially, exactly full, the pool keeps one spare page

MLX pays one block-table gather per timed call: the Nim kernel reads pool pages in place.
The python script checks its gather+sdpa path against an explicit page-walk reference before timing.

## GB/s convention

Decode attention is bandwidth-bound in principle: effective GB/s is the primary metric,
effective TFLOP/s reported alongside for convention parity with the base bench.
Both sides use the same byte and flop models:

- effective bytes: sum over sequences of attended_seqlen * Hkv * (Dk + Dv) *
  sizeof(elem), attended_seqlen = seqlen - windowLo, windowLo = max(0, seqlen - window)
- effective flops: 2 * sum over sequences of attended_seqlen * Hq * (Dk + Dv)

Truth cases need an L2-busting K+V working set of at least 2 GB K+V f32:

- once the working set fits cache, in-kernel iteration loops measure L2 bandwidth,
  which exceeds the 540 GB/s M4 Max DRAM peak
- a line whose computed bandwidth exceeds the peak prints INVALID (the measurement failed)

Measured on M4 Max the (32, 32) schedule computes 32 query entries for one valid decode entry,
reported GB/s runs far under the peak and the l2-resident case runs higher than the truth cases.
