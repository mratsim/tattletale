# SDPA GPU bench, Metal Nim vs MLX python

Nim side: `bench_scaled_dot_product_attention_metal.nim`, tile SDPA kernel over Metal.
Python side: `bench_scaled_dot_product_attention_mlx.py`, `mx.fast.scaled_dot_product_attention`.

## Run

    nim c -r -d:release --maxLoopIterationsVM:100000000 \
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
