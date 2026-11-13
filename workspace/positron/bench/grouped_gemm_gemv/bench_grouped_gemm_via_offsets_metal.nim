# Run (from the worktree root):
#   nim c -r -d:release --hints:off --warnings:off --outdir:build/bench --nimcache:nimcache/bench \
#     [E M [fp32|fp16]] workspace/positron/bench/grouped_gemm_gemv/bench_grouped_gemm_via_offsets_metal.nim
# E and M select one cell, M is the per-group M_g, no arguments run the full sweep over both dtypes.

import workspace/positron/bench/grouped_gemm_gemv/bench_grouped_gemm_gemv_template

when isMainModule:
  mainGroupedGemmViaOffsets()
