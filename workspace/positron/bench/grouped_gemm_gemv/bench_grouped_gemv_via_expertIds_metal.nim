# Run (from the worktree root):
#   nim c -r -d:release -d:groupedDtype=f32|f16 \
#     --outdir:build/bench workspace/positron/bench/grouped_gemm_gemv/bench_grouped_gemv_via_expertIds_metal.nim

import workspace/positron/bench/grouped_gemm_gemv/bench_grouped_gemm_gemv_template

when isMainModule:
  mainGroupedGemvViaExpertIds()
