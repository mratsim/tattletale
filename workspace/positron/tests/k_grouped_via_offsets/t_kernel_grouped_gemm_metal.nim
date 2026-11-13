## Ceramic grouped_gemm Metal driver: bijection, boundary, wide, edge, fuzz
##
## Test that grouped_gemm covers bijection, boundary crossing, edge cells and fuzz against a float64 reference.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_grouped_gemv_gemm/t_kernel_grouped_gemm_metal.nim

import workspace/crucible
import workspace/positron/tests/k_grouped_via_offsets/grouped_gemv_gemm_template

proc main =
  var engine = newEngine(runMetal)
  mainGroupedGEMM(engine)

when isMainModule:
  main()
