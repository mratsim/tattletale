## Ceramic grouped_gemv Metal driver: tier rule, bijection, edge, ragged N, undersized grid, fuzz
##
## Test that grouped_gemv covers the tier rule, bijection, edge, ragged N, undersized grid and fuzz against a float64 reference.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_grouped_gemv_gemm/t_kernel_grouped_gemv_metal.nim

import workspace/crucible
import workspace/positron/tests/k_grouped_via_offsets/grouped_gemv_gemm_template

proc main =
  var engine = newEngine(runMetal)
  mainGroupedGEMV(engine)

when isMainModule:
  main()
