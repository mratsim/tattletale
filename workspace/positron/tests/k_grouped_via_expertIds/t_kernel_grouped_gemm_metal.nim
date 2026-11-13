## Positron grouped_gemm Metal driver: fixture replay, seeded fuzz, graph-capture replay, fp32 and fp16.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/positron/tests/k_grouped_via_expertIds/t_kernel_grouped_gemm_metal.nim

import workspace/crucible
import grouped_gemm_template

proc main =
  var engine = newEngine(runMetal)
  # Metal simdgroup atoms: fp16 and fp32 operands, both accumulate in fp32
  groupedGemmReplayCase(engine, float32, "grouped_aligned_packed_f32")
  groupedGemmReplayCase(engine, float32, "grouped_boundary_cross_f32")
  groupedGemmReplayCase(engine, float32, "grouped_padding_heavy_f32")
  groupedGemmReplayCase(engine, float32, "grouped_wide_f32")
  groupedGemmReplayCase(engine, float16, "grouped_boundary_cross_f16")
  fuzzCase(engine)
  graphCaptureCase(engine)
  echo "[OK] grouped_gemm: fixtures (fp32, fp16), fuzz, graph-capture replay"

when isMainModule:
  main()
