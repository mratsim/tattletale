## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_sum/t_kernel_sum_axis_metal.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_sum/kernel_sum_template

proc main =
  var engine = newEngine(runMetal)
  run_sum_axis_tests(engine, moduleName = "test_k_sum_axis_metal")

when isMainModule:
  main()
