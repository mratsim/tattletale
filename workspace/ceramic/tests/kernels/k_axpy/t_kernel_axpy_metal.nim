## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_axpy/t_kernel_axpy_metal.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_axpy/kernel_axpy_template

proc main =
  var engine = bkMetal.newEngine()
  run_axpy_tests(engine, moduleName = "test_k_axpy_metal")

when isMainModule:
  main()
