## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_axpy/t_kernel_axpy_vulkan_via_glsl.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_axpy/kernel_axpy_template

proc main =
  var engine = newEngine(runVulkan)
  run_axpy_tests(engine, moduleName = "test_k_axpy_vulkan_via_glsl")

when isMainModule:
  main()
