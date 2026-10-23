## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_sum/t_kernel_sum_full_vulkan.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_sum/kernel_sum_template

proc main =
  var engine = newEngine(runVulkan)
  run_sum_full_tests(engine, moduleName = "test_k_sum_full_vulkan")

when isMainModule:
  main()
