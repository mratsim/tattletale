## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_axpy/t_kernel_axpy_vulkan_via_spirv.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_axpy/kernel_axpy_template

proc main =
  var engine = newEngine(runVulkan)
  run_axpy_tests(engine, moduleName = "test_k_axpy_vulkan_via_spirv")
  # the spirv leg: the ingested kernel artifact is binary words, not a text source,
  # getArtifact stays empty (a glslang/dxc ingest always leaves source text)
  doAssert engine.getArtifact().len == 0,
    "the kernel compiled through a text route, the Vulkan bridge emits words directly"

when isMainModule:
  main()
