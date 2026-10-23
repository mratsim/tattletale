## Run:
##   nim c -r -d:crucibleVulkanCodegen=hlsl --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_axpy/t_kernel_axpy_vulkan_via_hlsl.nim

import std/strutils
import workspace/crucible
import workspace/ceramic/tests/kernels/k_axpy/kernel_axpy_template

proc main =
  var engine = newEngine(runVulkan)
  run_axpy_tests(engine, moduleName = "test_k_axpy_vulkan_via_hlsl")
  # the HLSL leg: the ingested kernel source must be HLSL ([numthreads), not GLSL (#version)
  doAssert not engine.getArtifact().startsWith("#version"),
    "the kernel compiled through the GLSL route, the via_hlsl leg needs -d:crucibleVulkanCodegen=hlsl"

when isMainModule:
  main()
