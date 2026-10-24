## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_axpy/t_kernel_axpy_vulkan_via_glsl.nim

import workspace/crucible
import workspace/ceramic/tests/kernels/k_axpy/kernel_axpy_template

proc main =
  var engine = newEngine(runVulkan)
  run_axpy_tests(engine, moduleName = "test_k_axpy_vulkan_via_glsl",
                 lang = llGlsl)
  # the glsl route ingests printed text, the words route leaves the artifact empty
  # The artifact identity is the ingested source
  doAssert engine.getArtifact().len != 0,
    "glsl route: expected the printed GLSL source as the engine artifact"

when isMainModule:
  main()
