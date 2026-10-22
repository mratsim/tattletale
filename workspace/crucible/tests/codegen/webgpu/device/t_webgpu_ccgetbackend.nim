## Tests backend detection within a kernel.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/t_webgpu_ccgetbackend.nim

import std/strutils
import workspace/crucible

const kernelCode = webgpu:
  proc backendProbe(output: ptr UncheckedArray[int]) {.global.} =
    when ccGetBackend() == ctOpenCL: output[0] = 1
    elif ccGetBackend() == ctVulkan: output[0] = 2
    elif ccGetBackend() == ctMetal: output[0] = 3
    elif ccGetBackend() == ctCuda: output[0] = 4
    else: output[0] = 5

proc runTest() =
  doAssert "(&output)[0] = 5;" in kernelCode,
    "WGSL missing the ctWebGPU dispatch constant (5):\n" & kernelCode
  for other in ["(&output)[0] = 1;", "(&output)[0] = 2;", "(&output)[0] = 3;", "(&output)[0] = 4;"]:
    doAssert other notin kernelCode,
      "a non-WebGPU dispatch branch survived semantic analysis:\n" & kernelCode
  doAssert "when" notin kernelCode,
    "a when construct survived into the emitted source:\n" & kernelCode
  doAssert "ccGetBackend" notin kernelCode,
    "a ccGetBackend call survived into the emitted source:\n" & kernelCode
  echo "  OK — WGSL ccGetBackend emission (dispatch = 5, no when)"

when isMainModule:
  runTest()
