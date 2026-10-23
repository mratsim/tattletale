## Tests backend detection within a kernel.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/device/t_opencl_ccgetbackend.nim

import std/strutils
import workspace/crucible

const kernelCode = opencl:
  proc backendProbe(output: ptr UncheckedArray[int]) {.global.} =
    when ccGetBackend() == ctOpenCL: output[0] = 1
    elif ccGetBackend() == ctGlsl:
      output[0] = 2
    elif ccGetBackend() == ctMetal: output[0] = 3
    elif ccGetBackend() == ctCuda: output[0] = 4
    else: output[0] = 5

proc runTest() =
  doAssert "output[0] = 1;" in kernelCode,
    "OpenCL missing the ctOpenCL dispatch constant (1):\n" & kernelCode
  for other in ["output[0] = 2;", "output[0] = 3;", "output[0] = 4;", "output[0] = 5;"]:
    doAssert other notin kernelCode,
      "a non-OpenCL dispatch branch survived semantic analysis:\n" & kernelCode
  echo "  OK — OpenCL ccGetBackend emission (dispatch = 1, no when)"

when isMainModule:
  runTest()
