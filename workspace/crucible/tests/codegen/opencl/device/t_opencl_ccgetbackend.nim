## OpenCL: ccGetBackend detects the compile target inside a kernel body.
## The `when` chain folds at compile time onto the OpenCL dispatch constant,
## the other branches never reach the kernel.
##
## A launched kernel writes the ctOpenCL dispatch value on the OpenCL runtime.
## The test launches on this host, so no compile-signal fallback.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/device/t_opencl_ccgetbackend.nim

import std/unittest
import workspace/crucible

const kernelCode = opencl:
  proc backendProbe(output: ptr UncheckedArray[int32]) {.global.} =
    when ccGetBackend() == ctOpenCL:
      output[0] = 1
    elif ccGetBackend() == ctGlsl:
      output[0] = 2
    elif ccGetBackend() == ctMetal:
      output[0] = 3
    elif ccGetBackend() == ctCuda:
      output[0] = 4
    else:
      output[0] = 5

proc runTest() =
  suite "OpenCL - ccGetBackend compile-target detection":
    test "the launched kernel writes the ctOpenCL dispatch constant":
      var engine = newEngine(runOpenCL)
      engine.ingest(kernelCode)
      var res: array[1, int32] = [0'i32]
      let oDev = engine.allocDevice(int32, 1)
      engine.launchKernel("backendProbe", oDev, ())
      engine.copyToHostFromDevice(res.toOpenArray(0, 1 - 1), oDev)
      check res[0] == 1

when isMainModule:
  runTest()
