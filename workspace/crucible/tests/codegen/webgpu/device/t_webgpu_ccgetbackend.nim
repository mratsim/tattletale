## WebGPU: ccGetBackend detects the compile target inside a kernel body.
## The `when` chain folds at compile time onto the WebGPU dispatch constant,
## the other branches never reach the kernel.
##
## A launched kernel writes the ctWebGPU dispatch value on the wgpu runtime.
## The test launches on this host, so no compile-signal fallback.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/t_webgpu_ccgetbackend.nim

import std/unittest
import workspace/crucible

const kernelCode = webgpu:
  proc backendProbe(output: ptr UncheckedArray[int32]) {.global.} =
    when ccGetBackend() == ctOpenCL:
      output[0] = 1
    elif ccGetBackend() == ctGlsl:
      output[0] = 2
    elif ccGetBackend() == ctMetal:
      output[0] = 3
    elif ccGetBackend() == ctCuda:
      output[0] = 4
    elif ccGetBackend() == ctWebGPU:
      output[0] = 5
    else:
      output[0] = 9

proc runTest() =
  suite "WebGPU - ccGetBackend compile-target detection":
    test "the launched kernel writes the ctWebGPU dispatch constant":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelCode)
      var res: array[1, int32] = [0'i32]
      let oDev = engine.allocDevice(int32, 1)
      engine.launchKernel("backendProbe", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 5

when isMainModule:
  runTest()
