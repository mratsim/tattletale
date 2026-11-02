## OpenCL: `{.smem.}` local with an initializer reads back the actual values.
##
## Test that the initializer lowers to explicit `__local` stores.
## OpenCL C forbids the initializer form and honors runtime writes.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/device/t_opencl_smem_init.nim

import std/unittest
import workspace/crucible

const smemInitCl = opencl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[4, uint32] = [1'u32, 2, 3, 4]
    output[0] = scratch[0]
    output[1] = scratch[1]
    output[2] = scratch[2]
    output[3] = scratch[3]

proc runTest() =
  suite "OpenCL - {.smem.} initializer rewrite":
    test "__local storage reads back the initialized values":
      var engine = newEngine(runOpenCL)
      engine.ingest(smemInitCl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("k", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res == [1'u32, 2, 3, 4]

when isMainModule:
  runTest()
