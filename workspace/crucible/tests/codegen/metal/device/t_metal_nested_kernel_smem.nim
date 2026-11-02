## Metal: a `{.smem.}` local nested in a kernel body stays kernel-scope storage.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_nested_kernel_smem.nim

import std/unittest
import workspace/crucible

const nestedSmemMsl = metal:
  proc k(output: ptr UncheckedArray[uint32], flag: int32) {.global.} =
    if flag == 1:
      var scratch {.smem.}: array[4, uint32]
      scratch[0] = 7'u32
      scratch[1] = 8'u32
      output[0] = scratch[0]
      output[1] = scratch[1]
    else:
      output[0] = 0'u32
      output[1] = 0'u32

proc runTest() =
  suite "Metal - nested kernel smem":
    test "a nested {.smem.} local allocates and reads back":
      var engine = newEngine(runMetal)
      engine.ingest(nestedSmemMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("k", oDev, (1'i32,))
      var res: array[2, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res == [7'u32, 8]
      engine.launchKernel("k", oDev, (0'i32,))
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res == [0'u32, 0]

when isMainModule:
  runTest()
