## Metal: device fn taking `ptr UncheckedArray` + a runtime length.
## `fillArray` writes `p[i] = i + 10` through a `ptr UncheckedArray[uint32]`
## param passed from the kernel.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_ptr_unchecked_device_fn.nim


import workspace/crucible

import std/unittest

const fillMsl = metal:
  proc fillArray(p: ptr UncheckedArray[uint32]; n: uint32) {.device.} =
    for i in 0 ..< n:
      p[i] = i + 10'u32
  proc fillKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    fillArray(output, 8)

proc runTest() =
  suite "Metal - ptr UncheckedArray device fn":
    test "fillArray writes i + 10 into the kernel output buffer":
      var engine = bkMetal.init()
      engine.ingest(fillMsl)
      let oDev = engine.allocDevice(uint32, 8)
      engine.launchKernel("fillKernel", oDev, ())
      var res: array[8, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 8)
      engine.freeDevice(oDev)
      for i in 0 ..< 8:
        check res[i] == uint32(i + 10)

when isMainModule:
  runTest()
