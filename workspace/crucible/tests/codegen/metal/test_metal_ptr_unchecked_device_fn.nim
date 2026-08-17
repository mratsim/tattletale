## Metal: device fn taking `ptr UncheckedArray` + a runtime length.
## `fillArray` writes `p[i] = i + 10` through a `ptr UncheckedArray[uint32]`
## param passed from the kernel. Runs through `engine.run()` and asserts the
## output byte-exact.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_ptr_unchecked_device_fn.nim

import std/unittest
import workspace/crucible

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
      var res: array[8, uint32]
      engine.run("fillKernel", res, ())
      for i in 0 ..< 8:
        check res[i] == uint32(i + 10)

when isMainModule:
  runTest()
