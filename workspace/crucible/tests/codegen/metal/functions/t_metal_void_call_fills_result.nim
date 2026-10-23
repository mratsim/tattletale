## Metal: value-returning device fn whose last statement is a void call
## filling `result` through a var param. The result legalization pass must
## not wrap the void call in `result = ...`.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_void_call_fills_result.nim

import workspace/crucible

import std/unittest

const sqMsl = metal:
  proc fillResult(dst: var float32, x: float32) {.device.} =
    dst = x * x
  proc sqVia(x: float32): float32 {.device.} =
    fillResult(result, x)
  proc sqKernel(output: ptr UncheckedArray[float32], x: float32) {.global.} =
    output[thread_index_in_threadgroup] = sqVia(x)

proc runTest() =
  suite "Metal - device fn result filled by a void call":
    test "A device function using the implicit result variable returns its value":
      var engine = newEngine(runMetal)
      engine.ingest(sqMsl)
      let oDev = engine.allocDevice(float32, 4)
      engine.launchKernel<<(grid: (1, 1), blk: (4, 1))>>("sqKernel", oDev, (3.5'f32,))
      var res: array[4, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      for i in 0 ..< 4:
        check res[i] == 12.25'f32

when isMainModule:
  runTest()
