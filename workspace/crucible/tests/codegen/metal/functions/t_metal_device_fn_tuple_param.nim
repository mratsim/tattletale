## Metal: device proc with a runtime tuple-typed parameter.
##
## Test that a `metal:` kernel calls a device proc with a plain `(int, int)` tuple
## parameter and both elements reach the body as runtime values:
## a tuple-typed formal is not a bound static value.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_device_fn_tuple_param.nim

import std/unittest
import workspace/crucible

proc sumPair(p: ptr UncheckedArray[uint32], t: (int, int), base: uint32) {.device.} =
  p[0] = base + uint32(t[0])
  p[1] = base + uint32(t[1])

const tupleParamMsl = metal:
  proc tupleParamKernel(o: ptr UncheckedArray[uint32]) {.global.} =
    sumPair(o, (3, 4), 100'u32)

proc runTest() =
  suite "Metal - device proc with a runtime tuple parameter":
    test "tuple argument lowers as a runtime value, both elements reach the body":
      var engine = newEngine(runMetal)
      engine.ingest(tupleParamMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("tupleParamKernel", oDev, ())
      var res: array[2, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 103'u32
      check res[1] == 104'u32

when isMainModule:
  runTest()
