## Metal: a coordinate builtin referenced inside a `{.device.}` proc body.
## One threadgroup of 32 threads runs, every slot must hold its lane index.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_device_fn_builtin.nim

import std/unittest
import workspace/crucible

proc fillLane(p: ptr UncheckedArray[uint32]) {.device.} =
  let t = thread_index_in_threadgroup
  p[t] = uint32(t)

const laneIdMsl = metal:
  proc fillLaneKernel(p: ptr UncheckedArray[uint32]) {.global.} =
    fillLane(p)

proc runTest() =
  suite "Metal - coordinate builtin in a device fn":
    test "per-lane values 0..31 written by the device fn":
      var engine = bkMetal.init()
      engine.ingest(laneIdMsl)
      let oDev = engine.allocDevice(32 * sizeof(uint32))
      engine.run<<(grid: (1, 1), blk: (32, 1))>>("fillLaneKernel", oDev, ())
      var res: array[32, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 32 * sizeof(uint32))
      engine.freeDevice(oDev)
      for i in 0 ..< 32:
        check res[i] == uint32(i)

when isMainModule:
  runTest()
