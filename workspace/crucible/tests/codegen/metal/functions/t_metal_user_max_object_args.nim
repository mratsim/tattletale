## Metal: a user-defined `max` over object values lowers as a harvested proc.
##
## Test that a `metal:` kernel calling a device proc named `max` with object
## arguments harvests the proc body, the call does not emit the MSL builtin
## `max` over struct operands (no matching function).
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_user_max_object_args.nim

import std/unittest
import workspace/crucible

type V2 = object
  x: int32
  y: int32

proc max(a, b: V2): V2 {.device.} =
  result.x = if a.x > b.x: a.x else: b.x
  result.y = if a.y > b.y: a.y else: b.y

const maxObjMsl = metal:
  proc maxObjKernel(o: ptr UncheckedArray[int32]) {.global.} =
    let a = V2(x: 3, y: 10)
    let b = V2(x: 7, y: 2)
    let m = max(a, b)
    o[0] = m.x
    o[1] = m.y

proc runTest() =
  suite "Metal - user-defined max over object arguments":
    test "max(a, b) over objects harvests the proc, both fields elementwise":
      var engine = newEngine(runMetal)
      engine.ingest(maxObjMsl)
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("maxObjKernel", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 7
      check res[1] == 10

when isMainModule:
  runTest()
