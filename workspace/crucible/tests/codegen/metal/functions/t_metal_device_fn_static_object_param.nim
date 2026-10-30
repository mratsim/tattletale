## Metal: device proc with a bound `static` object parameter.
##
## Test that a `metal:` kernel calls a device proc whose `static` object
## formal is bound at the call site and the bound fields evaluate
## at Nim compile time.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_device_fn_static_object_param.nim

import std/unittest
import workspace/crucible

type MiniAtom = object
  k: int32
  tag: int32

func miniScale(mma: static MiniAtom, c: var array[4, int32]) {.inline.} =
  # static record param fields evaluate at Nim compile time
  c[0] = c[0] + mma.k
  c[1] = c[1] + mma.tag

const staticObjMsl = metal:
  proc staticObjKernel(o: ptr UncheckedArray[int32]) {.global.} =
    var c = [0'i32, 0'i32, 0'i32, 0'i32]
    miniScale(MiniAtom(k: 8'i32, tag: 3'i32), c)
    o[0] = c[0]
    o[1] = c[1]

proc runTest() =
  suite "Metal - device proc with a bound static object parameter":
    test "bound static object formal erases, its fields arrive as literals":
      var engine = newEngine(runMetal)
      engine.ingest(staticObjMsl)
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("staticObjKernel", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 8
      check res[1] == 3

when isMainModule:
  runTest()
