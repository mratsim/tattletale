## Metal: an alias and its target type resolve to one struct.
## `PairAlias` names `Pair` through a Nim alias and both names appear in one kernel.
## One struct is registered for both. The kernel launches and reads back through the unified type.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_alias_struct_unification.nim

import std/unittest
import workspace/crucible

type
  Pair = object
    x: int32
    y: int32
  PairAlias = Pair

const aliasMsl = metal:
  proc k(outA: ptr UncheckedArray[int32]) {.global.} =
    var p = Pair(x: 3'i32, y: 4'i32)
    var q: PairAlias
    q.x = p.x + 1
    q.y = p.y + 1
    outA[0] = q.x
    outA[1] = q.y

proc runTest() =
  suite "Metal - alias struct unification":
    test "alias and its target resolve to one struct, readback correct":
      var engine = newEngine(runMetal)
      engine.ingest(aliasMsl)
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel("k", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 2 - 1), oDev)
      check res[0] == 4
      check res[1] == 5

when isMainModule:
  runTest()
