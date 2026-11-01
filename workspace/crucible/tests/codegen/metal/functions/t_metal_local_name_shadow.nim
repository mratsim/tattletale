## Metal: uniquer-issued local names never collide with user-declared locals.
## Test that a shadowed local whose base name prefixes another local's name
## keeps every declaration distinct and every read back under Nim semantics.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_local_name_shadow.nim

import std/unittest
import workspace/crucible

const shadowMsl = metal:
  proc k(output: ptr UncheckedArray[int32], pick: int32) {.global.} =
    var lhs_val = 100'i32
    var lhs_val_1 = 200'i32
    if pick == 1:
      var lhs_val = 999'i32
      output[0] = lhs_val
    output[1] = lhs_val
    output[2] = lhs_val_1

proc runTest() =
  suite "Metal - local name shadowing":
    test "shadowed local suffixes past a user-declared suffixed name":
      var engine = newEngine(runMetal)
      engine.ingest(shadowMsl)
      let oDev = engine.allocDevice(int32, 3)
      engine.launchKernel("k", oDev, (1'i32,))
      var res: array[3, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 3)
      check res[0] == 999
      check res[1] == 100
      check res[2] == 200

when isMainModule:
  runTest()
