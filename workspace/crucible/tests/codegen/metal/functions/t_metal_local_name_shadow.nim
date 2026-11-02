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

const globalsMsl = metal:
  var x_1 = 555'i32
  proc k(output: ptr UncheckedArray[int32]) {.global.} =
    var x = 7'i32
    if true:
      var x = 8'i32
      output[0] = x
    output[1] = x
    output[2] = x_1

const resultMsl = metal:
  proc f(c: int32): int32 {.device.} =
    var result = c * 2
    result
  proc k(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = f(5'i32)

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

    test "program-scope names are immune to uniquer suffixes":
      var engine = newEngine(runMetal)
      engine.ingest(globalsMsl)
      let oDev = engine.allocDevice(int32, 3)
      engine.launchKernel("k", oDev, ())
      var res: array[3, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 3)
      check res == [8'i32, 7, 555]

    test "a non-void device proc may declare its own result":
      var engine = newEngine(runMetal)
      engine.ingest(resultMsl)
      let oDev = engine.allocDevice(int32, 1)
      engine.launchKernel("k", oDev, ())
      var res: array[1, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 10

when isMainModule:
  runTest()
