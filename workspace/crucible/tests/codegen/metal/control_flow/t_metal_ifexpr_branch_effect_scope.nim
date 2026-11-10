## if-expression branch effect scope
##
## Test that only the taken branch of an if-expression evaluates.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/red --nimcache:nimcache/red \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_ifexpr_branch_effect_scope.nim

import std/unittest
import workspace/crucible

const ifExprBranchesMsl = metal:
  proc ifExprBranches(o, x: ptr UncheckedArray[int32]) {.global.} =
    var a: int32 = 0
    var b: int32 = 0
    let v = (if x[0] == 0:
               (block: a = 1; 10'i32)
             else:
               (block: b = 1; 20'i32))
    o[0] = v
    o[1] = a
    o[2] = b

proc runTest() =
  suite "Metal - if-expression branch effect scope":

    test "then taken: else branch side effect stays absent":
      var engine = newEngine(runMetal)
      engine.ingest(ifExprBranchesMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 3)
      var res: array[3, int32]
      var x = [0'i32]
      engine.copyToDeviceFromHost(xDev, x.toOpenArray(0, 1 - 1))
      engine.launchKernel("ifExprBranches", oDev, (xDev,))
      engine.copyToHostFromDevice(res.toOpenArray(0, 3 - 1), oDev)
      check res[0] == 10
      check res[1] == 1  # then branch ran
      check res[2] == 0  # else branch did not run

    test "else taken: then branch side effect stays absent":
      var engine = newEngine(runMetal)
      engine.ingest(ifExprBranchesMsl)
      let xDev = engine.allocDevice(int32, 1)
      let oDev = engine.allocDevice(int32, 3)
      var res: array[3, int32]
      var x = [5'i32]
      engine.copyToDeviceFromHost(xDev, x.toOpenArray(0, 1 - 1))
      engine.launchKernel("ifExprBranches", oDev, (xDev,))
      engine.copyToHostFromDevice(res.toOpenArray(0, 3 - 1), oDev)
      check res[0] == 20
      check res[1] == 0  # then branch did not run
      check res[2] == 1  # else branch ran

when isMainModule:
  runTest()
