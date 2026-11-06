## Metal: an if-expression branch declaring a local named branch_val
##
## Test that the temp-name seeding sees a local declared inside a branch's
## statement-expression: the if-expression's value reads 11, not the local's name.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_ifexpr_branch_temp_name.nim

import std/unittest
import workspace/crucible

const ifexprTempNameMsl = metal:
  proc ifexprTempName(output: ptr UncheckedArray[int32]) {.global.} =
    var c: int32 = 1
    let y = (if c > 0:
      (block:
        var branch_val: int32 = 7
        11'i32)
    else:
      3'i32)
    output[0] = y

proc runTest() =
  suite "Metal - if-expression branch local versus result temp name":

    test "the branch value reads 11, not the branch local":
      var engine = newEngine(runMetal)
      engine.ingest(ifexprTempNameMsl)
      let oDev = engine.allocDevice(int32, 1)
      engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("ifexprTempName", oDev, ())
      var res: array[1, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 11

when isMainModule:
  runTest()
