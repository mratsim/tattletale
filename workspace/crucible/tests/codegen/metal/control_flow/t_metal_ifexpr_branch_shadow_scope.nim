## Metal: if-expression branch declaring a shadowing local
##
## Test that a branch-local shadow leaves the outer binding readable.
##
## Run:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_ifexpr_branch_shadow_scope.nim

import std/unittest
import workspace/crucible

const branchShadowMsl = metal:
  proc branchShadowKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var x = 7'u32
    let y = (if x < 10'u32:
      (block:
        var x = 3'u32
        x + 1'u32)
    else:
      0'u32)
    output[0] = x
    output[1] = y

proc runTest() =   # private: engines are destroyed at proc return
  suite "Metal - if-expression branch declaring a shadowing local":
    test "a branch-local shadow leaves the outer binding readable":
      var engine = newEngine(runMetal)
      engine.ingest(branchShadowMsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("branchShadowKernel", oDev, ())
      var output: array[2, uint32]
      engine.copyToHostFromDevice(output.toOpenArray(0, 2 - 1), oDev)
      # branch-local x = 3 computes y
      # x is the outer 7 after the expression
      check output[0] == 7
      check output[1] == 4

when isMainModule:
  runTest()
