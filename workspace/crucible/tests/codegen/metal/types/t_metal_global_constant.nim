## Globals live in the constant address space
##
## Test that a top-level var emits as a read-only program-scope constant
## that a kernel reads at run time.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_global_constant.nim

import std/unittest
import workspace/crucible

const globalConstantMsl = metal:
  var counter: int32 = 42
  proc bump(C: ptr UncheckedArray[int32]) {.global.} =
    C[0] = counter

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  test "a kernel reads the program-scope constant":
    var engine = newEngine(runMetal)
    engine.ingest(globalConstantMsl)
    let oDev = engine.allocDevice(int32, 1)
    # Sentinel setup: a host copy writes 0 before the kernel overwrites
    # it with the counter value 42.
    let zero: int32 = 0
    engine.copyToDeviceFromHost(oDev, addr zero, 1)
    engine.launchKernel("bump", oDev, ())
    var res: array[1, int32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    check res[0] == 42

when isMainModule:
  runTest()
