## Metal: a self-recursive device proc terminates and runs.
## The macro expansion and the Metal ingest both complete. The kernel reads back the recursion value and the process never hangs.
##
## Run (must SUCCEED, never hang):
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/manual_metal_self_recursion_terminates.nim

import workspace/crucible

const code = metal:
  proc f(a: int32): int32 {.device.} =
    if a > 0:
      result = f(a - 1)
    else:
      result = 0

  proc k(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = f(3'i32)

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(code)
  let oDev = engine.allocDevice(int32, 1)
  engine.launchKernel("k", oDev, ())
  var res: array[1, int32]
  engine.copyToHostFromDevice(res.toOpenArray(0, 1 - 1), oDev)
  doAssert res[0] == 0
  echo "recursion terminated, readback ", res[0]

when isMainModule:
  runTest()
