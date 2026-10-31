## Metal: `{.inline.}` device function kept and emitted with the target qualifier.
##
## Test that a non-kernel `{.inline.}` proc that reads no coordinate builtin
## compiles into the module and the launched kernel reads back the exact sums.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_inline_hint.nim

import std/unittest
import workspace/crucible

const inlineHintMsl = metal:
  proc addv(a, b: int32): int32 {.device, inline.} =
    result = a + b

  proc scaleKernel(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = addv(3, 4)
    output[1] = addv(30, 12)

proc runTest() =
  suite "Metal - inline device function emission":
    test "the launched kernel reads back the exact sums":
      var engine = newEngine(runMetal)
      engine.ingest(inlineHintMsl)
      let oDev = engine.allocDevice(int32, 2)
      # sentinel init: the kernel must overwrite every slot, a stale zero
      # would hide a missing write
      let sentinels: array[2, int32] = [-7, -7]
      engine.copyToDeviceFromHost(oDev, addr sentinels[0], 2)
      engine.launchKernel("scaleKernel", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 7
      check res[1] == 42

when isMainModule:
  runTest()
