## Metal: forceinline device fn ending in an explicit `return expr`.
##
## Test that the spliced call destination receives the returned value.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_inline_return_value.nim

import std/unittest
import workspace/crucible

const inlineRetMsl = metal:
  proc addv(a, b: int32): int32 {.device, forceinline.} =
    return a + b

  proc scaleKernel(output: ptr UncheckedArray[int32]) {.global.} =
    output[0] = addv(3, 4)
    let r = addv(10, 20)
    output[1] = r

proc runTest() =
  suite "Metal - forceinline call splices the returned value":
    test "the call destination receives the value of a trailing return expr":
      var engine = newEngine(runMetal)
      engine.ingest(inlineRetMsl)
      let oDev = engine.allocDevice(int32, 2)
      engine.launchKernel("scaleKernel", oDev, ())
      var res: array[2, int32]
      engine.copyToHostFromDevice(addr res[0], oDev, 2)
      check res[0] == 7
      check res[1] == 30

when isMainModule:
  runTest()
