## Metal: a combined `{.unroll, emit(...).}` pragma marks the loop and emits the text
##
## Test that a loop whose body states `{.unroll, emit(...).}` runs the emitted
## statement and computes the loop results, so the unroll marker does not drop
## the other pragma effects
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_unroll_emit_combined.nim

import workspace/crucible

const code = metal:
  proc unrollEmitKernel(output: ptr UncheckedArray[float32]) {.global.} =
    for i in 0 ..< 2:
      {.unroll, emit("output[2] = 7.0;").}
      output[i] = float32(i)

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(code)
  let oDev = engine.allocDevice(float32, 3)
  engine.launchKernel("unrollEmitKernel", oDev, ())
  var res: array[3, float32]
  engine.copyToHostFromDevice(addr res[0], oDev, 3)
  doAssert res[0] == 0.0'f32, "output[0] = " & $res[0]
  doAssert res[1] == 1.0'f32, "output[1] = " & $res[1]
  doAssert res[2] == 7.0'f32, "output[2] = " & $res[2]
  echo "  OK - t_metal_unroll_emit_combined"

when isMainModule:
  runTest()
