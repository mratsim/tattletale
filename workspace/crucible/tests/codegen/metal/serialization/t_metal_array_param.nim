## Metal: a kernel parameter of array type.
##
## Test that an `array[4, float32]` kernel parameter launches and the kernel
## reads the array values, the emitted `constant float a[4]` binding covered
## by the consumer itself.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_array_param.nim

import workspace/crucible

const arrayParamMetal = metal:
  proc arrKernel(o: ptr UncheckedArray[float32], a: array[4, float32]) {.global.} =
    o[0] = a[0] + a[3]

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(arrayParamMetal)
  let oDev = engine.allocDevice(float32, 1)
  engine.launchKernel("arrKernel", oDev, ([1.5'f32, 2.0'f32, 3.5'f32, 4.0'f32], ()))
  var res: array[1, float32]
  engine.copyToHostFromDevice(res.toOpenArray(0, 1 - 1), oDev)
  doAssert res[0] == 5.5'f32
  echo "  OK — the array parameter emits `constant float a[4]` and the kernel reads it"

when isMainModule:
  runTest()
