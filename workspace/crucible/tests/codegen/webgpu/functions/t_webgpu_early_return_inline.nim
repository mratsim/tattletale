## WebGPU: value-position calls inline across ordinary early returns.
##
## Test that a value-position call of a device fn with a one-branch
## conditional return and a pointer-holding struct argument inlines,
## computing the early-returned and falling-through values.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/functions/t_webgpu_early_return_inline.nim

import workspace/crucible

type AxpyView = object
  p: ptr UncheckedArray[float32]
  n: int32

proc pick(v: AxpyView, i: int32): float32 =
  if i < 0:
    return 0.0'f32
  v.p[i]

const earlyReturnWgsl = webgpu:
  proc earlyReturnKernel(output: ptr UncheckedArray[float32], v: AxpyView) {.global.} =
    output[0] = pick(v, -1'i32)
    output[1] = pick(v, 2'i32)

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(earlyReturnWgsl)
  let oDev = engine.allocDevice(float32, 2)
  let vDev = engine.allocDevice(float32, 4)
  var vData: array[4, float32] = [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32]
  engine.copyToDeviceFromHost(vDev, vData.toOpenArray(0, 4 - 1))
  engine.launchKernel("earlyReturnKernel", oDev, (vDev, 4'i32))
  var got: array[2, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 2 - 1), oDev)
  doAssert got[0] == 0.0'f32,
    "early return: got " & $got[0] & ", want 0.0"
  doAssert got[1] == 30.0'f32,
    "fallthrough: got " & $got[1] & ", want 30.0"

when isMainModule:
  runTest()
  echo "[OK] value-position calls inline across ordinary early returns"
