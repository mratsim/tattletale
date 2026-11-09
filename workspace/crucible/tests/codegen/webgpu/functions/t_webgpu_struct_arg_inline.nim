## WebGPU: pointer-holding struct arguments inline at device-function call sites.
##
## Test that a kernel calling a device function with pointer-holding struct
## arguments inlines the callee, and the dissolved field reads and writes
## compute the expected results.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/functions/t_webgpu_struct_arg_inline.nim

import workspace/crucible

type AxpyView = object
  p: ptr UncheckedArray[float32]
  n: int32

proc axpyBody(y: AxpyView, alpha: float32, x: AxpyView) =
  for i in 0 ..< y.n:
    y.p[i] = alpha * x.p[i] + y.p[i]

const axpyWgsl = webgpu:
  proc axpyStruct(y: AxpyView, alpha: float32, x: AxpyView) {.global.} =
    axpyBody(y, alpha, x)

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(axpyWgsl)
  let xDev = engine.allocDevice(float32, 4)
  let yDev = engine.allocDevice(float32, 4)
  var xData: array[4, float32] = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
  var yData: array[4, float32] = [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32]
  engine.copyToDeviceFromHost(xDev, xData.toOpenArray(0, 4 - 1))
  engine.copyToDeviceFromHost(yDev, yData.toOpenArray(0, 4 - 1))
  engine.launchKernel("axpyStruct", (yDev, 4'i32),
                      (2.5'f32, (xDev, 4'i32)))
  var got: array[4, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 4 - 1), yDev)
  for i in 0 ..< 4:
    let expected = 2.5'f32 * xData[i] + yData[i]
    doAssert got[i] == expected,
      "struct arg: element " & $i & " got " & $got[i] & " want " & $expected

  echo "[OK] pointer-holding struct arguments inline at the device-function call site"

when isMainModule:
  runTest()
