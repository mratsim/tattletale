## WebGPU: float16 arithmetic under the `enable f16` extension
##
## - `float16` keeps a first-class `f16` identity via the shader-f16 device feature, requested at runtime when exposed
## - `a * b + a` over float16 values emits native WGSL f16 arithmetic
## - readback checks bit-exact against the host fp32 reference
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/types/t_webgpu_f16_arithmetic.nim

import workspace/crucible

const f16MadWgsl = webgpu:
  proc f16MadKernel(o, a, b: ptr UncheckedArray[float16]) {.global, workgroup: (4, 1, 1).} =
    let i = int(thread_position_in_grid.x)
    o[i] = a[i] * b[i] + a[i]

proc runTest() =
  var engine = newEngine(runWebGPU)
  engine.ingest(f16MadWgsl)
  # every product and sum here is exact in fp16: the check is bit-exact
  let aHost = [toFp16(1.5'f32), toFp16(2.25'f32), toFp16(0.5'f32), toFp16(-3.0'f32)]
  let bHost = [toFp16(2.0'f32), toFp16(4.0'f32), toFp16(8.0'f32), toFp16(0.25'f32)]
  let aDev = engine.allocDevice(float16, 4)
  let bDev = engine.allocDevice(float16, 4)
  let oDev = engine.allocDevice(float16, 4)
  engine.copyToDeviceFromHost(aDev, aHost.toOpenArray(0, 4 - 1))
  engine.copyToDeviceFromHost(bDev, bHost.toOpenArray(0, 4 - 1))
  engine.launchKernel("f16MadKernel", oDev, (aDev, bDev))
  var res: array[4, float16]
  engine.copyToHostFromDevice(res.toOpenArray(0, 4 - 1), oDev)
  for k in 0 ..< 4:
    let refVal = toFp16(toFp32(aHost[k]) * toFp32(bHost[k]) + toFp32(aHost[k]))
    doAssert cast[uint16](res[k]) == cast[uint16](refVal),
      "lane " & $k & ": got " & $cast[uint16](res[k]) & ", expected " &
      $cast[uint16](refVal)
  echo "  OK: float16 multiply-add on WebGPU, 4 lanes, bit-exact vs the fp32 reference"

when isMainModule:
  runTest()
