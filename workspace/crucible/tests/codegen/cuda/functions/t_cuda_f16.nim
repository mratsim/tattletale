## CUDA: float16 arithmetic and the fp16 conversion builtins, launched and read back
##
## Test that `float16` values compute natively on CUDA, and that `toFp16`/`toFp32`
## lower to the `__float2half`/`__half2float` conversion builtins.
##
## - runs on a CUDA workstation (libcuda + libnvrtc, NVRTC ships cuda_fp16.h)
## - without a CUDA device the test refuses loudly, a wrong fp16 lowering is invisible at compile time
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_f16.nim

import workspace/crucible

const f16MadCuda = cuda:
  proc f16MadKernel(o, a, b: ptr UncheckedArray[float16]) {.global.} =
    let i = int(thread_position_in_grid.x)
    o[i] = a[i] * b[i] + a[i]

const f16RoundTripCuda = cuda:
  proc f16RoundTripKernel(o, i: ptr UncheckedArray[float32]) {.global.} =
    let k = int(thread_position_in_grid.x)
    var h: float16 = toFp16(i[k])
    o[k] = toFp32(h)

proc runTest() =
  var engine = newEngine(runCuda)

  # every product and sum here is exact in fp16: the check is bit-exact
  engine.ingest(f16MadCuda)
  let aHost = [toFp16(1.5'f32), toFp16(2.25'f32), toFp16(0.5'f32), toFp16(-3.0'f32)]
  let bHost = [toFp16(2.0'f32), toFp16(4.0'f32), toFp16(8.0'f32), toFp16(0.25'f32)]
  let aDev = engine.allocDevice(float16, 4)
  let bDev = engine.allocDevice(float16, 4)
  let oDev = engine.allocDevice(float16, 4)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], 4)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], 4)
  engine.launchKernel<<(1, 4)>>("f16MadKernel", oDev, (aDev, bDev))
  var mad: array[4, float16]
  engine.copyToHostFromDevice(addr mad[0], oDev, 4)
  for k in 0 ..< 4:
    let refVal = toFp16(toFp32(aHost[k]) * toFp32(bHost[k]) + toFp32(aHost[k]))
    doAssert cast[uint16](mad[k]) == cast[uint16](refVal),
      "lane " & $k & ": got " & $cast[uint16](mad[k]) & ", expected " &
      $cast[uint16](refVal)
  echo "  OK: float16 multiply-add, 4 lanes, bit-exact vs the fp32 reference"

  # f32 -> f16 -> f32 through the conversion builtins, each device lane equals its host-side conversion of the same input
  engine.ingest(f16RoundTripCuda)
  let inHost = [1.0'f32, 3.14159'f32, 65504.0'f32, 0.0001'f32]
  let iDev = engine.allocDevice(float32, 4)
  let rDev = engine.allocDevice(float32, 4)
  engine.copyToDeviceFromHost(iDev, addr inHost[0], 4)
  engine.launchKernel<<(1, 4)>>("f16RoundTripKernel", rDev, (iDev,))
  var res: array[4, float32]
  engine.copyToHostFromDevice(addr res[0], rDev, 4)
  for k in 0 ..< 4:
    let refVal = toFp32(toFp16(inHost[k]))
    doAssert abs(res[k] - refVal) < 1e-6'f32,
      "lane " & $k & ": got " & $res[k] & ", expected " & $refVal
  echo "  OK: f32 -> f16 -> f32 round trip through the conversion builtins, 4 lanes"

runTest()
echo "CUDA float16 arithmetic and conversions: GREEN"
