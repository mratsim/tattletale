## CUDA: bfloat16 arithmetic and the bf16 conversion builtins, launched and read back
##
## Test that `bfloat16` values compute natively on CUDA, and that `toBf16`/`toBf32`
## lower to the `__float2bfloat16`/`__bfloat162float` conversion builtins.
##
## - runs on a CUDA workstation (libcuda + libnvrtc, NVRTC ships cuda_bf16.h)
## - without a CUDA device the test refuses loudly, a wrong bf16 lowering is invisible at compile time
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_bf16.nim

import workspace/crucible

const bf16MadCuda = cuda:
  proc bf16MadKernel(o, a, b: ptr UncheckedArray[bfloat16]) {.global.} =
    let i = int(thread_position_in_grid.x)
    o[i] = a[i] * b[i] + a[i]

const bf16RoundTripCuda = cuda:
  proc bf16RoundTripKernel(o, i: ptr UncheckedArray[float32]) {.global.} =
    let k = int(thread_position_in_grid.x)
    var b: bfloat16 = toBf16(i[k])
    o[k] = toBf32(b)

proc runTest() =
  var engine = newEngine(runCuda)

  # every product and sum here is exact in bf16: the check is bit-exact
  engine.ingest(bf16MadCuda)
  let aHost = [toBf16(1.5'f32), toBf16(2.25'f32), toBf16(0.5'f32), toBf16(-3.0'f32)]
  let bHost = [toBf16(2.0'f32), toBf16(4.0'f32), toBf16(8.0'f32), toBf16(0.25'f32)]
  let aDev = engine.allocDevice(bfloat16, 4)
  let bDev = engine.allocDevice(bfloat16, 4)
  let oDev = engine.allocDevice(bfloat16, 4)
  engine.copyToDeviceFromHost(aDev, addr aHost[0], 4)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], 4)
  engine.launchKernel<<(1, 4)>>("bf16MadKernel", oDev, (aDev, bDev))
  var mad: array[4, bfloat16]
  engine.copyToHostFromDevice(addr mad[0], oDev, 4)
  for k in 0 ..< 4:
    let refVal = toBf16(toBf32(aHost[k]) * toBf32(bHost[k]) + toBf32(aHost[k]))
    doAssert cast[uint16](mad[k]) == cast[uint16](refVal),
      "lane " & $k & ": got " & $cast[uint16](mad[k]) & ", expected " &
      $cast[uint16](refVal)
  echo "  OK: bfloat16 multiply-add, 4 lanes, bit-exact vs the fp32 reference"

  # f32 -> bf16 -> f32 through the conversion builtins, each device lane equals its host-side conversion of the same input
  engine.ingest(bf16RoundTripCuda)
  let inHost = [1.0'f32, 3.14159'f32, 3.3895e38'f32, 0.0001'f32]
  let iDev = engine.allocDevice(float32, 4)
  let rDev = engine.allocDevice(float32, 4)
  engine.copyToDeviceFromHost(iDev, addr inHost[0], 4)
  engine.launchKernel<<(1, 4)>>("bf16RoundTripKernel", rDev, (iDev,))
  var res: array[4, float32]
  engine.copyToHostFromDevice(addr res[0], rDev, 4)
  for k in 0 ..< 4:
    let refVal = toBf32(toBf16(inHost[k]))
    doAssert abs(res[k] - refVal) <= abs(refVal) * 1e-6'f32,
      "lane " & $k & ": got " & $res[k] & ", expected " & $refVal
  echo "  OK: f32 -> bf16 -> f32 round trip through the conversion builtins, 4 lanes"

runTest()
echo "CUDA bfloat16 arithmetic and conversions: GREEN"
