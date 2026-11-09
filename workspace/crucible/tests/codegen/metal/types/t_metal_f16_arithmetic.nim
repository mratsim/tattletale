## Metal: float16 arithmetic lowers to native `half` operations
##
## - `float16` keeps a first-class 16-bit float identity in the IR
## - `a * b + a` over float16 values emits native MSL half arithmetic,
##   never an integer multiply over 16-bit storage
## - readback checks bit-exact against the host fp32 reference
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_f16_arithmetic.nim

import workspace/crucible

proc f16MadKernel(o, a, b: ptr UncheckedArray[float16]) =
  for k in 0 ..< 4:
    o[k] = a[k] * b[k] + a[k]

proc runTest() =
  let aHost = [toFp16(1.5'f32), toFp16(2.25'f32), toFp16(0.5'f32), toFp16(-3.0'f32)]
  let bHost = [toFp16(2.0'f32), toFp16(4.0'f32), toFp16(8.0'f32), toFp16(0.25'f32)]
  var engine = newEngine(runMetal)
  let aDev = engine.allocDevice(float16, 4)
  let bDev = engine.allocDevice(float16, 4)
  let oDev = engine.allocDevice(float16, 4)
  engine.copyToDeviceFromHost(aDev, aHost.toOpenArray(0, 4 - 1))
  engine.copyToDeviceFromHost(bDev, bHost.toOpenArray(0, 4 - 1))
  engine.launchDeviceFn f16MadKernel(oDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (1, 1)
  var res: array[4, float16]
  engine.copyToHostFromDevice(res.toOpenArray(0, 4 - 1), oDev)
  for k in 0 ..< 4:
    # every product and sum here is exact in fp16: the check is bit-exact
    let refVal = toFp16(toFp32(aHost[k]) * toFp32(bHost[k]) + toFp32(aHost[k]))
    doAssert cast[uint16](res[k]) == cast[uint16](refVal),
      "lane " & $k & ": got " & $cast[uint16](res[k]) & ", expected " &
      $cast[uint16](refVal)
  echo "  OK: float16 multiply-add on Metal, 4 lanes, bit-exact vs the fp32 reference"

when isMainModule:
  runTest()
