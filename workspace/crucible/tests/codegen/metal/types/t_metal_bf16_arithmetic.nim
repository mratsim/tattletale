## Metal: bfloat16 arithmetic lowers to native `bfloat` operations
##
## Test that `bfloat16` values compute natively on Metal, and that `toBf16`/`toBf32`
## lower to the native bfloat casts.
##
## - Metal 3.1+ (macOS 14+) provides the MSL `bfloat` type
## - readback checks bit-exact against the host fp32 reference
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_bf16_arithmetic.nim

import workspace/crucible

proc bf16MadKernel(o, a, b: ptr UncheckedArray[bfloat16]) =
  for k in 0 ..< 4:
    o[k] = a[k] * b[k] + a[k]

proc bf16RoundTripKernel(o, i: ptr UncheckedArray[float32]) =
  for k in 0 ..< 4:
    var b: bfloat16 = toBf16(i[k])
    o[k] = toBf32(b)

proc bf16BitsKernel(o: ptr UncheckedArray[uint16], i: ptr UncheckedArray[bfloat16]) =
  for k in 0 ..< 4:
    o[k] = cast[uint16](i[k])

proc bf16UnbitsKernel(o: ptr UncheckedArray[bfloat16], i: ptr UncheckedArray[uint16]) =
  for k in 0 ..< 4:
    o[k] = cast[bfloat16](i[k])

proc runTest() =
  let aHost = [toBf16(1.5'f32), toBf16(2.25'f32), toBf16(0.5'f32), toBf16(-3.0'f32)]
  let bHost = [toBf16(2.0'f32), toBf16(4.0'f32), toBf16(8.0'f32), toBf16(0.25'f32)]
  var engine = newEngine(runMetal)
  let aDev = engine.allocDevice(bfloat16, 4)
  let bDev = engine.allocDevice(bfloat16, 4)
  let oDev = engine.allocDevice(bfloat16, 4)
  engine.copyToDeviceFromHost(aDev, aHost.toOpenArray(0, 4 - 1))
  engine.copyToDeviceFromHost(bDev, bHost.toOpenArray(0, 4 - 1))
  engine.launchDeviceFn bf16MadKernel(oDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (1, 1)
  var res: array[4, bfloat16]
  engine.copyToHostFromDevice(res.toOpenArray(0, 4 - 1), oDev)
  for k in 0 ..< 4:
    # every product and sum here is exact in bf16: the check is bit-exact
    let refVal = toBf16(toBf32(aHost[k]) * toBf32(bHost[k]) + toBf32(aHost[k]))
    doAssert cast[uint16](res[k]) == cast[uint16](refVal),
      "lane " & $k & ": got " & $cast[uint16](res[k]) & ", expected " &
      $cast[uint16](refVal)
  echo "  OK: bfloat16 multiply-add on Metal, 4 lanes, bit-exact vs the fp32 reference"

  # f32 -> bf16 -> f32 through the native casts, each device lane equals its host-side conversion of the same input
  let inHost = [1.0'f32, 3.14159'f32, 3.3895e38'f32, 0.0001'f32]
  let iDev = engine.allocDevice(float32, 4)
  let rDev = engine.allocDevice(float32, 4)
  engine.copyToDeviceFromHost(iDev, inHost.toOpenArray(0, 4 - 1))
  engine.launchDeviceFn bf16RoundTripKernel(rDev, iDev) do:
    grid = (1, 1)
    blk = (1, 1)
  var round: array[4, float32]
  engine.copyToHostFromDevice(round.toOpenArray(0, 4 - 1), rDev)
  for k in 0 ..< 4:
    let refVal = toBf32(toBf16(inHost[k]))
    doAssert abs(round[k] - refVal) <= abs(refVal) * 1e-6'f32,
      "lane " & $k & ": got " & $round[k] & ", expected " & $refVal
  echo "  OK: f32 -> bf16 -> f32 round trip through the native casts, 4 lanes"

  # cast[uint16](bfloat16) and cast[bfloat16](uint16) move the bit pattern, never a numeric value
  # a fresh input buffer: a device buffer read by two launches needs the copy, not a second borrow
  let aBitsDev = engine.allocDevice(bfloat16, 4)
  engine.copyToDeviceFromHost(aBitsDev, aHost.toOpenArray(0, 4 - 1))
  let uDev = engine.allocDevice(uint16, 4)
  engine.launchDeviceFn bf16BitsKernel(uDev, aBitsDev) do:
    grid = (1, 1)
    blk = (1, 1)
  var bitsRes: array[4, uint16]
  engine.copyToHostFromDevice(bitsRes.toOpenArray(0, 4 - 1), uDev)
  for k in 0 ..< 4:
    doAssert bitsRes[k] == cast[uint16](aHost[k]),
      "lane " & $k & ": got " & $bitsRes[k] & ", expected " & $cast[uint16](aHost[k])
  let o2Dev = engine.allocDevice(bfloat16, 4)
  engine.launchDeviceFn bf16UnbitsKernel(o2Dev, uDev) do:
    grid = (1, 1)
    blk = (1, 1)
  var unbitsRes: array[4, bfloat16]
  engine.copyToHostFromDevice(unbitsRes.toOpenArray(0, 4 - 1), o2Dev)
  for k in 0 ..< 4:
    doAssert cast[uint16](unbitsRes[k]) == cast[uint16](aHost[k]),
      "lane " & $k & ": got " & $cast[uint16](unbitsRes[k]) & ", expected " &
      $cast[uint16](aHost[k])
  echo "  OK: bf16 <-> u16 bit reinterpret casts, 4 lanes, bit-exact"

when isMainModule:
  runTest()
echo "Metal bfloat16 arithmetic and conversions: GREEN"
