## Metal: the element conversion builtins lower to the native cast
##
## - `toFp16`/`toBf16`/`toFp32`/`toBf32` have host bodies, each
##   marked with the `{.builtin.}` pragma
## - Device calls lower to the backend native conversion, never into
##   inlined host bit-twiddling
## - f16 and bf16 run in separate kernels here, one kernel using
##   both types collapses them on this Metal (CSE into the half cast)
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_numeric_conv_to.nim

import std/[strutils, unittest]
import workspace/crucible

proc f16Kernel(o: ptr UncheckedArray[float32], i: ptr UncheckedArray[float32]) =
  for k in 0 ..< 4:
    var h: float16 = toFp16(i[k])
    o[k] = toFp32(h)

proc bf16Kernel(o: ptr UncheckedArray[float32], i: ptr UncheckedArray[float32]) =
  for k in 0 ..< 4:
    var b: bfloat16 = toBf16(i[k])
    o[k] = toBf32(b)

proc roundtripF16(engine: MetalEngine, inp: array[4, float32]): array[4, float32] =
  let iDev = engine.allocDevice(float32, 4)
  let oDev = engine.allocDevice(float32, 4)
  engine.copyToDeviceFromHost(iDev, unsafeAddr inp[0], 4)
  engine.launchDeviceFn f16Kernel(oDev, iDev) do:
    grid = (1, 1)
    blk = (1, 1)
  engine.copyToHostFromDevice(addr result[0], oDev, 4)

proc roundtripBf16(engine: MetalEngine, inp: array[4, float32]): array[4, float32] =
  let iDev = engine.allocDevice(float32, 4)
  let oDev = engine.allocDevice(float32, 4)
  engine.copyToDeviceFromHost(iDev, unsafeAddr inp[0], 4)
  engine.launchDeviceFn bf16Kernel(oDev, iDev) do:
    grid = (1, 1)
    blk = (1, 1)
  engine.copyToHostFromDevice(addr result[0], oDev, 4)

proc runTest() =
  suite "Metal - element conversion builtins":
    test "the host bodies round-trip exact values":
      for x in [1.0'f32, 2.5'f32, 0.5'f32, 100.0'f32, 7.0'f32]:
        check toFp32(toFp16(x)) == x
        check toBf32(toBf16(x)) == x
    test "f16 round-trips the values on device":
      var engine = newEngine(runMetal)
      let inp = [1.0'f32, 3.14159'f32, 65504.0'f32, 0.0001'f32]
      let res = engine.roundtripF16(inp)
      for k in 0 ..< 4:
        check abs(res[k] - toFp32(toFp16(inp[k]))) < 1e-6
    test "bf16 round-trips the values on device":
      var engine = newEngine(runMetal)
      let inp = [1.0'f32, 3.14159'f32, 65504.0'f32, 0.0001'f32]
      let res = engine.roundtripBf16(inp)
      for k in 0 ..< 4:
        check abs(res[k] - toBf32(toBf16(inp[k]))) < 1e-6

when isMainModule:
  runTest()
