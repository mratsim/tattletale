## launchDeviceFn on Metal: DeviceMem arguments, grouped parameters, compile-time arguments written into the kernel.
##
## Test that DeviceMem arguments pass through launchDeviceFn to pointer parameters.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn.nim

import workspace/crucible

proc mix(scale: static int, outP: ptr UncheckedArray[float32], a, b: int, alpha: float32) =
  outP[0] = float32(scale) * (float32(a + b) + alpha)

# no compile-time parameter: every argument crosses as a launch parameter
proc blend(outP: ptr UncheckedArray[float32], a, b: int, alpha: float32) =
  outP[0] = float32(a * b) - alpha

proc runTest() =
  var engine = bkMetal.init()

  block groupedWithStatic:
    let dev = engine.allocDevice(float32, 1)
    engine.launchDeviceFn mix(3, dev, 10, 4, 0.5'f32) do:
      grid = (1, 1)
      blk = (1, 1)
    var host: float32
    engine.copyToHostFromDevice(addr host, dev, 1)
    let expected = float32(3) * (float32(10 + 4) + 0.5'f32)
    doAssert host == expected,
      "mix: got " & $host & " want " & $expected
    echo "[OK] launchDeviceFn: grouped parameters, compile-time parameter, int/float/DeviceMem values"

  block allValues:
    let dev = engine.allocDevice(float32, 1)
    engine.launchDeviceFn blend(dev, 6, 7, 2.25'f32) do:
      grid = (1, 1)
      blk = (1, 1)
    var host: float32
    engine.copyToHostFromDevice(addr host, dev, 1)
    let expected = float32(6 * 7) - 2.25'f32
    doAssert host == expected,
      "blend: got " & $host & " want " & $expected
    echo "[OK] launchDeviceFn: every parameter is a runtime parameter"

runTest()
