## launchDeviceFn applies the callee's default values for omitted arguments
##
## A kernel with defaulted parameters launches with fewer arguments.
## The omitted trailing formals fill from their defaults, both compile-time (static) and runtime parameters.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/runtime/t_launch_device_fn_default_args.nim

import workspace/crucible

proc kernStatic(outP: ptr UncheckedArray[float32], a, b: int, S: static int = 3) =
  outP[0] = float32(S) * float32(a + b)

proc kernRuntime(outP: ptr UncheckedArray[float32], a, b: int, alpha: float32 = 0.5'f32) =
  outP[0] = float32(a + b) + alpha

proc kernBoth(outP: ptr UncheckedArray[float32], a, b: int, S: static int = 3, alpha: float32 = 0.5'f32) =
  outP[0] = float32(S) * (float32(a + b) + alpha)

proc runTest() =
  var engine = newEngine(runMetal)

  block staticDefault:
    let dev = engine.allocDevice(float32, 1)
    engine.launchDeviceFn kernStatic(dev, 10, 4) do:
      grid = (1, 1)
      blk = (1, 1)
    var host: float32
    engine.copyToHostFromDevice(addr host, dev, 1)
    let expected = float32(3) * float32(10 + 4)
    doAssert host == expected,
      "kernStatic: got " & $host & " want " & $expected
    echo "[OK] launchDeviceFn: omitted static parameter fills from its default"

  block runtimeDefault:
    let dev = engine.allocDevice(float32, 1)
    engine.launchDeviceFn kernRuntime(dev, 10, 4) do:
      grid = (1, 1)
      blk = (1, 1)
    var host: float32
    engine.copyToHostFromDevice(addr host, dev, 1)
    let expected = float32(10 + 4) + 0.5'f32
    doAssert host == expected,
      "kernRuntime: got " & $host & " want " & $expected
    echo "[OK] launchDeviceFn: omitted runtime parameter fills from its default"

  block bothDefaults:
    let dev = engine.allocDevice(float32, 1)
    engine.launchDeviceFn kernBoth(dev, 10, 4) do:
      grid = (1, 1)
      blk = (1, 1)
    var host: float32
    engine.copyToHostFromDevice(addr host, dev, 1)
    let expected = float32(3) * (float32(10 + 4) + 0.5'f32)
    doAssert host == expected,
      "kernBoth: got " & $host & " want " & $expected
    echo "[OK] launchDeviceFn: trailing static and runtime defaults both fill"

when isMainModule:
  runTest()
