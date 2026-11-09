## Metal: min/max ternary idioms, dead bindings, same-type conversions
##
## Test that a kernel with a min/max ternary idiom, a dead call binding,
## and a same-type `float32` conversion computes correct results.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_optimization_deletions.nim

import workspace/crucible

import std/unittest

const floatCode = metal:
  proc stamp(p: ptr UncheckedArray[float32]): float32 {.device.} =
    p[4] = 7.0'f32
    1.0'f32
  proc floatProbe(output: ptr UncheckedArray[float32]) {.global.} =
    let a = output[0]
    let b = output[1]
    # NaN-guarded min idiom over floats
    let m = if (a <= b) or (a != a): a else: b
    # bound to a call whose write is the observable effect, never read
    let dead = stamp(output)
    output[2] = m
    # conversion to the value's own type
    output[3] = float32(m)

const intCode = metal:
  proc intProbe(output: ptr UncheckedArray[int32]) {.global.} =
    let a = output[0]
    let b = output[1]
    let lo = if a <= b: a else: b
    let hi = if a <= b: b else: a
    output[2] = lo
    output[3] = hi

proc runTest() =
  suite "Metal - optimization-stage deletions":
    test "min idiom, dead call binding, same-type conversion":
      var engine = newEngine(runMetal)
      engine.ingest(floatCode)
      var inputs: array[2, float32] = [-2.5'f32, 1.5'f32]
      let oDev = engine.allocDevice(float32, 5)
      engine.copyToDeviceFromHost(oDev, inputs.toOpenArray(0, 2 - 1))
      engine.launchKernel("floatProbe", oDev, ())
      var res: array[5, float32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 5 - 1), oDev)
      check res[2] == -2.5'f32
      check res[3] == -2.5'f32
      check res[4] == 7.0'f32
    test "int min and max ternary idioms":
      var engine = newEngine(runMetal)
      engine.ingest(intCode)
      var inputs: array[2, int32] = [9'i32, -4'i32]
      let oDev = engine.allocDevice(int32, 4)
      engine.copyToDeviceFromHost(oDev, inputs.toOpenArray(0, 2 - 1))
      engine.launchKernel("intProbe", oDev, ())
      var res: array[4, int32]
      engine.copyToHostFromDevice(res.toOpenArray(0, 4 - 1), oDev)
      check res[2] == -4'i32
      check res[3] == 9'i32

when isMainModule:
  runTest()
