## OpenCL: system.* called as nnkCall goes through registerGenericInstOrExternalProc.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/test_opencl_call_nim_builtins.nim

import std/[unittest]
import workspace/crucible

template runMulKernel(t: typedesc; expected0, expected1: float32) =
  const k = opencl:
    proc mulKernel(C: ptr UncheckedArray[float32]) {.global.} =
      for i in 0 ..< 2:
        let a = t(i + 1)
        let b = t(1)
        let x = `*`(a, b)
        C[i] = float32(x)
  block:
    var engine = bkOpenCL.init()
    engine.ingest(k)
    var res: array[2, float32]
    let oDev = engine.allocDevice(2 * sizeof(float32))
    engine.run("mulKernel", oDev, ())
    engine.copyToHostFromDevice(addr res[0], oDev, 2 * sizeof(float32))
    engine.freeDevice(oDev)
    check res[0] == expected0
    check res[1] == expected1

template runFloatMulKernel(t: typedesc; expected0, expected1: float32) =
  const k = opencl:
    proc mulKernel(C: ptr UncheckedArray[float32]) {.global.} =
      for i in 0 ..< 2:
        let a = t(i + 1) * t(0.25'f32)
        let b = t(2.0'f32)
        let x = `*`(a, b)
        C[i] = float32(x)
  block:
    var engine = bkOpenCL.init()
    engine.ingest(k)
    var res: array[2, float32]
    let oDev = engine.allocDevice(2 * sizeof(float32))
    engine.run("mulKernel", oDev, ())
    engine.copyToHostFromDevice(addr res[0], oDev, 2 * sizeof(float32))
    engine.freeDevice(oDev)
    check res[0] == expected0
    check res[1] == expected1

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "OpenCL - call nim builtins":

    test "system.`*` (int32) as nnkCall":
      runMulKernel(int32, 1'f32, 2'f32)

    test "system.`*` (float32) as nnkCall":
      runFloatMulKernel(float32, 0.5'f32, 1.0'f32)

when isMainModule:
  runTest()
