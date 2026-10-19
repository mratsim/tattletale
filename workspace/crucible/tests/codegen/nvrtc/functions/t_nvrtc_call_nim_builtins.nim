## registerGenericInstOrExternalProc: system `*` as nnkCall plus user-defined `*` on a struct
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_call_nim_builtins.nim
import std/[unittest]
import workspace/crucible

template runIntKernel(t: typedesc; expected0, expected1: float32) =
  const k = cuda:
    proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
      for i in 0 ..< 2:
        let a = t(i + 2)
        let b = t(3)
        let x = `*`(a, b)
        C[i] = float32(x)
  block:
    var buf: array[2, float32]
    var engine = bkCuda.newEngine()
    engine.ingest(k)
    let oDev = engine.allocDevice(float32, buf.len)
    engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
    engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
    check buf[0] == expected0
    check buf[1] == expected1

template runFloatKernel(t: typedesc; expected0, expected1: float32) =
  const k = cuda:
    proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
      for i in 0 ..< 2:
        let a = t(i + 1) * t(0.25'f32)
        let b = t(2.0'f32)
        let x = `*`(a, b)
        C[i] = float32(x)


  echo "==== k ===="
  echo k
  echo "==========="

  block:
    var buf: array[2, float32]
    var engine = bkCuda.newEngine()
    engine.ingest(k)
    let oDev = engine.allocDevice(float32, buf.len)
    engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
    engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
    check buf[0] == expected0
    check buf[1] == expected1

type Wrapper = object
  val: int

proc `*`(a, b: Wrapper): Wrapper =
  Wrapper(val: a.val * b.val)

const wrapperKernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    for i in 0 ..< 2:
      let a = Wrapper(val: i + 3)
      let b = Wrapper(val: 4)
      let c = `*`(a, b)
      C[i] = float32(c.val)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "NVRTC - call nim builtins":

    test "system.`*` (int32) as nnkCall":
      runIntKernel(int32, 6'f32, 9'f32)

    test "system.`*` (int64) as nnkCall":
      runIntKernel(int64, 6'f32, 9'f32)

    test "system.`*` (int16) as nnkCall":
      runIntKernel(int16, 6'f32, 9'f32)

    test "system.`*` (float32) as nnkCall":
      runFloatKernel(float32, 0.5'f32, 1.0'f32)

    test "system.`*` (float64) as nnkCall":
      runFloatKernel(float64, 0.5'f32, 1.0'f32)

    test "system.`*` (uint32) as nnkCall":
      runIntKernel(uint32, 6'f32, 9'f32)

    test "user-defined `*` on struct as nnkCall":
      var buf: array[2, float32]
      var engine = bkCuda.newEngine()
      engine.ingest(wrapperKernel)
      let oDev = engine.allocDevice(float32, buf.len)
      engine.launchKernel<<(1, 1)>>("kernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
      check buf[0] == 12'f32
      check buf[1] == 16'f32

when isMainModule:
  runTest()
