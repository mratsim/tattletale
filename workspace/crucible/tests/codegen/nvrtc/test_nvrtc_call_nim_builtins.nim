## Tests that registerGenericInstOrExternalProc handles:
##   1. system.* as nnkCall for int32, int64, int16, float32, float64, uint32
##   2. User-defined `*` on struct as nnkCall — non-magic pull-in
##
## Self-contained — no ceramic dependency.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_call_nim_builtins.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

template runMulKernel(t: typedesc; expected0, expected1: float32) =
  const k = cuda:
    proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
      for i in 0 ..< 2:
        let a = t(i + 2)
        let b = t(3)
        let x = `*`(a, b)
        C[i] = float32(x)
  block:
    var buf: array[2, float32]
    var nv = initNvrtc(k)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
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

suite "NVRTC - call nim builtins":

  test "system.`*` (int32) as nnkCall":
    runMulKernel(int32, 6'f32, 9'f32)

  test "system.`*` (int64) as nnkCall":
    runMulKernel(int64, 6'f32, 9'f32)

  test "system.`*` (int16) as nnkCall":
    runMulKernel(int16, 6'f32, 9'f32)

  test "system.`*` (float32) as nnkCall":
    runMulKernel(float32, 6'f32, 9'f32)

  test "system.`*` (float64) as nnkCall":
    runMulKernel(float64, 6'f32, 9'f32)

  test "system.`*` (uint32) as nnkCall":
    runMulKernel(uint32, 6'f32, 9'f32)

  test "user-defined `*` on struct as nnkCall":
    var buf: array[2, float32]
    var nv = initNvrtc(wrapperKernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 12'f32
    check buf[1] == 16'f32
