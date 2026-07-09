## Tests that registerGenericInstOrExternalProc handles
## system.* called as nnkCall on the OpenCL backend.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_call_nim_builtins.nim

import std/[unittest]
import workspace/crucible/src/codegen/cl

template runMulKernel(t: typedesc; expected0, expected1: float32) =
  const k = opencl:
    proc mulKernel(C: ptr UncheckedArray[float32]) {.global.} =
      for i in 0 ..< 2:
        let a = t(i + 1)
        let b = t(1)
        let x = `*`(a, b)
        C[i] = float32(x)
  block:
    var ctx = initOpenCL()
    defer: ctx.shutdown()
    let result = execOpenCL(
      ctx, k, "mulKernel",
      outputBytes = 8,
      inputs = []
    )
    let res = cast[ptr array[2, float32]](result[0].addr)
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
    var ctx = initOpenCL()
    defer: ctx.shutdown()
    let result = execOpenCL(
      ctx, k, "mulKernel",
      outputBytes = 8,
      inputs = []
    )
    let res = cast[ptr array[2, float32]](result[0].addr)
    check res[0] == expected0
    check res[1] == expected1

suite "OpenCL - call nim builtins":

  test "system.`*` (int32) as nnkCall":
    runMulKernel(int32, 1'f32, 2'f32)

  test "system.`*` (float32) as nnkCall":
    runFloatMulKernel(float32, 0.5'f32, 1.0'f32)
