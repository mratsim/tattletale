## Test: WebGPU scalar by-value marshalling
##
## Covers the ArgBlob negative-size path (runtime/engines/arg_blobs.nim):
## by-value scalar args are flattened as `size = -sizeof(T)` blobs and the
## wgpu engine (runtime/engines/wgpu.nim) turns them into small storage
## buffers. This is the first test that actually RUNS kernels with scalar
## (non-pointer) params through `engine.run` instead of only echoing WGSL.
##
## Scalar inputs bind read-only (`WGPUBufferBindingTypeReadOnlyStorage`) to
## match the WGSL backend, which emits `var<storage, read>` for non-pointer
## params (codegen/targets/wgsl_lang.nim `genGlobal`); ptr inputs stay
## read-write (`var<storage, read_write>` → Storage binding).
##
## A Nim `bool` arg marshals as a 4-byte i32 (arg_blobs.nim `blobOf`), the
## width every shader backend declares: WGSL cannot use bool storage
## variables and emits i32, OpenCL C has no bool and emits int, while
## CUDA/GLSL 1-byte `bool` kernels still read the right value from byte 0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_scalar_marshalling.nim

import std/[unittest, strformat]
import workspace/crucible

# ── Kernels: one scalar param each, echoed into the output ─────────────────

const kernelI32 = webgpu:
  proc scalarI32Kernel(output: ptr UncheckedArray[uint32]; x: int32) {.global.} =
    output[0] = uint32(x)

const kernelU32 = webgpu:
  proc scalarU32Kernel(output: ptr UncheckedArray[uint32]; x: uint32) {.global.} =
    output[0] = x

const kernelF32 = webgpu:
  proc scalarF32Kernel(output: ptr UncheckedArray[float32]; x: float32) {.global.} =
    output[0] = x

const kernelBool = webgpu:
  proc scalarBoolKernel(output: ptr UncheckedArray[uint32]; flag: bool) {.global.} =
    if flag:
      output[0] = 1'u32
    else:
      output[0] = 0'u32

# ── Host side ───────────────────────────────────────────────────────────────

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "WebGPU - scalar by-value marshalling":

    test "int32 by value (4-byte blob)":
      var engine = bkWGSL.init()
      engine.ingest(kernelI32)
      echo kernelI32
      var res: array[1, uint32]
      engine.run("scalarI32Kernel", res, (-42'i32,))
      check res[0] == uint32(-42'i32)

    test "uint32 by value (4-byte blob)":
      var engine = bkWGSL.init()
      engine.ingest(kernelU32)
      echo kernelU32
      var res: array[1, uint32]
      engine.run("scalarU32Kernel", res, (7'u32,))
      check res[0] == 7

    test "float32 by value (4-byte blob)":
      var engine = bkWGSL.init()
      engine.ingest(kernelF32)
      echo kernelF32
      var res: array[1, float32]
      engine.run("scalarF32Kernel", res, (1.5'f32,))
      check res[0] == 1.5'f32

    test "bool by value (widened to i32 on the host)":
      var engine = bkWGSL.init()
      engine.ingest(kernelBool)
      echo kernelBool
      var res: array[1, uint32]
      engine.run("scalarBoolKernel", res, (true,))
      check res[0] == 1
      engine.run("scalarBoolKernel", res, (false,))
      check res[0] == 0

when isMainModule:
  runTest()
