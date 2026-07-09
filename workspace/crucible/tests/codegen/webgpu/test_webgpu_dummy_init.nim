## Test: WebGPU WGSL codegen for structs with dummy/empty fields
##
## When a struct has no fields (crucible pads it with `char _`),
## the object constructor must emit a valid initializer, not empty braces.
##
## To compile and run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_dummy_init.nim
##
## For full execution (requires libwgpu_native.so):
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible/src/codegen/wgpu

type
  FixMe*[V: static int] = object

const kernelCode = webgpu:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = webgpu:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

suite "WebGPU - dummy-field initializers":
  test "single dummy struct const":
    var buf: array[1, uint32]
    var ctx = initWgpu()
    defer: ctx.shutdown()
    echo kernelCode
    let result = execWgpu(ctx, kernelCode, "dummyKernel", 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1


  test "tuple of dummy structs const":
    var buf: array[1, uint32]
    var ctx = initWgpu()
    defer: ctx.shutdown()
    echo kernelCode2
    let result = execWgpu(ctx, kernelCode2, "dummyKernel", 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1
