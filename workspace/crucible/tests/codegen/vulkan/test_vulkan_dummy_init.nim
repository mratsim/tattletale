## Test: Vulkan GLSL codegen for structs with dummy/empty fields
##
## When a struct has no fields (crucible pads it with `char _`),
## the object constructor must emit a valid initializer, not empty braces.
##
## To compile and run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible/src/codegen/vk

type
  FixMe*[V: static int] = object

const kernelCode = vulkan:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = vulkan:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

suite "Vulkan - dummy-field initializers":
  test "single dummy struct const":
    var buf: array[1, uint32]
    var ctx = initVulkan()
    defer: ctx.shutdown()
    echo kernelCode
    let result = execVulkan(ctx, kernelCode, "dummyKernel", 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1

  test "tuple of dummy structs const":
    var buf: array[1, uint32]
    var ctx = initVulkan()
    defer: ctx.shutdown()
    echo kernelCode2
    let result = execVulkan(ctx, kernelCode2, "dummyKernel", 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1
