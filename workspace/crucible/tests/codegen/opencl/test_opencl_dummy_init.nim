## Test: OpenCL codegen for structs with dummy/empty fields
##
## When a struct has no fields (crucible pads it with `char _`),
## the object constructor must emit a valid initializer, not empty braces.
##
## To compile and run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible/src/codegen/cl

type
  FixMe*[V: static int] = object

const kernelCode = opencl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = opencl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

suite "OpenCL - dummy-field initializers":
  test "single dummy struct const":
    var buf: array[1, uint32]
    var ctx = initOpenCL()
    defer: ctx.shutdown()
    echo "===="
    echo kernelCode
    echo "===="
    let result = execOpenCL(ctx, kernelCode, "dummyKernel",
      outputBytes = 4,
      inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1

  test "tuple of dummy structs const":
    var buf: array[1, uint32]
    var ctx = initOpenCL()
    defer: ctx.shutdown()
    echo "===="
    echo kernelCode2
    echo "===="
    let result = execOpenCL(ctx, kernelCode2, "dummyKernel",
      outputBytes = 4,
      inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1
