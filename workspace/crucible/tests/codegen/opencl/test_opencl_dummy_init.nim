## Test: OpenCL codegen for structs with dummy/empty fields
##
## When a struct has no fields (crucible pads it with `char _`),
## the object constructor must emit a valid initializer, not empty braces.
##
## To compile and run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_dummy_init.nim

import std/[unittest, strformat]
import workspace/crucible

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
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode)
    echo "===="
    echo kernelCode
    echo "===="
    var buf: array[1, uint32]
    engine.run("dummyKernel", buf, ())
    check buf[0] == 1

  test "tuple of dummy structs const":
    var engine = bkOpenCL.init()
    engine.ingest(kernelCode2)
    echo "===="
    echo kernelCode2
    echo "===="
    var buf: array[1, uint32]
    engine.run("dummyKernel", buf, ())
    check buf[0] == 1
