## Test: crucible CUDA codegen for structs with dummy/empty fields
##
## When a struct has no fields (crucible pads it with `char _`),
## the object constructor must emit a valid initializer, not empty braces.
## The NVRTC compiler validates the generated code.

import std/[unittest, strformat]
import workspace/crucible/src/codegen/nvrtc

type
  FixMe*[V: static int] = object

const kernelCode = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const kernelCode2 = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (FixMe[1](), FixMe[8]())
    C[0] = 1'u32

suite "CUDA - dummy-field initializers":
  test "single dummy struct const":
    var buf: array[1, uint32]
    var nv = initNvrtc(kernelCode)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 1

  test "tuple of dummy structs const":
    var buf: array[1, uint32]
    var nv = initNvrtc(kernelCode2)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 1
