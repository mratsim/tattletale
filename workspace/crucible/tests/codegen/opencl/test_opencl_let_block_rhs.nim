## Test: let-block-RHS with evalOnceAs pattern (OpenCL)
##
## Verifies that `let L = block: const tmp; tmp` generates valid OpenCL C.
## The constexpr must be lifted before the let, not inlined into the RHS.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_let_block_rhs.nim

import std/[unittest, strformat]
import workspace/crucible/src/codegen/cl

type
  Int*[V: static int] = object
  Tuple2*[A, B] = object
    f0: A
    f1: B

# Pattern A: direct constructor in let
const kernelDirect = opencl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = Tuple2[Int[8], Int[16]]()
    C[0] = 1'u32

# Pattern B: separate const + let
const kernelConstLet = opencl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1'u32

# Pattern C: block with const + yield (evalOnceAs pattern)
const kernelBlock = opencl:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
      tmp
    C[0] = 1'u32

suite "OpenCL - let-block-RHS":
  test "Pattern A — direct tuple let":
    var buf: array[1, uint32]
    var ctx = initOpenCL()
    defer: ctx.shutdown()
    echo kernelDirect
    let result = execOpenCL(ctx, kernelDirect, "dummyKernel",
      outputBytes = 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1

  test "Pattern B — const + let":
    var buf: array[1, uint32]
    var ctx = initOpenCL()
    defer: ctx.shutdown()
    echo kernelConstLet
    let result = execOpenCL(ctx, kernelConstLet, "dummyKernel",
      outputBytes = 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1

  test "Pattern C — block with const then yield":
    var buf: array[1, uint32]
    var ctx = initOpenCL()
    defer: ctx.shutdown()
    echo kernelBlock
    let result = execOpenCL(ctx, kernelBlock, "dummyKernel",
      outputBytes = 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1
