## Test: let-block-RHS with evalOnceAs pattern (Vulkan GLSL)
##
## Verifies that `let L = block: const tmp; tmp` generates valid GLSL.
## The constexpr must be lifted before the let, not inlined into the RHS.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_let_block_rhs.nim

import std/[unittest, strformat]
import workspace/crucible/src/codegen/vk

type
  Int*[V: static int] = object
  Tuple2*[A, B] = object
    f0: A
    f1: B

# Pattern A: direct constructor in let
const kernelDirect = vulkan:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = Tuple2[Int[8], Int[16]]()
    C[0] = 1'u32

# Pattern B: separate const + let
const kernelConstLet = vulkan:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1'u32

# Pattern C: block with const + yield (evalOnceAs pattern)
const kernelBlock = vulkan:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
      tmp
    C[0] = 1'u32

suite "Vulkan - let-block-RHS":
  test "Pattern A — direct tuple let":
    var buf: array[1, uint32]
    var ctx = initVulkan()
    defer: ctx.shutdown()
    echo kernelDirect
    let result = execVulkan(ctx, kernelDirect, "dummyKernel", 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1

  test "Pattern B — const + let":
    var buf: array[1, uint32]
    var ctx = initVulkan()
    defer: ctx.shutdown()
    echo kernelConstLet
    let result = execVulkan(ctx, kernelConstLet, "dummyKernel", 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1

  test "Pattern C — block with const then yield":
    var buf: array[1, uint32]
    var ctx = initVulkan()
    defer: ctx.shutdown()
    echo kernelBlock
    let result = execVulkan(ctx, kernelBlock, "dummyKernel", 4, inputs = [])
    check cast[ptr uint32](result[0].addr)[] == 1
