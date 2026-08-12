## Test: let-block-RHS with evalOnceAs pattern (Vulkan GLSL)
##
## Verifies that `let L = block: const tmp; tmp` generates valid GLSL.
## The constexpr must be lifted before the let, not inlined into the RHS.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_let_block_rhs.nim

import std/[unittest, strformat]
import workspace/crucible

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
    var engine = bkVulkan.init()
    engine.ingest(kernelDirect)
    echo kernelDirect
    var buf: array[1, uint32]
    engine.run("dummyKernel", buf, ())
    check buf[0] == 1

  test "Pattern B — const + let":
    var engine = bkVulkan.init()
    engine.ingest(kernelConstLet)
    echo kernelConstLet
    var buf: array[1, uint32]
    engine.run("dummyKernel", buf, ())
    check buf[0] == 1

  test "Pattern C — block with const then yield":
    var engine = bkVulkan.init()
    engine.ingest(kernelBlock)
    echo kernelBlock
    var buf: array[1, uint32]
    engine.run("dummyKernel", buf, ())
    check buf[0] == 1
