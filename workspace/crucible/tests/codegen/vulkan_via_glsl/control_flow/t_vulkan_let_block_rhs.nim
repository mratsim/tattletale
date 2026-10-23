## Vulkan: let-block-RHS with the evalOnceAs pattern (GLSL).
## `let L = block: const tmp; tmp` must lift the constexpr before the let, never inline it into the RHS.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/control_flow/t_vulkan_let_block_rhs.nim

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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Vulkan - let-block-RHS":
    test "Pattern A — direct tuple let":
      var engine = bkVulkan.newEngine()
      engine.ingest(kernelDirect)
      echo kernelDirect
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      var buf: array[1, uint32]
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

    test "Pattern B — const + let":
      var engine = bkVulkan.newEngine()
      engine.ingest(kernelConstLet)
      echo kernelConstLet
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      var buf: array[1, uint32]
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

    test "Pattern C — block with const then yield":
      var engine = bkVulkan.newEngine()
      engine.ingest(kernelBlock)
      echo kernelBlock
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      var buf: array[1, uint32]
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

when isMainModule:
  runTest()
