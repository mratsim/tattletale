## evalOnceAs-generated constexpr values resolve inside let RHS across three
## declaration shapes: direct tuple, const + let, and a const-yielding block.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/test_nvrtc_let_block_rhs.nim

import std/[unittest]
import workspace/crucible

type
  Int*[V: static int] = object
    discard

type
  Tuple2*[A, B] = object
    f0: A
    f1: B

# ── Pattern A: direct tuple ──
const kernelDirect = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = Tuple2[Int[8], Int[16]]()
    C[0] = 1'u32

# ── Pattern B: const + let (simulates evalOnceAs) ──
const kernelConstLet = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1'u32

# ── Pattern C: let with block that defines const then yields value ──
#   This is what evalOnceAs actually expands to under the hood.
const kernelBlock = cuda:
  proc kernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
      tmp
    C[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Crucible - TensorView codegen repro":
    test "Pattern A — direct tuple let":
      var output: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(kernelDirect)
      let oA = engine.allocDevice(sizeof(output))
      engine.launchKernel<<(1, 1)>>("kernel", oA, ())
      engine.copyToHostFromDevice(addr output[0], oA, sizeof(output))
      engine.freeDevice(oA)
      check output[0] == 1

    test "Pattern B — const + let":
      let code = kernelConstLet
      var output: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(code)
      let oB = engine.allocDevice(sizeof(output))
      engine.launchKernel<<(1, 1)>>("kernel", oB, ())
      engine.copyToHostFromDevice(addr output[0], oB, sizeof(output))
      engine.freeDevice(oB)
      check output[0] == 1

    test "Pattern C — block with const then yield":
      let code = kernelBlock
      var output: array[1, uint32]
      var engine = bkCuda.init()
      engine.ingest(code)
      let oC = engine.allocDevice(sizeof(output))
      engine.launchKernel<<(1, 1)>>("kernel", oC, ())
      engine.copyToHostFromDevice(addr output[0], oC, sizeof(output))
      engine.freeDevice(oC)
      check output[0] == 1

when isMainModule:
  runTest()
