## OpenCL: `let L = block: const tmp; tmp` (evalOnceAs pattern) generates valid
## OpenCL C, the constexpr lifted before the let, not inlined into the RHS.
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/control_flow/t_opencl_let_block_rhs.nim

import std/[unittest, strformat]
import workspace/crucible

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

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "OpenCL - let-block-RHS":
    test "Pattern A — direct tuple let":
      var engine = bkOpenCL.newEngine()
      engine.ingest(kernelDirect)
      echo kernelDirect
      var buf: array[1, uint32]
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

    test "Pattern B — const + let":
      var engine = bkOpenCL.newEngine()
      engine.ingest(kernelConstLet)
      echo kernelConstLet
      var buf: array[1, uint32]
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

    test "Pattern C — block with const then yield":
      var engine = bkOpenCL.newEngine()
      engine.ingest(kernelBlock)
      echo kernelBlock
      var buf: array[1, uint32]
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("dummyKernel", oDev, ())
      engine.copyToHostFromDevice(addr buf[0], oDev, 1)
      check buf[0] == 1

when isMainModule:
  runTest()
