## Metal: let-block-RHS with the evalOnceAs pattern, a block-RHS `let` emits
## its lifted constexpr before the let, never inlined into the RHS.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_let_block_rhs.nim


import workspace/crucible

import std/unittest

type
  Int*[V: static int] = object
  Tuple2*[A, B] = object
    f0: A
    f1: B

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

# Pattern A: direct constructor in let
const kernelDirect = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = Tuple2[Int[8], Int[16]]()
    C[0] = uint32(toIntVal L.f0)

# Pattern B: separate const + let
const kernelConstLet = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = uint32(toIntVal L.f0)

# Pattern C: block with const + yield (evalOnceAs pattern)
const kernelBlock = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
      tmp
    C[0] = uint32(toIntVal L.f0)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - let-block-RHS":
    test "Pattern A — direct tuple let":
      var engine = bkMetal.init()
      engine.ingest(kernelDirect)
      echo kernelDirect
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("dummyKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 8

    test "Pattern B — const + let":
      var engine = bkMetal.init()
      engine.ingest(kernelConstLet)
      echo kernelConstLet
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("dummyKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 8

    test "Pattern C — block with const then yield":
      var engine = bkMetal.init()
      engine.ingest(kernelBlock)
      echo kernelBlock
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("dummyKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 8

when isMainModule:
  runTest()
