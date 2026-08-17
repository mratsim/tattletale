## Metal: let-block-RHS with the evalOnceAs pattern. `let L = block:
## const tmp; tmp` must emit valid MSL with the constexpr lifted
## before the let, never inlined into the RHS. Three shapes execute
## on the device:
##   A: direct constructor in the let
##   B: separate const + let
##   C: block with a const then a yield (evalOnceAs)
## Each kernel writes the constexpr value of the let's `f0` field,
## asserted byte-exact after `engine.run()`.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_let_block_rhs.nim

import std/unittest
import workspace/crucible

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
      var res: array[1, uint32]
      engine.run("dummyKernel", res, ())
      check res[0] == 8

    test "Pattern B — const + let":
      var engine = bkMetal.init()
      engine.ingest(kernelConstLet)
      echo kernelConstLet
      var res: array[1, uint32]
      engine.run("dummyKernel", res, ())
      check res[0] == 8

    test "Pattern C — block with const then yield":
      var engine = bkMetal.init()
      engine.ingest(kernelBlock)
      echo kernelBlock
      var res: array[1, uint32]
      engine.run("dummyKernel", res, ())
      check res[0] == 8

when isMainModule:
  runTest()
