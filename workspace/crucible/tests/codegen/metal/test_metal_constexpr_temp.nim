## Metal: constexpr temporaries leaking into expression slots.
## The liftConstexprFrom pass hoists a constexpr declaration out of an expression slot
## (gpuVar.vInit, gpuDot.dParent, binop operands) into a preceding statement,
## because `constexpr` is a declaration, not an expression value. Six patterns
## execute on the device:
##   A: constexpr tuple in a let RHS
##   B: constexpr Int values in arithmetic
##   C: template with a block `{ const tmp; yield tmp }`
##   D: let-tuple bracket access
##   E: block with two constexprs and tuple field access
##   F: constexpr tuple field access (tup[0] as gpuDot.dParent)
## Every kernel writes a compile-time-derived value that is asserted
## byte-exact after `engine.run()`.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_constexpr_temp.nim

import std/unittest
import workspace/crucible

type
  Int*[V: static int] = object
  Tuple2*[A, B] = object
    f0: A
    f1: B

template `+`*[A, B: static int](a: Int[A], b: Int[B]): Int[A + B] = Int[A + B]()
template `*`*[A, B: static int](a: Int[A], b: Int[B]): Int[A * B] = Int[A * B]()
template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

# Pattern A
const kernelA = metal:
  proc testA(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = uint32(toIntVal L.f0)

# Pattern B
const kernelB = metal:
  proc testB(C: ptr UncheckedArray[uint32]) {.global.} =
    const a {.genSym.} = Int[8]()
    const b {.genSym.} = Int[16]()
    let x = Int[0]() + a * b
    C[0] = uint32(toIntVal x)

# Pattern C
template wrapConst(a, b: untyped): untyped =
  block:
    const tmp {.genSym.} = Tuple2[typeof(a), typeof(b)](f0: a, f1: b)
    tmp

const kernelC = metal:
  proc testC(C: ptr UncheckedArray[uint32]) {.global.} =
    let pair = wrapConst(Int[8](), Int[16]())
    C[0] = uint32(toIntVal pair.f0)

# Pattern D
const kernelD = metal:
  proc testD(C: ptr UncheckedArray[uint32]) {.global.} =
    let pos = (Int[0](), Int[0]())
    let D = (Int[1](), Int[8]())
    let idx = Int[0]() + D[0] * pos[0] + D[1] * pos[1]
    C[0] = uint32(toIntVal idx)

# Pattern E
const kernelE = metal:
  proc testE(C: ptr UncheckedArray[uint32]) {.global.} =
    let idx = block:
      const coord {.genSym.} = (Int[0](), Int[0]())
      const stride {.genSym.} = (Int[1](), Int[8]())
      Int[0]() + stride[0] * coord[0] + stride[1] * coord[1]
    C[0] = uint32(toIntVal idx)

# Pattern F
const kernelF = metal:
  proc testF(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]
    C[0] = uint32(toIntVal first)

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo kernelA

  suite "Metal - constexpr tuple init":
    test "Pattern A — constexpr tuple in let RHS":
      var engine = bkMetal.init()
      engine.ingest(kernelA)
      var res: array[1, uint32]
      engine.run("testA", res, ())
      check res[0] == 8
    test "Pattern B — constexpr in arithmetic":
      var engine = bkMetal.init()
      engine.ingest(kernelB)
      var res: array[1, uint32]
      engine.run("testB", res, ())
      check res[0] == 128
    test "Pattern C — template wrapConst":
      var engine = bkMetal.init()
      engine.ingest(kernelC)
      var res: array[1, uint32]
      engine.run("testC", res, ())
      check res[0] == 8
    test "Pattern D — tuple bracket access":
      var engine = bkMetal.init()
      engine.ingest(kernelD)
      var res: array[1, uint32]
      engine.run("testD", res, ())
      check res[0] == 0
    test "Pattern E — block with constexpr temp":
      var engine = bkMetal.init()
      engine.ingest(kernelE)
      var res: array[1, uint32]
      engine.run("testE", res, ())
      check res[0] == 0
    test "Pattern F — constexpr tuple field access":
      var engine = bkMetal.init()
      engine.ingest(kernelF)
      var res: array[1, uint32]
      engine.run("testF", res, ())
      check res[0] == 8

when isMainModule:
  runTest()
