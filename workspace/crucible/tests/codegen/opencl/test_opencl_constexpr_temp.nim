## OpenCL: constexpr temporaries leaking into expression slots, six patterns
## (const+let, arithmetic, block lift, bracket access, block with two constexprs, tuple field access).
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/test_opencl_constexpr_temp.nim

import std/[unittest]
import workspace/crucible

type
  Int*[V: static int] = object
    discard
  Tuple2*[A, B] = object
    f0: A
    f1: B

template `+`*[A, B: static int](a: Int[A], b: Int[B]): Int[A + B] = Int[A + B]()
template `*`*[A, B: static int](a: Int[A], b: Int[B]): Int[A * B] = Int[A * B]()
template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

# Pattern A — constexpr tuple in let RHS
const kernelA = opencl:
  proc testA(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1'u32

# Pattern B — constexpr in arithmetic expression
const kernelB = opencl:
  proc testB(C: ptr UncheckedArray[uint32]) {.global.} =
    const a {.genSym.} = Int[8]()
    const b {.genSym.} = Int[16]()
    let x = Int[0]() + a * b
    C[0] = uint32(toIntVal x)

# Pattern C — template wrapConst (block with constexpr)
template wrapConst(a, b: untyped): untyped =
  block:
    const tmp {.genSym.} = Tuple2[typeof(a), typeof(b)](f0: a, f1: b)
    tmp
const kernelC = opencl:
  proc testC(C: ptr UncheckedArray[uint32]) {.global.} =
    let pair = wrapConst(Int[8](), Int[16]())
    C[0] = 1'u32

# Pattern D — tuple bracket access on let
const kernelD = opencl:
  proc testD(C: ptr UncheckedArray[uint32]) {.global.} =
    let pos = (Int[0](), Int[0]())
    let D = (Int[1](), Int[8]())
    let idx = Int[0]() + D[0] * pos[0] + D[1] * pos[1]
    C[0] = uint32(toIntVal idx)

# Pattern E — block with constexpr temp (evalOnceAs sim)
const kernelE = opencl:
  proc testE(C: ptr UncheckedArray[uint32]) {.global.} =
    let idx = block:
      const coord {.genSym.} = (Int[0](), Int[0]())
      const stride {.genSym.} = (Int[1](), Int[8]())
      Int[0]() + stride[0] * coord[0] + stride[1] * coord[1]
    C[0] = uint32(toIntVal idx)

# Pattern F — constexpr tuple field access (original bug)
const kernelF = opencl:
  proc testF(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]
    C[0] = uint32(toIntVal first)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "OpenCL - constexpr tuple init":
    test "Pattern A — constexpr tuple in let RHS":
      var engine = bkOpenCL.init()
      engine.ingest(kernelA)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("testA", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 1

    test "Pattern B — constexpr in arithmetic":
      var engine = bkOpenCL.init()
      engine.ingest(kernelB)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("testB", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 128

    test "Pattern C — template wrapConst":
      var engine = bkOpenCL.init()
      engine.ingest(kernelC)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("testC", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 1

    test "Pattern D — tuple bracket access":
      var engine = bkOpenCL.init()
      engine.ingest(kernelD)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("testD", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 0

    test "Pattern E — block with constexpr temp":
      var engine = bkOpenCL.init()
      engine.ingest(kernelE)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("testE", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 0

    test "Pattern F — constexpr tuple field access":
      var engine = bkOpenCL.init()
      engine.ingest(kernelF)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(sizeof(uint32))
      engine.launchKernel("testF", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, sizeof(uint32))
      engine.freeDevice(oDev)
      check res[0] == 8

when isMainModule:
  runTest()
