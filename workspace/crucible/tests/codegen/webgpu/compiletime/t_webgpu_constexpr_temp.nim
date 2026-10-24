## WebGPU: constexpr temporaries leaking into expression slots, six kernel shapes that mix const declarations with expression use.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/compiletime/t_webgpu_constexpr_temp.nim

import std/[unittest, strformat]
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

# Pattern A
const kernelA = webgpu:
  proc testA(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp; C[0] = 1'u32

# Pattern B
const kernelB = webgpu:
  proc testB(C: ptr UncheckedArray[uint32]) {.global.} =
    const a {.genSym.} = Int[8](); const b {.genSym.} = Int[16]()
    let x = Int[0]() + a * b; C[0] = uint32(toIntVal x)

# Pattern C
template wrapConst(a, b: untyped): untyped =
  block:
    const tmp {.genSym.} = Tuple2[typeof(a), typeof(b)](f0: a, f1: b)
    tmp
const kernelC = webgpu:
  proc testC(C: ptr UncheckedArray[uint32]) {.global.} =
    let temps = wrapConst(Int[8](), Int[16]()); C[0] = 1'u32

# Pattern D
const kernelD = webgpu:
  proc testD(C: ptr UncheckedArray[uint32]) {.global.} =
    let pos = (Int[0](), Int[0]()); let D = (Int[1](), Int[8]())
    let idx = Int[0]() + D[0] * pos[0] + D[1] * pos[1]
    C[0] = uint32(toIntVal idx)

# Pattern E
const kernelE = webgpu:
  proc testE(C: ptr UncheckedArray[uint32]) {.global.} =
    let idx = block:
      const coord {.genSym.} = (Int[0](), Int[0]())
      const stride {.genSym.} = (Int[1](), Int[8]())
      Int[0]() + stride[0] * coord[0] + stride[1] * coord[1]
    C[0] = uint32(toIntVal idx)

# Pattern F
const kernelF = webgpu:
  proc testF(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]; C[0] = uint32(toIntVal first)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelA
  echo "\n---\n"
  echo kernelB
  echo "\n---\n"
  echo kernelC
  echo "\n---\n"
  echo kernelD
  echo "\n---\n"
  echo kernelE
  echo "\n---\n"
  echo kernelF

  suite "WebGPU - constexpr tuple init":
    test "Pattern A — constexpr tuple in let RHS":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelA)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("testA", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 1
    test "Pattern B — constexpr in arithmetic":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelB)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("testB", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 128
    test "Pattern C — template wrapConst":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelC)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("testC", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 1
    test "Pattern D — tuple bracket access":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelD)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("testD", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 0
    test "Pattern E — block with constexpr temp":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelE)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("testE", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 0
    test "Pattern F — constexpr tuple field access":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelF)
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("testF", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 8

when isMainModule:
  runTest()
