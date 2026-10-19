## Vulkan (GLSL): constexpr temporaries leaking into expression slots.
## Patterns A-F exercise the constexpr-lift pass positions:
## let RHS, arithmetic, template block, tuple bracket and field access.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/compiletime/t_vulkan_constexpr_temp.nim

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
const kernelA = vulkan:
  proc testA(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp; C[0] = 1'u32

# Pattern B
const kernelB = vulkan:
  proc testB(C: ptr UncheckedArray[uint32]) {.global.} =
    const a {.genSym.} = Int[8](); const b {.genSym.} = Int[16]()
    let x = Int[0]() + a * b; C[0] = uint32(toIntVal x)

# Pattern C
template wrapConst(a, b: untyped): untyped =
  block:
    const tmp {.genSym.} = Tuple2[typeof(a), typeof(b)](f0: a, f1: b)
    tmp
const kernelC = vulkan:
  proc testC(C: ptr UncheckedArray[uint32]) {.global.} =
    let pair = wrapConst(Int[8](), Int[16]()); C[0] = 1'u32

# Pattern D
const kernelD = vulkan:
  proc testD(C: ptr UncheckedArray[uint32]) {.global.} =
    let pos = (Int[0](), Int[0]()); let D = (Int[1](), Int[8]())
    let idx = Int[0]() + D[0] * pos[0] + D[1] * pos[1]
    C[0] = uint32(toIntVal idx)

# Pattern E
const kernelE = vulkan:
  proc testE(C: ptr UncheckedArray[uint32]) {.global.} =
    let idx = block:
      const coord {.genSym.} = (Int[0](), Int[0]())
      const stride {.genSym.} = (Int[1](), Int[8]())
      Int[0]() + stride[0] * coord[0] + stride[1] * coord[1]
    C[0] = uint32(toIntVal idx)

# Pattern F
const kernelF = vulkan:
  proc testF(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]; C[0] = uint32(toIntVal first)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo kernelA
  echo "\n---"
  echo kernelB
  echo "\n---"
  echo kernelC
  echo "\n---"
  echo kernelD
  echo "\n---"
  echo kernelE
  echo "\n---"
  echo kernelF

suite "Vulkan - constexpr tuple init":
  test "Pattern A — constexpr tuple in let RHS":
    var engine = bkVulkan.init()
    engine.ingest(kernelA)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("testA", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    check res[0] == 1
  test "Pattern B — constexpr in arithmetic":
    var engine = bkVulkan.init()
    engine.ingest(kernelB)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("testB", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    check res[0] == 128
  test "Pattern C — template wrapConst":
    var engine = bkVulkan.init()
    engine.ingest(kernelC)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("testC", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    check res[0] == 1
  test "Pattern D — tuple bracket access":
    var engine = bkVulkan.init()
    engine.ingest(kernelD)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("testD", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    check res[0] == 0
  test "Pattern E — block with constexpr temp":
    var engine = bkVulkan.init()
    engine.ingest(kernelE)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("testE", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    check res[0] == 0
  test "Pattern F — constexpr tuple field access":
    var engine = bkVulkan.init()
    engine.ingest(kernelF)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("testF", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    check res[0] == 8

when isMainModule:
  runTest()
