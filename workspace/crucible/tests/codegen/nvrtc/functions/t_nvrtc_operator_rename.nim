## `+`/`*` inside a do-block template body resolve as nnkInfix (gpuBinOp),
## not as nnkCall gpuCall dispatch.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_operator_rename.nim

import std/[unittest]
import workspace/crucible

type
  Int*[V: static int] = object
    discard

template `+`*[A, B: static int](a: Int[A], b: Int[B]): Int[A + B] = Int[A + B]()
template `*`*[A, B: static int](a: Int[A], b: Int[B]): Int[A * B] = Int[A * B]()
template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

# Pattern A: do-block body (foldZipWith style)
template foldDo(body: untyped): untyped =
  block:
    let acc {.inject.} = Int[0]()
    let it_a {.inject.} = Int[8]()
    let it_b {.inject.} = Int[2]()
    body

const kernelDo = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let x = foldDo:
      acc + it_a * it_b
    C[0] = float32(toIntVal x)

# Pattern B: Template that takes state and passes it to another template
template chainFold(state: typed; body: untyped): untyped =
  block:
    let s = state
    let acc {.inject.} = s
    let it_a {.inject.} = Int[8]()
    let it_b {.inject.} = Int[2]()
    body

const kernelChain = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let x = chainFold(Int[0]()):
      acc + it_a * it_b
    C[0] = float32(toIntVal x)

# Pattern C: Recursive fold (like foldZipWith_recurse)
template recFold(idx: static int; state: typed; body: untyped): untyped =
  when idx == 0:
    block:
      let acc {.inject.} = state
      let it_a {.inject.} = Int[8]()
      let it_b {.inject.} = Int[2]()
      let field = body
      recFold(1, field, body)
  else:
    block:
      let acc {.inject.} = state
      let it_a {.inject.} = Int[3]()
      let it_b {.inject.} = Int[4]()
      body

const kernelRec = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let x = recFold(0, Int[0]()):
      acc + it_a * it_b
    C[0] = float32(toIntVal x)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo "═══════════════════════════════════════════════════════════════════"
  echo kernelRec
  echo "═══════════════════════════════════════════════════════════════════"

  suite "Operator rename via do-block":
    test "do-block body +/*":
      var output: array[1, float32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelDo)
      let o1 = engine.allocDevice(float32, output.len)
      engine.launchKernel<<(1, 1)>>("kernel", o1, ())
      engine.copyToHostFromDevice(addr output[0], o1, output.len)
      check output[0] == 16.0'f32

    test "chained template with state":
      var output: array[1, float32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelChain)
      let o2 = engine.allocDevice(float32, output.len)
      engine.launchKernel<<(1, 1)>>("kernel", o2, ())
      engine.copyToHostFromDevice(addr output[0], o2, output.len)
      check output[0] == 16.0'f32

    test "recursive fold (2 iterations)":
      var output: array[1, float32]
      var engine = bkCuda.newEngine()
      engine.ingest(kernelRec)
      let o3 = engine.allocDevice(float32, output.len)
      engine.launchKernel<<(1, 1)>>("kernel", o3, ())
      engine.copyToHostFromDevice(addr output[0], o3, output.len)
      # iter0: 0 + 8*2 = 16
      # iter1: 16 + 3*4 = 28
      check output[0] == 28.0'f32

when isMainModule:
  runTest()
