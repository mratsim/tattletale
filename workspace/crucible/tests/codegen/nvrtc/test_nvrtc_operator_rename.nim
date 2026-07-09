## Minimal repro: operators via `template body: untyped` with do-block
##
## In ceramic, foldZipWith captures the loop body as `body: untyped`.
## The call uses Nim's do-block syntax:
##   foldZipWith(a, b, init):
##     acc + it_a * it_b
##
## This test checks whether +/* inside a do-block body are resolved
## as nnkInfix (gpuBinOp, correct) or nnkCall (gpuCall, →add/mul).
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_operator_rename.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc

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

echo "═══════════════════════════════════════════════════════════════════"
echo kernelRec
echo "═══════════════════════════════════════════════════════════════════"

suite "Operator rename via do-block":
  test "do-block body +/*":
    var output: array[1, float32]
    var nv = initNvrtc(kernelDo)
    nv.numBlocks = 1; nv.threadsPerBlock = 1
    nv.compile(); nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 16.0'f32

  test "chained template with state":
    var output: array[1, float32]
    var nv = initNvrtc(kernelChain)
    nv.numBlocks = 1; nv.threadsPerBlock = 1
    nv.compile(); nv.getPtx()
    nv.execute("kernel", output, ())
    check output[0] == 16.0'f32

  test "recursive fold (2 iterations)":
    var output: array[1, float32]
    var nv = initNvrtc(kernelRec)
    nv.numBlocks = 1; nv.threadsPerBlock = 1
    nv.compile(); nv.getPtx()
    nv.execute("kernel", output, ())
    # iter0: 0 + 8*2 = 16
    # iter1: 16 + 3*4 = 28
    check output[0] == 28.0'f32
