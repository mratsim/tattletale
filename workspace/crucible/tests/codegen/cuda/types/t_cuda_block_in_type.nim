## Block expression through the CUDA type resolver
##
## Test that a device function calling a `()` operator template
## with a `block:` body and local temps compiles through the `cuda:` macro
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/types/t_cuda_block_in_type.nim
import std/strformat
import workspace/crucible

type
  MyLayout = object
    m, n: int32

  MyView = object
    data: ptr UncheckedArray[float32]
    layout: MyLayout

# Exercised shape: `block:` body, local temps, a `result` var, block tail returned
template `()`(v: MyView, a, b: int32): auto =
  block:
    let coord = (a, b)
    let offset = coord[0] * v.layout.n + coord[1]
    var result: MyView
    result.data = cast[ptr UncheckedArray[float32]](
      cast[uint64](v.data) + cast[uint64](offset * 4))
    result.layout = MyLayout(m: v.layout.m - a, n: v.layout.n - b)
    result

proc deviceFn(view: MyView) =
  discard view(0, 0).data[0]

const kernel = cuda:
  proc reproKernel(
    output: ptr UncheckedArray[float32],
    input: ptr UncheckedArray[float32],
    M, N: int32,
  ) {.global.} =
    let v = MyView(data: input, layout: MyLayout(m: M, n: N))
    deviceFn(v)

when isMainModule:
  # The `cuda:` macro expansion is the test: reaching runtime means
  # the type resolver handled the block expression
  echo kernel
  echo "  OK — compiled"
