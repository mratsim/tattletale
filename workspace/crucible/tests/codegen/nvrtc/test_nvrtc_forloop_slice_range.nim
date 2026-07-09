## When `..<` expands to Slice[int] inside cuda: block, the for-loop
## codegen emits `for(int i = Slicei32{0, N}; ...)` instead of decomposing
## the Slice into start/end ints — the nnkForStmt handler only checks
## for nnkInfix range expressions.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_forloop_slice_range.nim

import std/[unittest, macros]
import workspace/crucible/src/codegen/nvrtc

type StaticInt*[V: static int] = object

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: StaticInt[V]): int = V

# Same pattern as ceramic's `..<` for Int[V] bounds:
# `0 ..< MyInt[128]()` → Slice[int](a: 0, b: 127)
template `..<`*[V: static int](start: int; bound: StaticInt[V]): Slice[int] =
  Slice[int](a: start, b: pred(V))

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let n = StaticInt[128]()
    for i in 0 ..< n:
      C[i] = 1.0'f32

suite "For-loop with Slice[int] range":
  test "compiles and runs via NVRTC":
    var buf: array[128, float32]
    var nv = initNvrtc(kernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    check buf[0] == 1.0'f32
    check buf[127] == 1.0'f32
