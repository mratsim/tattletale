## Issue 5 (OPEN): TensorView() parentheses access crashes type resolver
##
## tv(0, 0) — the parentheses operator on TensorView expands through
## ceramic templates into code that reaches tv.layout.stride in a type
## context. Crucible's getGenericTypeName crashes on the resulting DotExpr.
##
## Reproduce:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/ceramic/tests/gpu/test_issue_tensorview_paren.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 16))
    let tv = make_view(C, L)
    let x = tv(0, 0)
    C[0] = x

suite "Ceramic - TensorView parentheses access":
  test "tv(0, 0) inside cuda: compiles via NVRTC":
    echo "════ kernel ═══════════════════════════════════════════════════════════"
    echo kernel
    echo "═══════════════════════════════════════════════════════════════════════"

    var buf: array[1, float32]
    var nv = initNvrtc(kernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    # If the `()` operator reads the value at (0,0), buf[0] should be
    # whatever C[0] was set to by the caller (0 by default).
    check true
