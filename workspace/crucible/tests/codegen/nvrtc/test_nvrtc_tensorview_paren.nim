## Crucible × Ceramic: TensorView() parentheses access
##
## tv(0, 0) expands through ceramic templates into code that reaches
## tv.layout.stride in a generic type context. Crucible's
## getGenericTypeName crashes on the resulting DotExpr.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_tensorview_paren.nim

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

suite "Crucible - TensorView parentheses access":
  test "tv(0,0) inside cuda: compiles via NVRTC":
    let code = kernel
    echo code
    var nv = initNvrtc(code)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    check true
