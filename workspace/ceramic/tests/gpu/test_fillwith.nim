## fillWith on TensorView inside cuda: — for-loop expansion with `[]` access
##
## Issue 4: gemm/fillWith expand to for-loops with `[]` on TensorView.
## Still blocked after issue 2/3 fixes — investigating remaining blockers.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/ceramic/tests/gpu/test_ceramic_issue4_fillwith.nim

import std/[unittest]
import workspace/crucible/src/codegen/nvrtc
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernel_fillwith_gpu

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((8, 16))
    var tv = make_view(C, L)
    fillWith(tv, 42.0'f32)

suite "Ceramic - fillWith on TensorView":
  test "fillWith inside cuda: compiles via NVRTC":
    var buf: array[128, float32]
    var nv = initNvrtc(kernel)
    nv.numBlocks = 1
    nv.threadsPerBlock = 1
    nv.compile()
    nv.getPtx()
    nv.execute("kernel", buf, ())
    # fillWith writes 42.0 to every element
    check buf[0] == 42.0'f32
    check buf[127] == 42.0'f32
