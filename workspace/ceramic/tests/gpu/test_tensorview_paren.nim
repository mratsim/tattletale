## tv(0, 0) — parentheses operator on TensorView inside cuda:
## compiles and runs on GPU, verifying the computed linear index.
##
## Reproduce:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/ceramic/tests/gpu/test_issue_tensorview_paren.nim

import std/[unittest]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    # ── Build a layout: shape (8,16), LayoutLeft stride (1,8) ──
    let L = make_layout((8, 16))
    # ── Create a tensor view wrapping C with layout L ──
    let tv = make_view(C, L)
    # ── Access element at coordinate (0,5) via the () operator ──
    #   crd2idx computes: 0*stride[0] + 5*stride[1] = 0*1 + 5*8 = 40
    #   tv.data + 40 → C + 40 → reads buf[40] = 99.0
    let x = tv(0, 5)
    # ── Write the value to C[0] so the host can verify it ──
    C[0] = x

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "Ceramic - TensorView parentheses access":
    test "tv(0, 5) computes linear index from coordinates":
      var buf: array[64, float32]
      buf[0] = -1.0'f32
      buf[40] = 99.0'f32   # 0*1 + 5*8 = 40
      var engine = bkCuda.init()
      engine.ingest(kernel)
      engine.run<<(1, 1)>>("kernel", buf, ())
      # tv(0, 5) with LayoutLeft stride (1, 8) → linear index 0*1 + 5*8 = 40
      check buf[0] == 99.0'f32

when isMainModule:
  runTest()
