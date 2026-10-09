## Run:
##   nim ceramic_cgs runner=cgs_layout_compose dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# compose of two static rank-2 layouts
const composeStaticMsl = metal:
  proc composeStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = make_layout((4, 4), (1, 4))
    let b = make_layout((2, 2), (1, 2))
    let r = compose(a, b)
    C[0] = float32 toInt crd2idx(r, (1, 1))

# compose of a static rank-2 layout with a nested layout
const composeNestedMsl = metal:
  proc composeNestedKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = make_layout((8, 8), (1, 8))
    let b = make_layout(((2, 2), (2, 8)), ((1, 4), (2, 8)))
    let r = compose(a, b)
    C[0] = float32 toInt crd2idx(r, ((1, 1), (1, 1)))

# compose with a runtime rank-1 LHS
const composeRank1Msl = metal:
  proc composeRank1Kernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let a = make_layout(int(M), 2)
    let b = make_layout((4, 4), (1, 4))
    let r = compose(a, b)
    C[0] = float32 toInt crd2idx(r, (1, 1))

# compose of a runtime layout with a static layout, the thrfrg_A/B/C compose call sites in k_layout_gemm.nim
const composeDynMsl = metal:
  proc composeDynKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let a = make_layout((int(M), int(N)), (1, 16))
    let b = make_layout((4, 4), (1, 4))
    let r = compose(a, b)
    C[0] = float32 toInt crd2idx(r, (1, 1))

# ── kernel rows ──

cgsReport("cgs_layout_compose", [
  cgsReceipt("composeStaticKernel", composeStaticMsl),
  cgsReceipt("composeNestedKernel", composeNestedMsl),
  cgsReceipt("composeRank1Kernel", composeRank1Msl),
  cgsReceipt("composeDynKernel", composeDynMsl)])
