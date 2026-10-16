## Run:
##   nim ceramic_cgs runner=cgs_layout_coalesce dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# coalesce of a static contiguous rank-3 layout
const coalesceStaticMsl = metal:
  proc coalesceStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((2, 4, 8), (1, 2, 8))
    let r = coalesce(L)
    C[0] = float32 toInt crd2idx(r, 7)

# coalesce of a static layout with a stride-0 trailing dimension
const coalesceZerosMsl = metal:
  proc coalesceZerosKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 1), (1, 0))
    let r = coalesce(L)
    C[0] = float32 toInt crd2idx(r, 2)

# coalesce of a runtime layout, the copy-chain pattern
const coalesceDynMsl = metal:
  proc coalesceDynKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let p = make_view(C, (int(M), int(N)), (1, int(S)))
    let r = coalesce(p.layout)
    C[0] = float32 toInt size(r)

# ── kernel rows ──

cgsReport("cgs_layout_coalesce", [
  cgsReceipt("coalesceStaticKernel", coalesceStaticMsl),
  cgsReceipt("coalesceZerosKernel", coalesceZerosMsl),
  cgsReceipt("coalesceDynKernel", coalesceDynMsl)])
