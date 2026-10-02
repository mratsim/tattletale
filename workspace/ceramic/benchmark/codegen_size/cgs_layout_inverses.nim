## Run:
##   nim ceramic_cgs runner=cgs_layout_inverses dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# isolation baseline, a dynamic shape rank-2 layout with static strides, no inverse call
const baselineOverheadMsl = metal:
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int M, int N), (1, 16))
    C[0] = float32 toIntVal size(L)

# right_inverse over the baseline input, copyFrom quasi-inverse call-site shape
const rightInverseMsl = metal:
  proc rightInverseKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int M, int N), (1, 16))
    let r = right_inverse(L)
    C[0] = float32 toIntVal size(r)

# right_inverse over a runtime-stride rank-2 layout, chain keeps the static-stride run, dynamic stride leaf ends it
const rightInverseDynStrideMsl = metal:
  proc rightInverseDynStrideKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let r = right_inverse(L)
    C[0] = float32 toIntVal size(r)

# right_inverse over a fully static layout, compile-time fold path, still emits the inline chain
const rightInverseStaticMsl = metal:
  proc rightInverseStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 2), (1, 16))
    let r = right_inverse(L)
    C[0] = float32 toIntVal size(r)

# left_inverse over the same dynamic shape rank-2 layout, all strides static per the divisibility precondition
const leftInverseMsl = metal:
  proc leftInverseKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int M, int N), (1, 16))
    let r = left_inverse(L)
    C[0] = float32 toIntVal size(r)

# ── kernel rows ──

cgsReport("cgs_layout_inverses", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("rightInverseKernel", rightInverseMsl, baselineOverheadMsl.len),
  cgsReceipt("rightInverseDynStrideKernel", rightInverseDynStrideMsl, baselineOverheadMsl.len),
  cgsReceipt("rightInverseStaticKernel", rightInverseStaticMsl),
  cgsReceipt("leftInverseKernel", leftInverseMsl, baselineOverheadMsl.len)])
