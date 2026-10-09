## Run:
##   nim ceramic_cgs runner=cgs_layout_make_layout_like dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# like input, one runtime layout construction, no like call
const baselineOverheadMsl = metal:
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, int(S)))
    C[0] = float32 toInt size(L)

# make_layout_like on a compacting static rank-2 layout (2,3):(2,1) -> (3,1)
const makeLayoutLikeCompactMsl = metal:
  proc makeLayoutLikeCompactKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((2, 3), (2, 1))
    let r = make_layout_like(L)
    C[0] = float32 toInt size(r)

# make_layout_like on a runtime-stride layout (the make_tensor_like site)
const makeLayoutLikeDynStrideMsl = metal:
  proc makeLayoutLikeDynStrideKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let p = make_view(C, (int(M), int(N)), (1, int(S)))
    let r = make_layout_like(p.layout)
    C[0] = float32 toInt size(r)

# ── kernel rows ──

cgsReport("cgs_layout_make_layout_like", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("makeLayoutLikeCompactKernel", makeLayoutLikeCompactMsl, baselineOverheadMsl.len),
  cgsReceipt("makeLayoutLikeDynStrideKernel", makeLayoutLikeDynStrideMsl, baselineOverheadMsl.len)])
