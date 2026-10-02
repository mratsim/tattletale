## Run:
##   nim ceramic_cgs runner=cgs_layout_make_fragment_like dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# fragment input, one static V-block layout construction consumed by size, no like call
const baselineOverheadMsl = metal:
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout(((16, 2), 16), ((1, 16), 4))
    C[0] = float32 toIntVal size(L)

# make_fragment_like with a (16, 2) V block, feeds the tensor-core fragment call site
const makeFragmentLikeVMsl = metal:
  proc makeFragmentLikeVKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout(((16, 2), 16), ((1, 16), 4))
    let f = make_fragment_like(L, (16, 2))
    C[0] = float32 toIntVal size(f)

# make_fragment_like with a broadcast V, feeds the epilogue broadcast-bias call site
const makeFragmentLikeBroadcastMsl = metal:
  proc makeFragmentLikeBroadcastKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 8), (0, 1))
    let f = make_fragment_like(L, 4)
    C[0] = float32 toIntVal size(f)

# ── kernel rows ──

cgsReport("cgs_layout_make_fragment_like", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("makeFragmentLikeVKernel", makeFragmentLikeVMsl, baselineOverheadMsl.len),
  cgsReceipt("makeFragmentLikeBroadcastKernel", makeFragmentLikeBroadcastMsl, baselineOverheadMsl.len)])
