## Codesize ledger, make_layout_like family.
##
## make_layout_like is the shape-preserving stride compaction constructor (the CuTe factored form).
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend.
## cgsReport renders cost of 1 call plus the marginal over the paired baseline, one row per kernel:
## - baselineOverheadKernel = the like input, one runtime layout construction consumed by size, no like call
## - makeLayoutLikeDynStride pairs it directly, makeLayoutLikeCompact pairs it over a static input
##   ((2, 3):(2, 1)), its Marginal column mixes that input difference
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# like input, one runtime layout construction consumed by size, no like call
const baselineOverheadMsl = metal:
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, int(S)))
    C[0] = float32 toIntVal size(L)

# make_layout_like on a compacting static rank-2 layout (2,3):(2,1) -> (3,1)
const makeLayoutLikeCompactMsl = metal:
  proc makeLayoutLikeCompactKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((2, 3), (2, 1))
    let r = make_layout_like(L)
    C[0] = float32 toIntVal size(r)

# make_layout_like on a runtime-stride layout (the make_tensor_like site)
const makeLayoutLikeDynStrideMsl = metal:
  proc makeLayoutLikeDynStrideKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let p = make_view(C, (int(M), int(N)), (1, int(S)))
    let r = make_layout_like(p.layout)
    C[0] = float32 toIntVal size(r)

# ── kernel rows ──

cgsReport("cgs_layout_make_layout_like", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("makeLayoutLikeCompactKernel", makeLayoutLikeCompactMsl, baselineOverheadMsl.len),
  cgsReceipt("makeLayoutLikeDynStrideKernel", makeLayoutLikeDynStrideMsl, baselineOverheadMsl.len)])
