## Codesize ledger, make_fragment_like family.
##
## make_fragment_like is the fragment/rest-path constructor through the tiled_product chain (the CuTe composition).
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend.
## cgsReport renders cost of 1 call plus the marginal over the paired floor, one row per kernel:
## - floorFragmentLike = the fragment input, one static V-block layout construction consumed by size, no like call
## - makeFragmentLikeV pairs it directly, makeFragmentLikeBroadcast pairs it over a broadcast input
##   ((4, 8):(0, 1)), its Marginal column mixes that input difference
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# fragment input, one static V-block layout construction consumed by size, no like call
const floorFragmentLikeMsl = metal:
  proc floorFragmentLikeKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout(((16, 2), 16), ((1, 16), 4))
    C[0] = float32 toIntVal size(L)

# make_fragment_like with a (16, 2) V block, feeds the tensor-core fragment call site
const makeFragmentLikeVMsl = metal:
  proc makeFragmentLikeVKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout(((16, 2), 16), ((1, 16), 4))
    let f = make_fragment_like(L)
    C[0] = float32 toIntVal size(f)

# make_fragment_like with a broadcast V, feeds the epilogue broadcast-bias call site
const makeFragmentLikeBroadcastMsl = metal:
  proc makeFragmentLikeBroadcastKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 8), (0, 1))
    let f = make_fragment_like(L)
    C[0] = float32 toIntVal size(f)

# ── kernel rows ──

cgsReport("cgs_layout_make_fragment_like", [
  cgsReceipt("floorFragmentLikeKernel", floorFragmentLikeMsl),
  cgsReceipt("makeFragmentLikeVKernel", makeFragmentLikeVMsl, floorFragmentLikeMsl.len),
  cgsReceipt("makeFragmentLikeBroadcastKernel", makeFragmentLikeBroadcastMsl, floorFragmentLikeMsl.len)])
