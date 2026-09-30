## Codesize ledger, layout filter family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## floorFilterKernel is the floor, a dynamic rank-2 layout with one stride-0 dimension consumed by size, filter rows pair it.
##
## filter_zeros over the fully static input still emits the Int-object shape walk, its row stands alone.
##
## cgs_layout_coalesce carries a filter_inactive row over the static all-zero-stride input, this family covers runtime inputs.
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# isolation floor, a dynamic rank-2 layout with one stride-0 dimension
# consumed by size, no filter call
const floorFilterMsl = metal:
  proc floorFilterKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int M, int N), (1, Int[0]()))
    C[0] = float32 toIntVal size(L)

# filter_zeros over the same dynamic layout, Int[0] stride folds its shape
# leaf to 1, runtime shape leaves pass through
const filterZerosDynMsl = metal:
  proc filterZerosDynKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int M, int N), (1, Int[0]()))
    let r = filter_zeros(L)
    C[0] = float32 toIntVal size(r)

# filter_zeros over a fully static layout, compile-time Int-branch path,
# emits the Int-object walk
const filterZerosStaticMsl = metal:
  proc filterZerosStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 2), (1, Int[0]()))
    let r = filter_zeros(L)
    C[0] = float32 toIntVal size(r)

# filter_inactive over the same dynamic layout, filter_zeros + coalesce chain
const filterInactiveDynMsl = metal:
  proc filterInactiveDynKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int M, int N), (1, Int[0]()))
    let r = filter_inactive(L)
    C[0] = float32 toIntVal size(r)

# ── kernel rows ──

cgsReport("cgs_layout_filters", [
  cgsReceipt("floorFilterKernel", floorFilterMsl),
  cgsReceipt("filterZerosDynKernel", filterZerosDynMsl, floorFilterMsl.len),
  cgsReceipt("filterZerosStaticKernel", filterZerosStaticMsl),
  cgsReceipt("filterInactiveDynKernel", filterInactiveDynMsl, floorFilterMsl.len)])
