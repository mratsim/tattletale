## Codesize ledger, zip/group family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## cgsReport renders cost of 1 call plus the marginal over the paired floor.
## No floor pairs exist in this family, the Marginal column prints `-` throughout.
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the zip/group track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# groupDimensions wrapping dimensions [0, 2) of a static rank-4 layout
const groupDimensionsMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc groupDimensionsKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((2, 3, 5, 7))
    let g = groupDimensions(L, 0, 2)
    C[0] = float32 toIntVal crd2idx(g, ((1, 2), 1, 1))

# zipDimensions interleaving two static rank-2 layouts
const zipDimensionsMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc zipDimensionsKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = make_layout((2, 3), (1, 3))
    let b = make_layout((2, 3), (6, 2))
    let z = zipDimensions(a, b)
    C[0] = float32 toIntVal crd2idx(z, ((1, 1), (1, 2)))

# ── kernel rows ──

cgsReport("cgs_layout_zip_group", [
  cgsReceipt("groupDimensionsKernel", groupDimensionsMsl),
  cgsReceipt("zipDimensionsKernel", zipDimensionsMsl)])
