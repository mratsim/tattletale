## Codesize ledger, product family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend.
## cgsReport renders cost of 1 call plus the marginal over the paired floor, one row per kernel:
## - floorProduct = the product input, one runtime block layout plus a static tiler, consumed by size, no product call
## - tiledProduct/flatProduct pair it directly, logicalProductComp pairs it over a different input
##   ((M,N):(1,16), tiler (16, 16)), its Marginal column mixes that input difference
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the product track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# product input, one runtime block layout plus a static tiler, consumed by size, no product call
const floorProductMsl = metal:
  proc floorProductKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let blk = make_layout((int(M), 4), (1, 4))
    let tiler = make_layout((2, 2))
    C[0] = float32 toIntVal size(blk)

# tiled_product, concat with the block dimension kept grouped
const tiledProductMsl = metal:
  proc tiledProductKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let blk = make_layout((int(M), 4), (1, 4))
    let p = tiled_product(blk, make_layout((2, 2)))
    C[0] = float32 toIntVal size(p)

# flat_product, concat with both dimensions unpacked
const flatProductMsl = metal:
  proc flatProductKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let blk = make_layout((int(M), 4), (1, 4))
    let p = flat_product(blk, make_layout((2, 2)))
    C[0] = float32 toIntVal size(p)

# logical_product, covers the compose(complement(a, ...), tiler) site
const logicalProductCompMsl = metal:
  proc logicalProductCompKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, 16))
    let p = logical_product(L, make_layout((16, 16)))
    C[0] = float32 toIntVal size(p)

# ── kernel rows ──

cgsReport("cgs_layout_product", [
  cgsReceipt("floorProductKernel", floorProductMsl),
  cgsReceipt("tiledProductKernel", tiledProductMsl, floorProductMsl.len),
  cgsReceipt("flatProductKernel", flatProductMsl, floorProductMsl.len),
  cgsReceipt("logicalProductCompKernel", logicalProductCompMsl, floorProductMsl.len)])
