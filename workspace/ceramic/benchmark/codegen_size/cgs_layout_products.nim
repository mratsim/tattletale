## Run:
##   nim ceramic_cgs runner=cgs_layout_products dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# product input, one runtime block layout plus a static tiler, no product call
const baselineOverheadMsl = metal:
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let blk = make_layout((int(M), 4), (1, 4))
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

# blocked_product over the static rank-2 block and tiler
const blockedCurrentMsl = metal:
  proc blockedCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = blocked_product(blk, tiler)
    C[0] = float32 toIntVal size(r)

# raked_product over the static rank-2 block and tiler
const rakedCurrentMsl = metal:
  proc rakedCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let blk = make_layout((8, 4), (16, 1))
    let tiler = make_layout(2, 2)
    let r = raked_product(blk, tiler)
    C[0] = float32 toIntVal size(r)

# tile_to_shape, the production form
const tileToShapeCurrentMsl = metal:
  proc tileToShapeCurrentKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = tile_to_shape(make_layout((2, 3), (1, 2)), (2, 3, 12))
    C[0] = float32 toIntVal size(r)

# ── kernel rows ──

cgsReport("cgs_layout_products", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("tiledProductKernel", tiledProductMsl, baselineOverheadMsl.len),
  cgsReceipt("flatProductKernel", flatProductMsl, baselineOverheadMsl.len),
  cgsReceipt("logicalProductCompKernel", logicalProductCompMsl, baselineOverheadMsl.len),
  cgsReceipt("blockedCurrentKernel", blockedCurrentMsl),
  cgsReceipt("rakedCurrentKernel", rakedCurrentMsl),
  cgsReceipt("tileToShapeCurrentKernel", tileToShapeCurrentMsl),
  cgsReceipt("tileToShapeCurrentKernel", tileToShapeCurrentMsl)])
