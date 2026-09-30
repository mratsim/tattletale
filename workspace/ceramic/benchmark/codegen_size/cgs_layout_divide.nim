## Codesize ledger, divide family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend.
## cgsReport renders cost of 1 call plus the marginal over the paired floor, one row per kernel:
## - floorRank2 = the isolation floor, a dynamic rank-2 layout with one direct coord read, every rank-2 row pairs it
## - zippedDivideOnly/logicalDivideOnly = the divide chains in isolation on that input, no view, no read
## - call-site rows cover logical_divide with a Layout tiler and zipped_divide over tuple, Layout,
##   and rank-4 tiler inputs, the rank-4 row has no matching floor and prints `-`
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# isolation floor, a dynamic rank-2 layout with one direct coord read, no divide chain
const floorRank2Msl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc floorRank2Kernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    C[0] = float32(crd2idx(L, (3, 2)))

# zipped_divide on a rank-2 dynamic layout, no view, no read
const zippedDivideOnlyMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc zippedDivideOnlyKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let zd = zipped_divide(L, (16, 16))
    C[0] = float32(crd2idx(zd, ((3, 2), (1, 0))))

# logical_divide on a rank-2 dynamic layout, no view, no read
const logicalDivideOnlyMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc logicalDivideOnlyKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let ld = logical_divide(L, (16, 16))
    C[0] = float32(crd2idx(ld, ((3, 2), (1, 0))))

# logical_divide with a Layout tiler, covers the complement + compose general path
const logicalDivideCompMsl = metal:
  proc logicalDivideCompKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, 16))
    let d = logical_divide(L, make_layout((16, 4)))
    C[0] = float32 toIntVal size(d)

# zipped_divide with a tuple tiler on a runtime rank-2 layout
const zippedDivideTupleMsl = metal:
  proc zippedDivideTupleKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, int(N)))
    let zd = zipped_divide(L, (16, 16))
    C[0] = float32 toIntVal size(zd)

# zipped_divide with a Layout tiler, routes through the tile_unzip general path
const zippedDivideLayoutMsl = metal:
  proc zippedDivideLayoutKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, int(N)))
    let zd = zipped_divide(L, make_layout((16, 4), (1, 4)))
    C[0] = float32 toIntVal size(zd)

# zipped_divide on a runtime rank-4 layout, mirrors the NHWC gmem view pattern
const zippedDivideRank4Msl = metal:
  proc zippedDivideRank4Kernel(C: ptr UncheckedArray[float32]; N4, C4, H4, W4: int32) {.global.} =
    let L = make_layout((int(N4), int(C4), int(H4), int(W4)),
                        (int(C4 * H4 * W4), int(H4 * W4), int(W4), 1))
    let zd = zipped_divide(L, (8, 8, 8, 8))
    C[0] = float32 toIntVal size(zd)

# ── kernel rows ──

cgsReport("cgs_layout_divide", [
  cgsReceipt("floorRank2Kernel", floorRank2Msl),
  cgsReceipt("zippedDivideOnlyKernel", zippedDivideOnlyMsl, floorRank2Msl.len),
  cgsReceipt("logicalDivideOnlyKernel", logicalDivideOnlyMsl, floorRank2Msl.len),
  cgsReceipt("logicalDivideCompKernel", logicalDivideCompMsl, floorRank2Msl.len),
  cgsReceipt("zippedDivideTupleKernel", zippedDivideTupleMsl, floorRank2Msl.len),
  cgsReceipt("zippedDivideLayoutKernel", zippedDivideLayoutMsl, floorRank2Msl.len),
  cgsReceipt("zippedDivideRank4Kernel", zippedDivideRank4Msl)])
