## Run:
##   nim ceramic_cgs runner=cgs_layout_divides dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# isolation baseline, a dynamic rank-2 layout with one direct coord read, no divide chain
const baselineOverheadMsl = metal:
# measured kernel, the call site reads crd2idx directly
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    C[0] = float32(crd2idx(L, (3, 2)))

# zipped_divide on a rank-2 dynamic layout, no view, no read
const zippedDivideOnlyMsl = metal:
# measured kernel, the call site reads crd2idx directly
  proc zippedDivideOnlyKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let zd = zipped_divide(L, (16, 16))
    C[0] = float32(crd2idx(zd, ((3, 2), (1, 0))))

# logical_divide on a rank-2 dynamic layout, no view, no read
const logicalDivideOnlyMsl = metal:
# measured kernel, the call site reads crd2idx directly
  proc logicalDivideOnlyKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let ld = logical_divide(L, (16, 16))
    C[0] = float32(crd2idx(ld, ((3, 2), (1, 0))))

# size isolation baselines, each divide row pairs a same-input
# size baseline, no divide chain, so the Marginal stays the chain
# measured kernel, the call site reads size directly
const sizeOverheadMsl = metal:
  proc sizeOverheadKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int N))
    C[0] = float32 toIntVal size(L)

# measured kernel, the call site reads size directly
const sizeOverheadCompMsl = metal:
  proc sizeOverheadCompKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, 16))
    C[0] = float32 toIntVal size(L)

# measured kernel, the call site reads size directly
const sizeOverheadRank4Msl = metal:
  proc sizeOverheadRank4Kernel(C: ptr UncheckedArray[float32]; N4, C4, H4, W4: int32) {.global.} =
    let L = make_layout((int(N4), int(C4), int(H4), int(W4)),
                        (int(C4 * H4 * W4), int(H4 * W4), int(W4), 1))
    C[0] = float32 toIntVal size(L)

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

# zipped_divide with a Layout tiler, the logical_divide whole-layout call applies to the entire layout
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

cgsReport("cgs_layout_divides", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("sizeOverheadKernel", sizeOverheadMsl),
  cgsReceipt("sizeOverheadCompKernel", sizeOverheadCompMsl, baselineOverheadMsl.len),
  cgsReceipt("sizeOverheadRank4Kernel", sizeOverheadRank4Msl, sizeOverheadMsl.len),
  cgsReceipt("zippedDivideOnlyKernel", zippedDivideOnlyMsl, baselineOverheadMsl.len),
  cgsReceipt("logicalDivideOnlyKernel", logicalDivideOnlyMsl, baselineOverheadMsl.len),
  cgsReceipt("logicalDivideCompKernel", logicalDivideCompMsl, sizeOverheadCompMsl.len),
  cgsReceipt("zippedDivideTupleKernel", zippedDivideTupleMsl, sizeOverheadMsl.len),
  cgsReceipt("zippedDivideLayoutKernel", zippedDivideLayoutMsl, sizeOverheadMsl.len),
  cgsReceipt("zippedDivideRank4Kernel", zippedDivideRank4Msl, sizeOverheadRank4Msl.len)])
