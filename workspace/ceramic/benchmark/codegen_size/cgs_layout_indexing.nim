## Run:
##   nim ceramic_cgs runner=cgs_layout_indexing dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# isolation baseline, a dynamic rank-2 layout with one direct coord read
const baselineOverheadMsl = metal:
# measured kernel, the call site reads crd2idx directly
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    C[0] = float32(crd2idx(L, (3, 2)))

# crd2idx over a runtime rank-4 layout and a runtime multi-dim coord, NHWC gmem view read
const crd2idxRank4Msl = metal:
# measured kernel, the call site reads crd2idx directly
  proc crd2idxRank4Kernel(C: ptr UncheckedArray[float32];
      N4, C4, H4, W4, n, c, h, w: int32) {.global.} =
    let L = make_layout((int(N4), int(C4), int(H4), int(W4)),
                        (int(C4 * H4 * W4), int(H4 * W4), int(W4), 1))
    C[0] = float32(crd2idx(L, (int n, int c, int h, int w)))

# crd2idx over a fully static layout, compile-time folding path
const crd2idxStaticMsl = metal:
# measured kernel, the call site reads crd2idx directly
  proc crd2idxStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((16, 64), (1, 16))
    C[0] = float32 toInt(crd2idx(L, (3, 2)))

# idx2crd_cpu over a runtime rank-2 layout, flat-index divmod decomposition, cpu wrapper shares the unsuffixed macro tree
const idx2crdCpuMsl = metal:
  proc idx2crdCpuKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let r = idx2crd_cpu(L, 37)
    C[0] = float32 r[0]

# L(i, j) call-operator accessor over the baseline input, an in-bounds check plus crd2idx
const callOperatorMsl = metal:
# measured kernel, the call site reads crd2idx directly
  proc callOperatorKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    C[0] = float32 L(3, 2)

# ── kernel rows ──

cgsReport("cgs_layout_indexing", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("crd2idxRank4Kernel", crd2idxRank4Msl),
  cgsReceipt("crd2idxStaticKernel", crd2idxStaticMsl),
  cgsReceipt("idx2crdCpuKernel", idx2crdCpuMsl, baselineOverheadMsl.len),
  cgsReceipt("callOperatorKernel", callOperatorMsl, baselineOverheadMsl.len)])
