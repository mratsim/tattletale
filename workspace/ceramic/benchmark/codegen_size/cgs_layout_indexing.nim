## Codesize ledger, layout indexing family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## floorCrd2idxKernel is the floor, a dynamic rank-2 layout with one direct coord read, index rows pair it.
##
## crd2idx delegates to the gpu module, idx2crd has no gpu-suffixed form, the cpu-suffixed wrapper shares the unsuffixed
## macro divmod tree and pairs the floor with a negative marginal, idx2crd emits less than the coord read.
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# isolation floor, a dynamic rank-2 layout with one direct coord read, mirror
# of the divide family floor on this family's own row
const floorCrd2idxMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc floorCrd2idxKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    C[0] = float32(crd2idx(L, (3, 2)))

# crd2idx over a runtime rank-4 layout and a runtime multi-dim coord, NHWC gmem view read
const crd2idxRank4Msl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc crd2idxRank4Kernel(C: ptr UncheckedArray[float32];
      N4, C4, H4, W4, n, c, h, w: int32) {.global.} =
    let L = make_layout((int(N4), int(C4), int(H4), int(W4)),
                        (int(C4 * H4 * W4), int(H4 * W4), int(W4), 1))
    C[0] = float32(crd2idx(L, (int n, int c, int h, int w)))

# crd2idx over a fully static layout, compile-time folding path
const crd2idxStaticMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc crd2idxStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((16, 64), (1, 16))
    C[0] = float32 toIntVal(crd2idx(L, (3, 2)))

# idx2crd_cpu over a runtime rank-2 layout, flat-index divmod decomposition, cpu wrapper shares the unsuffixed macro tree
const idx2crdCpuMsl = metal:
  proc idx2crdCpuKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    let r = idx2crd_cpu(L, 37)
    C[0] = float32 r[0]

# L(i, j) call-operator accessor over the floor input, underscore check plus crd2idx
const callOperatorMsl = metal:
# tiles-allow measured kernel, the call site needs raw crd2idx
  proc callOperatorKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let L = make_layout((int M, int N), (1, int S))
    C[0] = float32 L(3, 2)

# ── kernel rows ──

cgsReport("cgs_layout_indexing", [
  cgsReceipt("floorCrd2idxKernel", floorCrd2idxMsl),
  cgsReceipt("crd2idxRank4Kernel", crd2idxRank4Msl),
  cgsReceipt("crd2idxStaticKernel", crd2idxStaticMsl),
  cgsReceipt("idx2crdCpuKernel", idx2crdCpuMsl, floorCrd2idxMsl.len),
  cgsReceipt("callOperatorKernel", callOperatorMsl, floorCrd2idxMsl.len)])
