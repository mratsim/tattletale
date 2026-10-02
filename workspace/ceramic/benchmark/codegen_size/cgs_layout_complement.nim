## Run:
##   nim ceramic_cgs runner=cgs_layout_complement dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_layout_copy_gpu
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# complement of a runtime-shape rank-2 layout, static strides, the direct-call pattern of tests/test_layout_algebra.nim
const complementDirectMsl = metal:
  proc complementDirectKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let L = make_layout((int(M), int(N)), (1, 16))
    let r = complement(L)
    C[0] = float32 toIntVal size(r)

# complement of a fully static layout, the compile-time path
const complementStaticMsl = metal:
  proc complementStaticKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((4, 2), (1, 16))
    let r = complement(L)
    C[0] = float32 toIntVal size(r)

# the copy chain, right_inverse + coalesce(compose(dst, R)) in copyFrom,
# the KV-write pattern, row-padded dst and row-compact src
const copyChainMsl = metal:
  proc copyChainKernel(D, S: ptr UncheckedArray[float32]; M, N, Sd, Ss: int32) {.global.} =
    var dst = make_view(D, (int(M), int(N)), (1, int(Sd)))
    let src = make_view(S, (int(M), int(N)), (1, int(Ss)))
    copyFrom(dst, src)

# local_tile, complement via zipped_divide -> logical_divide
const localTileCompMsl = metal:
  proc localTileCompKernel(C: ptr UncheckedArray[float32]; M, N, i, j: int32) {.global.} =
    let p = make_view(C, (int(M), int(N)), (1, int(N)))
    let t = local_tile(p, (16, 16), (int(i), int(j)))
    C[0] = t(0, 0)

# ── kernel rows ──

cgsReport("cgs_layout_complement", [
  cgsReceipt("complementDirectKernel", complementDirectMsl),
  cgsReceipt("complementStaticKernel", complementStaticMsl),
  cgsReceipt("copyChainKernel", copyChainMsl),
  cgsReceipt("localTileCompKernel", localTileCompMsl)])
