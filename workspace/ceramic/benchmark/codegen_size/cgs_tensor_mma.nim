## Run:
##   nim ceramic_cgs runner=cgs_tensor_mma dump=true
##   from the tattletale/ directory.
##
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_mma_configgen
import workspace/ceramic/src/hardware/h_mma_registry
import workspace/ceramic/src/hardware/h_mma_properties
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

{.experimental: "callOperator".}

#  the m16n8k8 tf32 atom with a (2, 2, 1) thread tiling, the same setup
#  locked by tests/atoms_mma/test_atoms_mma_partitioning.nim, 128 threads,
#  A tile unit (32, 8), covered C tile (32, 16)
const atom = SM80_16x8x8_F32TF32TF32F32_TN

template tiled(thrM, thrN, thrK: static int): untyped =
  TiledMma[typeof(atom), typeof(make_layout((thrM, thrN, thrK)))](
    atom: atom, threadLayout: make_layout((thrM, thrN, thrK)))

const tma = tiled(2, 2, 1)

# ── baseline, no fragment machinery ──

# dynamic rank-2 view, one manual-offset element read
const baselineOverheadMsl = metal:
  proc baselineOverheadKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let p = make_view(C, (int M, int N), (1, int S))
    C[0] = float32 p.data[int R0 * 1 + int C0 * int S]

# ── thrfrg: the fragment layout of an operand tile ──

# static col-major A tile (64, 32):(1, 64), rest (2, 4) per thread
const thrfrgAMsl = metal:
  proc thrfrgAKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let L = make_layout((64, 32), (1, 64))
    let f = tma.thrfrg_A(L)
    C[0] = float32 toIntVal(cosize(f))

# static col-major C tile (32, 16):(1, 32)
const thrfrgCMsl = metal:
  proc thrfrgCKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0: int32) {.global.} =
    let L = make_layout((32, 16), (1, 32))
    let f = tma.thrfrg_C(L)
    C[0] = float32 toIntVal(cosize(f))

# ── partition: the thread's fragment view of a tensor ──

# A operand: view (64, 32) col-major, thread T's fragment view construction (tensors_mma_partitioning.nim partition_A)
const partitionAMsl = metal:
  proc partitionAKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0, TID: int32) {.global.} =
    let p = make_view(C, (64, 32), (1, 64))
    let thr = tma.get_slice(int TID)
    let av = tma.partition_A(thr, p)
    C[0] = float32 av.data[0]

# C operand: view (32, 16) col-major, thread T's fragment view construction
const partitionCMsl = metal:
  proc partitionCKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0, TID: int32) {.global.} =
    let p = make_view(C, (32, 16), (1, 32))
    let thr = tma.get_slice(int TID)
    let cv = tma.partition_C(thr, p)
    C[0] = float32 cv.data[0]

# ── make_fragment: the register buffer in hardware order ──

# fragment buffer allocated from thread T's A partition view, one round trip
const makeFragmentAMsl = metal:
  proc makeFragmentAKernel(C: ptr UncheckedArray[float32]; M, N, S, R0, C0, TID: int32) {.global.} =
    let p = make_view(C, (64, 32), (1, 64))
    let thr = tma.get_slice(int TID)
    let av = tma.partition_A(thr, p)
    var aFrag = make_fragment_A(tma.atom, av)
    aFrag[0] = 1.5'f32
    C[0] = float32 aFrag[0]

# ── cStoreMask: the C-store predication mask ──

# tileM/tileN are the thread layout's exact coverage (32, 16), the predication loop
const cStoreMaskMsl = metal:
  proc cStoreMaskKernel(C: ptr UncheckedArray[float32]; M, N, TID: int32) {.global.} =
    C[0] = float32 cStoreMask(tma, int TID, 32, 16, int M, int N)

# ── kernel rows ──
# Marginal = cost minus baseline, computed inside cgsReport:
# - thrfrg rows sit on baselineOverheadKernel
# - partition rows sit on their thrfrg row, the layout-build cost is shared
# - makeFragmentA sits on partitionA, the fragment-buffer cost over the view
# - cStoreMask sits on baselineOverheadKernel
cgsReport("cgs_tensor_mma", [
  cgsReceipt("baselineOverheadKernel", baselineOverheadMsl),
  cgsReceipt("thrfrgAKernel", thrfrgAMsl, baselineOverheadMsl.len),
  cgsReceipt("thrfrgCKernel", thrfrgCMsl, baselineOverheadMsl.len),
  cgsReceipt("partitionAKernel", partitionAMsl, thrfrgAMsl.len),
  cgsReceipt("partitionCKernel", partitionCMsl, thrfrgCMsl.len),
  cgsReceipt("makeFragmentAKernel", makeFragmentAMsl, partitionAMsl.len),
  cgsReceipt("cStoreMaskKernel", cStoreMaskMsl, baselineOverheadMsl.len)])
