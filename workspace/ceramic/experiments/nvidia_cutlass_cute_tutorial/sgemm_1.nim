## sgemm_1 — faithful CuTe sgemm_1.cu port using ceramic primitives
##
## Mirrors the original kernel body structure statement-by-statement.
## GPU validation via NVRTC.

import std/[strformat, random]
import workspace/ceramic/src/int_tuples {.all.}
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_fillwith_gpu
import workspace/ceramic/src/kernel_copy_gpu
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_axpby_gpu
import workspace/ceramic/experiments/experiment_testutils
import workspace/crucible/src/codegen/nvrtc

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Device kernel body (faithful sgemm_1.cu port)
# ═════════════════════════════════════════════════════════════════════════

proc sgemm_1_kernel(
       mA, mB, mC: distinct TensorView,
       cta_tiler: auto,
       tA, tB, tC: distinct Layout) =
  ## sgemm_1 device kernel — identical algorithm to CuTe sgemm_1.cu.
  ##
  ## Setup:
  ##   CTA coordinate
  ##   local_tile with Step (×3)
  ##   local_partition (×6)
  ##   make_tensor_like + clear for accumulator
  ## Main loop:
  ##   copy A tile, copy B tile, gemm
  ## Epilogue:
  ##   axpby

  # ── CTA coordinate ──
  let cta_coord = (0, 0, X())  # blockIdx.x, blockIdx.y, _ on GPU

  # ── CTA tile extraction (with Step) ──
  #   mA: (M,K)  cta_tiler: (BLK_M, BLK_N, BLK_K)
  #   Step (_1, X, _1): keep M and K, drop N
  let gA = local_tile(mA, cta_tiler, cta_coord, (Y, X, Y))  # (BLK_M, BLK_K, k)
  #   Step (X, _1, _1): keep N and K, drop M
  let gB = local_tile(mB, cta_tiler, cta_coord, (X, Y, Y))  # (BLK_N, BLK_K, k)
  #   Step (_1, _1, X): keep M and N, drop K
  let gC = local_tile(mC, cta_tiler, cta_coord, (Y, Y, X))  # (BLK_M, BLK_N)

  # ── Shared memory tiles ──
  let sA = make_tensor_like(gA(_, _, 0))  # (BLK_M, BLK_K)
  let sB = make_tensor_like(gB(_, _, 0))  # (BLK_N, BLK_K)

  # ── A/B thread partitioning (3-arg) ──
  let tAgA = local_partition(gA, tA, 0)  # (THR_M, THR_K, k) — threadIdx.x on GPU
  var tAsA = local_partition(sA, tA, 0)  # (THR_M, THR_K)
  let tBgB = local_partition(gB, tB, 0)  # (THR_N, THR_K, k)
  var tBsB = local_partition(sB, tB, 0)  # (THR_N, THR_K)

  # ── C thread partitioning (4-arg with Step) ──
  #   sA: (BLK_M, BLK_K), tC: (THR_M, THR_N)
  #   Step (_1, X): partition M by tC mode 0, keep K whole
  let tCsA = local_partition(sA, tC, 0, (Y, X))  # (THR_M, BLK_K)
  #   sB: (BLK_N, BLK_K)
  #   Step (X, _1): keep N whole, partition K by tC mode 1
  let tCsB = local_partition(sB, tC, 0, (X, Y))  # (THR_N, BLK_K)
  #   gC: (BLK_M, BLK_N)
  #   Step (_1, _1): partition both modes
  var tCgC = local_partition(gC, tC, 0, (Y, Y))  # (THR_M, THR_N)

  # ── Accumulators ──
  var tCrC = make_tensor_like(tCgC)  # (THR_M, THR_N)
  fillWith(tCrC, float32(0))

  # ── Main loop ──
  let kTileMax = size(tAgA.layout.mode(2))
  for kTile in 0 ..< kTileMax:
    # Copy gmem → smem (via thread-partitioned tiles)
    copyFrom(tAsA, tAgA(_, _, kTile))  # A (THR_M, THR_K) -> (THR_M, THR_K)
    copyFrom(tBsB, tBgB(_, _, kTile))  # B (THR_N, THR_K) -> (THR_N, THR_K)
    syncthreads()                      # CuTe: wait for all threads to write smem
    # Compute: C += A * B
    gemm(tCrC, tCsA, tCsB)             # (THR_M, THR_N) += (THR_M, BLK_K) × (THR_N, BLK_K)
    syncthreads()                      # CuTe: wait for all threads to read smem

  # ── Epilogue ──
  axpby(float32(1), tCrC, float32(0), tCgC)       # C = 1·acc + 0·C

# ═════════════════════════════════════════════════════════════════════════
#  GPU kernel (cuda: block for NVRTC)
# ═════════════════════════════════════════════════════════════════════════

const sgemmKernel = cuda:
  proc gemmKernel(
    C: ptr UncheckedArray[float32],
    A: ptr UncheckedArray[float32],
    B: ptr UncheckedArray[float32],
    M, N, K: int32
  ) {.global.} =
    let mA = make_view(A, make_layout((int(M), int(K)), (Int[1](), int(M))))
    let mB = make_view(B, make_layout((int(N), int(K)), (Int[1](), int(N))))
    var mC = make_view(C, make_layout((int(M), int(N)), (Int[1](), int(M))))
    let cta_tiler = (Int[128](), Int[128](), Int[8]())
    let tA = make_layout((Int[32](), Int[8]()))
    let tB = make_layout((Int[32](), Int[8]()))
    let tC = make_layout((Int[16](), Int[16]()))
    sgemm_1_kernel(mA, mB, mC, cta_tiler, tA, tB, tC)

# ═════════════════════════════════════════════════════════════════════════
#  Test
# ═════════════════════════════════════════════════════════════════════════

when isMainModule:
  echo &"Testing sgemm_1 via run_gemm_and_validate_colmajor..."

  echo "════════ kernel ═══════════════════════════════════════════════════════"
  echo sgemmKernel
  echo "═══════════════════════════════════════════════════════════════════════"

  run_gemm_and_validate_colmajor(sgemmKernel, "gemmKernel")
  echo &"  OK — sgemm_1 GPU correctness test passed"
