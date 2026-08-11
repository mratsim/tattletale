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
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/ceramic/experiments/experiment_testutils
import workspace/crucible/src/codegen/nvrtc

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Device kernel body (faithful sgemm_1.cu port)
# ═════════════════════════════════════════════════════════════════════════

proc sgemm_1_kernel(
       mA, mB, mC: distinct TensorView,
       cta_tiler: auto,
       tA, tB, tC: distinct Layout,
       alpha, beta: float32) =
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
  # CuTe: make_coord(blockIdx.x, blockIdx.y, _)
  # blockIdx.x/y are plain int by the CUDA stub, matching the layout
  # templates' int/Int params without casts.
  let cta_coord = (blockIdx.x, blockIdx.y, X())

  # ── CTA tile extraction (with Step) ──
  #   mA: (M,K)  cta_tiler: (BLK_M, BLK_N, BLK_K)
  #   Step (_1, X, _1): keep M and K, drop N
  let gA = local_tile(mA, cta_tiler, cta_coord, (Y, X, Y))  # (BLK_M, BLK_K, k)
  #   Step (X, _1, _1): keep N and K, drop M
  let gB = local_tile(mB, cta_tiler, cta_coord, (X, Y, Y))  # (BLK_N, BLK_K, k)
  #   Step (_1, _1, X): keep M and N, drop K
  let gC = local_tile(mC, cta_tiler, cta_coord, (Y, Y, X))  # (BLK_M, BLK_N)

  # ── Shared memory tiles ──
  # CuTe: __shared__ smemA/smemB + make_tensor(make_smem_ptr(...)).
  # make_tensor_like would allocate a PER-THREAD local array (each thread
  # would only ever see its own rake) — real GEMM smem must be block-shared.
  var smemA {.shared.}: array[128 * 8, float32]  # (BLK_M, BLK_K)
  var smemB {.shared.}: array[128 * 8, float32]  # (BLK_N, BLK_K)
  let sA = make_view(addr smemA[0], make_layout((128, 8)))  # (BLK_M, BLK_K)
  let sB = make_view(addr smemB[0], make_layout((128, 8)))  # (BLK_N, BLK_K)

  # ── A/B thread partitioning (3-arg) ──
  # CuTe: local_partition(gA, tA, threadIdx.x)
  let tAgA = local_partition(gA, tA, threadIdx.x)  # (THR_M, THR_K, k)
  var tAsA = local_partition(sA, tA, threadIdx.x)  # (THR_M, THR_K)
  let tBgB = local_partition(gB, tB, threadIdx.x)  # (THR_N, THR_K, k)
  var tBsB = local_partition(sB, tB, threadIdx.x)  # (THR_N, THR_K)

  # ── C thread partitioning (4-arg with Step) ──
  #   sA: (BLK_M, BLK_K), tC: (THR_M, THR_N)
  #   Step (_1, X): partition M by tC mode 0, keep K whole
  let tCsA = local_partition(sA, tC, threadIdx.x, (Y, X))  # (THR_M, BLK_K)
  #   sB: (BLK_N, BLK_K)
  #   Step (X, _1): keep N whole, partition K by tC mode 1
  let tCsB = local_partition(sB, tC, threadIdx.x, (X, Y))  # (THR_N, BLK_K)
  #   gC: (BLK_M, BLK_N)
  #   Step (_1, _1): partition both modes
  var tCgC = local_partition(gC, tC, threadIdx.x, (Y, Y))  # (THR_M, THR_N)

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
    gemm_ref(tCrC, tCsA, tCsB)   # (THR_M, THR_N) += (THR_M, BLK_K) × (THR_N, BLK_K)
    syncthreads()                      # CuTe: wait for all threads to read smem

  # ── Epilogue ──
  # C = alpha·acc + beta·C — matches CuTe sgemm_1's gemm_device, which
  # takes alpha/beta as kernel params and does axpby(alpha, tCrC, beta, tCgC)
  # (the tutorial's main() happens to pass alpha=1, beta=0).
  var epi = initEpiAXPBY(alpha, beta, tCgC)
  epi.preflight()
  epi.apply(tCgC, tCrC)

# ═════════════════════════════════════════════════════════════════════════
#  GPU kernel (cuda: block for NVRTC)
# ═════════════════════════════════════════════════════════════════════════

const sgemmKernel = cuda:
  proc gemmKernel(
    C: ptr UncheckedArray[float32],
    A: ptr UncheckedArray[float32],
    B: ptr UncheckedArray[float32],
    M, N, K: int32,
    alpha, beta: float32
  ) {.global.} =
    # alpha/beta: generic GEMM contract C = alpha·(A@B) + beta·C — same
    # signature as CuTe gemm_device(..., Alpha alpha, Beta beta).
    let mA = make_view(A, make_layout((int(M), int(K)), (1, int(M))))
    let mB = make_view(B, make_layout((int(N), int(K)), (1, int(N))))
    var mC = make_view(C, make_layout((int(M), int(N)), (1, int(M))))
    let cta_tiler = makeIntTuple((128, 128, 8))
    let tA = make_layout((32, 8))
    let tB = make_layout((32, 8))
    let tC = make_layout((16, 16))
    sgemm_1_kernel(mA, mB, mC, cta_tiler, tA, tB, tC, alpha, beta)

# ═════════════════════════════════════════════════════════════════════════
#  Test
# ═════════════════════════════════════════════════════════════════════════

when isMainModule:
  echo &"Testing sgemm_1 via run_gemm_and_validate_colmajor..."

  # echo "════════ kernel ═══════════════════════════════════════════════════════"
  # echo sgemmKernel
  # echo "═══════════════════════════════════════════════════════════════════════"

  run_gemm_and_validate_colmajor(sgemmKernel, "gemmKernel")
  echo &"  OK — sgemm_1 GPU correctness test passed"
