## Manual GPU test: the tiled GEMM (gemm_tiled) via NVRTC/CUDA.
##
## C(32×16) = α·A(32×16)·B(16×16) + β·C — 1×1 grid, single k-block
## (K = TILE_K = 16), config (α, β) = (1.0, 0.0), 128 threads.
##
## gemm_tiled(tma, threadIdx, alpha, A, B, beta, C) = tiling + thread
## decomposition + k-block loop over gemm_ukernel, with a fused
## α·(A·B) + β·C epilogue. The C buffer is pre-filled with NaN: the β=0
## branch must skip the C read, so a spurious read (or a dropped store)
## produces NaN != expected.
##
## The atom is the parameter — SM80_16x8x8_F32TF32TF32F32_TN; the tiling
## (2×2×1 atoms) and tile geometry are derived inside the driver func
## (gemm_tiled convention); the cuda: kernel is a thin wrapper. The oracle
## harness lives in gemm_test_lib.
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm80_gemm_tiled_cuda.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible/src/codegen/nvrtc

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

func gemmTiledMicrotile(tma: static TiledMma; threadIdx: int;
                     alpha: float32;
                     C: ptr UncheckedArray[float32];
                     A, B: ptr UncheckedArray[uint32];
                     beta: float32) {.inline.} =
  ## C(32×16) = α·A(32×16)·B(16×16) + β·C — 1×1 tiled m16n8k8 tf32,
  ## 128 threads, single k-block (K = TILE_K = 16), fused epilogue.
  ## Tile geometry: 2×2×1 atoms over the (2,2,1) thread layout.
  const
    TILE_K = 16                  # k-block size in elements
    thrM = toIntVal(tma.threadLayout.shape[0])
    thrN = toIntVal(tma.threadLayout.shape[1])
    thrK = toIntVal(tma.threadLayout.shape[2])
    TILE_M = thrM * tma.atom.mnk.m   # 2·16 = 32
    TILE_N = thrN * tma.atom.mnk.n   # 2·8  = 16
  # gmem col-major: A[m + k·32], B[n + k·16], C[m + n·32]
  let tA = make_view(A, make_layout((TILE_M, TILE_K), (1, TILE_M)))
  let tB = make_view(B, make_layout((TILE_N, TILE_K), (1, TILE_N)))
  var tC = make_view(C, make_layout((TILE_M, TILE_N), (1, TILE_M)))
  tma.gemm_tiled(threadIdx, alpha, tA, tB, beta, tC, TILE_K)

func gemmTiledMicrotileK32(tma: static TiledMma; threadIdx: int;
                         alpha: float32;
                         C: ptr UncheckedArray[float32];
                         A, B: ptr UncheckedArray[uint32];
                         beta: float32) {.inline.} =
  ## C(32×16) = α·A(32×32)·B(16×32) + β·C — 1×1 tiled m16n8k8 tf32,
  ## 128 threads, TWO k-blocks of 16 (K=32, BLK_K=16) — exercises the kb
  ## loop (F1: each block's fragment must span only its slicesPerBlock).
  const
    TILE_K = 32                  # full K extent
    BLK_K = 16                   # k-block size passed to gemm_tiled
    thrM = toIntVal(tma.threadLayout.shape[0])
    thrN = toIntVal(tma.threadLayout.shape[1])
    thrK = toIntVal(tma.threadLayout.shape[2])
    TILE_M = thrM * tma.atom.mnk.m   # 2·16 = 32
    TILE_N = thrN * tma.atom.mnk.n   # 2·8  = 16
  # gmem col-major: A[m + k·32], B[n + k·32]
  let tA = make_view(A, make_layout((TILE_M, TILE_K), (1, TILE_M)))
  let tB = make_view(B, make_layout((TILE_N, TILE_K), (1, TILE_N)))
  var tC = make_view(C, make_layout((TILE_M, TILE_N), (1, TILE_M)))
  tma.gemm_tiled(threadIdx, alpha, tA, tB, beta, tC, BLK_K)

const kernelCode = cuda:
  proc gemmTiledKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    gemmTiledMicrotile(tiled, int(threadIdx.x), alpha, C, A, B, beta)

const kernelCodeK32 = cuda:
  proc gemmTiledKernelK32(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    gemmTiledMicrotileK32(tiled, int(threadIdx.x), alpha, C, A, B, beta)

when isMainModule:
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()
  testTiled(nv, tiled, "SM80")
  var nv2 = initNvrtc(kernelCodeK32)
  nv2.compile()
  nv2.getPtx()
  testTiledMultiBlock(nv2, tiled, "SM80")
