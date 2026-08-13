## Manual GPU test: the tiled GEMM (gemm_tiled) via NVRTC/CUDA.
##
## C(32×16) = α·A(32×16)·B(16×16) + β·C. 1×1 grid, K = TILE_K = 16
## (two k_blocks through gemm_ukernel), config (α, β) = (1.0, 0.0),
## 128 threads.
##
## gemm_tiled(tma, dFrag, A, B, TileShape, threadIdx) = tiling + thread
## decomposition + fragment gathering (one k-tile: K == BLK_K) + the
## k_block loop in gemm_ukernel, accumulating into the caller's dFrag.
## The fused epilogue (EpiAXPBY: preflight + apply, D = α·AB + β·C) runs
## after the call (gemm_cta owns the accumulator + epilogue in the
## production path).
## NaN: the β=0 branch must skip the C read, so a spurious read (or a
## dropped store) produces NaN != expected.
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
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/ceramic/src/kernel_fillwith_gpu
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

func gemmTiledMicrotile(tma: static TiledMma; threadIdx: int;
                     alpha: float32;
                     C: ptr UncheckedArray[float32];
                     A, B: ptr UncheckedArray[uint32];
                     beta: float32) {.inline.} =
  ## C(32×16) = α·A(32×16)·B(16×16) + β·C — 1×1 tiled m16n8k8 tf32,
  ## 128 threads, K = TILE_K = BLK_K = 16, fused epilogue.
  ## Tile geometry: 2×2×1 atoms over the (2,2,1) thread layout.
  const
    TILE_K = 16                  # the full K and the k-tile depth (BLK_K == K)
    thrM = toIntVal(tma.threadLayout.shape[0])
    thrN = toIntVal(tma.threadLayout.shape[1])
    thrK = toIntVal(tma.threadLayout.shape[2])
    TILE_M = thrM * tma.atom.mnk.m   # 2·16 = 32
    TILE_N = thrN * tma.atom.mnk.n   # 2·8  = 16
  # gmem col-major: A[m + k·32], B[n + k·16], C[m + n·32]
  let tA = make_view(A, make_layout((TILE_M, TILE_K), (1, TILE_M)))
  let tB = make_view(B, make_layout((TILE_N, TILE_K), (1, TILE_N)))
  var tC = make_view(C, make_layout((TILE_M, TILE_N), (1, TILE_M)))
  # the epilogue op carries alpha/beta and the thread's C fragment view
  let thr = tma.get_slice(threadIdx)
  var tCv = tma.partition_C(thr, tC)
  var epi = initEpiAXPBY(alpha, beta, tCv)
  # the accumulator: zeroed once, gemm_tiled accumulates into it, the
  # epilogue runs once after (gemm_cta owns this in the production path)
  var dFrag = make_tensor(float32, tCv.layout.shape)
  dFrag.fillWith(float32(0))
  tma.gemm_tiled(dFrag, tA, tB, (TILE_M, TILE_N, TILE_K), threadIdx)
  epi.preflight()
  epi.apply(tCv, dFrag)

func gemmTiledMicrotileK32(tma: static TiledMma; threadIdx: int;
                         alpha: float32;
                         C: ptr UncheckedArray[float32];
                         A, B: ptr UncheckedArray[uint32];
                         beta: float32) {.inline.} =
  ## C(32×16) = α·A(32×32)·B(16×32) + β·C — 1×1 tiled m16n8k8 tf32,
  ## 128 threads, K = 32 = BLK_K (four k_blocks through one gemm_ukernel
  ## call in the single k-tile pass.
  const
    TILE_K = 32                  # the full K and the k-tile depth (BLK_K == K)
    thrM = toIntVal(tma.threadLayout.shape[0])
    thrN = toIntVal(tma.threadLayout.shape[1])
    thrK = toIntVal(tma.threadLayout.shape[2])
    TILE_M = thrM * tma.atom.mnk.m   # 2·16 = 32
    TILE_N = thrN * tma.atom.mnk.n   # 2·8  = 16
  # gmem col-major: A[m + k·32], B[n + k·32]
  let tA = make_view(A, make_layout((TILE_M, TILE_K), (1, TILE_M)))
  let tB = make_view(B, make_layout((TILE_N, TILE_K), (1, TILE_N)))
  var tC = make_view(C, make_layout((TILE_M, TILE_N), (1, TILE_M)))
  let thr = tma.get_slice(threadIdx)
  var tCv = tma.partition_C(thr, tC)
  var epi = initEpiAXPBY(alpha, beta, tCv)
  # the accumulator: zeroed once, gemm_tiled accumulates into it, the
  # epilogue runs once after (gemm_cta owns this in the production path)
  var dFrag = make_tensor(float32, tCv.layout.shape)
  dFrag.fillWith(float32(0))
  tma.gemm_tiled(dFrag, tA, tB, (TILE_M, TILE_N, TILE_K), threadIdx)
  epi.preflight()
  epi.apply(tCv, dFrag)

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

proc runTest() =
  var engine = bkCuda.init(kernelCode)
  testTiled(engine, tiled, "SM80")
  var engineK32 = bkCuda.init(kernelCodeK32)
  testTiledMultiBlock(engineK32, tiled, "SM80")

when isMainModule:
  runTest()
