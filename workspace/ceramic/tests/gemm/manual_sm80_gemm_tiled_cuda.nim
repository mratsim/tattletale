## Manual GPU test: the tiled GEMM (gemm_tiled) via NVRTC/CUDA.
##
## C(32×16) = α·A(32×16)·B(16×16) + β·C. 1×1 grid, K = TILE_K = 16
## (two k slices through gemm_warp), config (α, β) = (1.0, 0.0),
## 128 threads.
##
## gemm_tiled(tma, dFrag, sA, sB, TileShape, threadIdx) computes the
## thread's fragment for one tileK-sized slice of K:
## thread tiling → fragment gathering from the prepared smem tile →
## gemm_warp's loop over the K dimension, accumulating into the
## caller's dFrag.
## Kernel declares its own {.smem.} smem tiles, copies the full
## tileK-sized slice of K from gmem (unmasked: this test runs full
## tiles), syncthreads(), then calls gemm_tiled on the smem views.
## Fused epilogue runs after the call. gemm_tiled leaves the epilogue
## to the caller: EpiAXPBY, preflight + apply, D = α·AB + β·C.
## NaN: the β=0 branch must skip the C read, so a spurious read (or a
## dropped store) produces NaN != expected.
##
## Atom is the parameter, SM80_16x8x8_F32TF32TF32F32_TN. Tiling
## (2×2×1 atoms) and tile geometry derive inside the driver funcs per
## the gemm_tiled convention. The cuda: kernel is a thin wrapper.
## Reference harness lives in gemm_test_lib.
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_sm80_gemm_tiled_cuda.nim --nimcache:nimcache/tests/manual_sm80_gemm_tiled_cuda.nim \
##     workspace/ceramic/tests/gemm/manual_sm80_gemm_tiled_cuda.nim

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
  ## C(32×16) = α·A(32×16)·B(16×16) + β·C: 1×1 tiled m16n8k8 tf32,
  ## 128 threads, K = TILE_K = 16, fused epilogue.
  ## Tile geometry: 2×2×1 atoms over the (2,2,1) thread layout.
  const
    TILE_K = 16                  # the full K, one tileK-sized slice (tileK == K)
    thrM = toIntVal(tma.threadLayout.shape[0])
    thrN = toIntVal(tma.threadLayout.shape[1])
    thrK = toIntVal(tma.threadLayout.shape[2])
    TILE_M = thrM * tma.atom.mnk.m   # 2·16 = 32
    TILE_N = thrN * tma.atom.mnk.n   # 2·8  = 16
    blockSize = toIntVal(tma.atom.threadCount(opA)) * thrM * thrN * thrK
  # gmem col-major: A[m + k·32], B[n + k·16], C[m + n·32]
  let tA = make_view(A, make_layout((TILE_M, TILE_K), (1, TILE_M)))
  let tB = make_view(B, make_layout((TILE_N, TILE_K), (1, TILE_N)))
  var tC = make_view(C, make_layout((TILE_M, TILE_N), (1, TILE_M)))
  # the epilogue op carries alpha/beta and the thread's C fragment view
  let thr = tma.get_slice(threadIdx)
  var tCv = tma.partition_C(thr, tC)
  var epi = initEpiAXPBY(alpha, beta, tCv)
  # accumulator: zeroed once, then gemm_tiled accumulates into it
  var dFrag = make_tensor(float32, tCv.layout.shape)
  dFrag.fillWith(float32(0))
  # prepare the full tileK-sized slice of K gmem → smem, unmasked:
  # this test runs full tiles with no ragged lanes
  var smemA {.smem.}: array[TILE_M * TILE_K, uint32]
  var smemB {.smem.}: array[TILE_N * TILE_K, uint32]
  let sA = make_view(addr smemA[0], make_layout((TILE_M, TILE_K)))
  let sB = make_view(addr smemB[0], make_layout((TILE_N, TILE_K)))
  var o = threadIdx
  while o < TILE_M * TILE_K:
    sA(o) = tA(o)
    o += blockSize
  o = threadIdx
  while o < TILE_N * TILE_K:
    sB(o) = tB(o)
    o += blockSize
  syncthreads()
  tma.gemm_tiled(dFrag, sA, sB, (TILE_M, TILE_N, TILE_K), threadIdx)
  epi.preflight()
  var tmp = make_tensor(float32, dFrag.layout.shape)
  epi.apply(tmp, dFrag)
  epi.finalStore(tCv, tmp)

func gemmTiledMicrotileK32(tma: static TiledMma; threadIdx: int;
                         alpha: float32;
                         C: ptr UncheckedArray[float32];
                         A, B: ptr UncheckedArray[uint32];
                         beta: float32) {.inline.} =
  ## C(32×16) = α·A(32×32)·B(16×32) + β·C: 1×1 tiled m16n8k8 tf32,
  ## 128 threads, K = tileK = 32: four k slices in one gemm_tiled pass.
  const
    TILE_K = 32                  # the full K, one tileK-sized slice (tileK == K)
    thrM = toIntVal(tma.threadLayout.shape[0])
    thrN = toIntVal(tma.threadLayout.shape[1])
    thrK = toIntVal(tma.threadLayout.shape[2])
    TILE_M = thrM * tma.atom.mnk.m   # 2·16 = 32
    TILE_N = thrN * tma.atom.mnk.n   # 2·8  = 16
    blockSize = toIntVal(tma.atom.threadCount(opA)) * thrM * thrN * thrK
  # gmem col-major: A[m + k·32], B[n + k·32]
  let tA = make_view(A, make_layout((TILE_M, TILE_K), (1, TILE_M)))
  let tB = make_view(B, make_layout((TILE_N, TILE_K), (1, TILE_N)))
  var tC = make_view(C, make_layout((TILE_M, TILE_N), (1, TILE_M)))
  let thr = tma.get_slice(threadIdx)
  var tCv = tma.partition_C(thr, tC)
  var epi = initEpiAXPBY(alpha, beta, tCv)
  # accumulator: zeroed once, then gemm_tiled accumulates into it
  var dFrag = make_tensor(float32, tCv.layout.shape)
  dFrag.fillWith(float32(0))
  # prepare the full tileK-sized slice of K gmem → smem, unmasked:
  # this test runs full tiles with no ragged lanes
  var smemA {.smem.}: array[TILE_M * TILE_K, uint32]
  var smemB {.smem.}: array[TILE_N * TILE_K, uint32]
  let sA = make_view(addr smemA[0], make_layout((TILE_M, TILE_K)))
  let sB = make_view(addr smemB[0], make_layout((TILE_N, TILE_K)))
  var o = threadIdx
  while o < TILE_M * TILE_K:
    sA(o) = tA(o)
    o += blockSize
  o = threadIdx
  while o < TILE_N * TILE_K:
    sB(o) = tB(o)
    o += blockSize
  syncthreads()
  tma.gemm_tiled(dFrag, sA, sB, (TILE_M, TILE_N, TILE_K), threadIdx)
  epi.preflight()
  var tmp = make_tensor(float32, dFrag.layout.shape)
  epi.apply(tmp, dFrag)
  epi.finalStore(tCv, tmp)

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
