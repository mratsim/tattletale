## Manual GPU test: gemm_cta (Level 4) via NVRTC/CUDA.
##
## Four shipped epilogues over a 2×2 CTA grid (tile 32×16,
## K = TILE_K = 32), 128 threads:
##   gemmCtaKernel       EpiAXPBY,   D = α·AB + β·C
##   gemmCtaIdentityKernel  EpiIdentity, D = AB
##   gemmCtaReLUKernel   EpiReLU,    D = max(0, AB)
##   gemmCtaBiasKernel   EpiAddBias, D = AB + bias (column broadcast)
## C is prefilled with NaN: a dropped store leaves NaN in the output,
## and the β=0 path must skip the C read (a spurious read also fails).
##
## Each kernel builds (M, K), (N, K), (M, N) input views over the raw
## buffers. gemm_cta slices its tile grid (flat_divide) by CTA
## coordinate, so the tile origin comes from the layout, not manual
## pointer math. Reference, trial loop and report live in gemm_test_lib
## (testGemmCta*).
##
## Atom parameter: SM80_16x8x8_F32TF32TF32F32_TN tiled (2,2,1).
## Requires an sm_80+ GPU. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_cta_cuda.nim --nimcache:nimcache/tests/manual_gemm_cta_cuda.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_cta_cuda.nim

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
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

const kernelCode = cuda:
  # Epilogue-op state: alpha/beta/C/bias never appear as kernel arguments.
  proc gemmCtaKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    # 1D grid (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2.
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

  proc gemmCtaIdentityKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiIdentity()
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

const kernelCodeBias = cuda:
  proc gemmCtaReLUKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiReLU()
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

  proc gemmCtaBiasKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      bias: ptr UncheckedArray[float32]) {.global.} =
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    # Bias is a (N,) column vector: the input view has stride-0 rows, so
    # every row of a column reads the same bias element. The view is
    # partitioned like C, which bakes each thread's column base into the
    # fragment's data pointer. The (mCTA, nCTA) tile origin derives from
    # the layout.
    let pBias = make_view(bias, (64, 32), (0, 1))
    var biasView = tiled.partition_C(thr, local_tile(pBias, (32, 16), (mCTA, nCTA)))
    var epi = initEpiAddBias(biasView)
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

const kernelCodeSingle = cuda:
  proc gemmCtaKernelSingle(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    # 1×1 CTA grid: only the thread id varies
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (16, 32), (1, 16))
    let pC = make_view(C, (32, 16), (1, 32))
    let tC = local_tile(pC, (32, 16), (0, 0))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, 32, 16, 32, epi, (32, 16, 32), 0, 0, int(threadIdx.x))

proc runTest() =
  var engine = bkCuda.init(kernelCode)
  testGemmCta(engine, tiled, "SM80")
  testGemmCtaBeta(engine, tiled, "SM80")
  testGemmCtaIdentity(engine, tiled, "SM80")
  var engineBias = bkCuda.init(kernelCodeBias)
  testGemmCtaReLU(engineBias, tiled, "SM80")
  testGemmCtaBias(engineBias, tiled, "SM80")
  var engineSingle = bkCuda.init(kernelCodeSingle)
  testGemmCtaSingle(engineSingle, tiled, "SM80")

when isMainModule:
  runTest()
