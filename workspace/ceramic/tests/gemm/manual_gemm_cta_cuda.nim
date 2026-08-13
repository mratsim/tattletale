## Manual GPU test: gemm_cta (Level 4) via NVRTC/CUDA.
##
## The four shipped epilogues over a 2×2 CTA grid (tile 32×16, K =
## TILE_K = 32, four k_blocks through gemm_ukernel), 128 threads:
##   gemmCtaKernel       EpiAXPBY,   D = α·AB + β·C
##   gemmCtaIdentityKernel  EpiIdentity, D = AB
##   gemmCtaReLUKernel   EpiReLU,    D = max(0, AB)
##   gemmCtaBiasKernel   EpiAddBias, D = AB + bias (column broadcast)
## The C buffer is pre-filled with NaN: a dropped store leaves NaN !=
## expected, and the β=0 branch must skip the C read (a spurious read
## also fails).
##
## The kernels build (M, K), (N, K), (M, N) problem views over the raw
## buffers; gemm_cta slices its tile grid (flat_divide) by CTA
## coordinate, so the tile origin comes from the layout, no manual
## pointer math. The oracle, trial loop and report live in
## gemm_test_lib (testGemmCta*), shared with the OpenCL twin
## (manual_gemm_cta_opencl).
##
## The atom is the parameter, SM80_16x8x8_F32TF32TF32F32_TN tiled
## (2,2,1). Requires an sm_80+ GPU. Run with:
##   nim cpp -r workspace/ceramic/tests/gemm/manual_gemm_cta_cuda.nim

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
  # Each kernel wrapper builds the problem views, derives the CTA's C
  # tile by local_tile over the C problem layout (the origin comes from
  # the grid coords, baked by crd2idx), partitions it to the per-thread
  # destination fragment (D), constructs the epilogue op with its state,
  # and hands both to gemm_cta. The grid never sees alpha/beta/C/bias:
  # they are op state.
  proc gemmCtaKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
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
    gemm_cta(tiled, tCv, pA, pB, 64, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

  proc gemmCtaIdentityKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
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
    gemm_cta(tiled, tCv, pA, pB, 64, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

const kernelCodeBias = cuda:
  proc gemmCtaReLUKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
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
    gemm_cta(tiled, tCv, pA, pB, 64, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

  proc gemmCtaBiasKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      bias: ptr UncheckedArray[float32]) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    # The bias is a (N,) column vector: the problem view has stride-0 rows
    # (every row of a column reads the same bias element) and the same
    # partition as C, which bakes each thread's column base into the
    # fragment's data pointer. The tile grid over the (64, 32) bias
    # problem layout and the (mCTA, nCTA) slice derive the tile origin by
    # layout algebra, no +% pointer math.
    let pBias = make_view(bias, (64, 32), (0, 1))
    var biasView = tiled.partition_C(thr, local_tile(pBias, (32, 16), (mCTA, nCTA)))
    var epi = initEpiAddBias(biasView)
    gemm_cta(tiled, tCv, pA, pB, 64, 32, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

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
    gemm_cta(tiled, tCv, pA, pB, 32, 16, epi, (32, 16, 32), 0, 0, int(threadIdx.x))

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
