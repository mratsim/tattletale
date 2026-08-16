## Manual GPU test: gemm_cta's loop over the K dimension via NVRTC/CUDA.
##
## C(64×32) = α·A(64×64)·B(32×64) + β·C over a 2×2 CTA grid with two
## tileK-sized slices of K (K = 64 = 2·tileK, tileK = 32), 128 threads.
## Each CTA gets a full-K (32, 64) / (16, 64) tile view. gemm_cta
## accumulates the two slices of K into the persistent accumulator and
## runs the fused epilogue once (gemm_tiled computes one slice each).
## Reference, trial loop and report live in gemm_test_lib
## (testGemmCtaK64).
##
## Atom parameter: SM80_16x8x8_F32TF32TF32F32_TN tiled (2,2,1).
## Requires an sm_80+ GPU. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_cta_k64_cuda.nim --nimcache:nimcache/tests/manual_gemm_cta_k64_cuda.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_cta_k64_cuda.nim

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
  # 1D grid (engine LaunchConfig.grid is a single int): blk = blockIdx.x,
  # 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2.
  proc gemmCtaK64Kernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 64), (1, 64))
    let pB = make_view(B, (32, 64), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 64, epi, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

proc runTest() =
  var engine = bkCuda.init(kernelCode)
  testGemmCtaK64(engine, tiled, "SM80")

when isMainModule:
  runTest()
