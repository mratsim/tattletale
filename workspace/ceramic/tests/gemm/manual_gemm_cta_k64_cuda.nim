## Manual GPU test: the k-tile loop through gemm_cta via NVRTC/CUDA.
##
## C(64×32) = α·A(64×64)·B(32×64) + β·C over a 2×2 CTA grid with TWO
## k-tiles: K = 64 = 2·BLK_K with BLK_K = 32, 128 threads. gemm_cta
## hands each CTA a full-K (32, 64) / (16, 64) tile view; gemm_cta
## loops the two k-tiles into the persistent accumulator and runs the
## fused epilogue once (gemm_tiled computes one k-tile each). The
## oracle, trial loop and report live in
## gemm_test_lib (testGemmCtaK64), shared with the OpenCL twin
## (manual_gemm_cta_k64_opencl).
##
## A separate file from manual_gemm_cta_cuda: the K64 kernel inlines the
## full gemm_tiled machinery into the OpenCL twin's module, and one
## module per kernel keeps the OpenCL codegen's compile-time AST walk
## within budget.
##
## TODO (file-level slop): M=64, N=32, tile (32,16), strides (1,64) and
## the m0/n0 tile origins are hardcoded in the kernel below, as in
## manual_gemm_cta_cuda. The dims must flow from the problem views and
## the origins from local_tile, per the gemm_cta view migration.
## The atom is the parameter, SM80_16x8x8_F32TF32TF32F32_TN tiled
## (2,2,1). Requires an sm_80+ GPU. Run with:
##   nim cpp -r workspace/ceramic/tests/gemm/manual_gemm_cta_k64_cuda.nim

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
  # 1D grid linearization (engine LaunchConfig.grid is a single int):
  # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
  # (same decomposition as the OpenCL twin's get_global_id path).
  proc gemmCtaK64Kernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, A, 64, B, 32, epi, 64, 32, 64, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

proc runTest() =
  var engine = bkCuda.init(kernelCode)
  testGemmCtaK64(engine, tiled, "SM80")

when isMainModule:
  runTest()
