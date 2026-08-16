## Manual GPU test: gemm_gpu via NVRTC/CUDA.
##
## gemm_gpu: the test kernels build input views over the raw buffers
## and hand them to gemm_gpu.
## gemm_gpu derives the tma policy (atom_selector + threadLayoutOf + tile_shape),
## the CTA position from the ambient blockIdx.x/y,
## and the per-thread epilogue shard, then runs the gemm_cta body.
## Test kernels never see M/N/K, the tile, the thread layout or the grid coords.
##
## TODO: row-major and arbitrary-stride views. The layout algebra is stride-agnostic,
## so the copies can convert any layout into the smem and register arrangements.
## Only col-major fixtures are covered today.
##
## Five kernels run over a 1×1 CTA grid (tile 32×32, tileK 32),
## 256 threads. K64 kernel runs two tileK-sized slices of K.
## Epilogues:
##   gemmGpuKernel          EpiAXPBY,    D = α·AB + β·C
##   gemmGpuIdentityKernel  EpiIdentity, D = AB
##   gemmGpuReLUKernel      EpiReLU,     D = max(0, AB)
##   gemmGpuBiasKernel      EpiAddBias,  D = AB + bias (column broadcast)
## C buffer is pre-filled with NaN: a dropped store leaves NaN != expected,
## and the β=0 branch must skip the C read. A spurious read also fails.
##
## Reference, trial loop and report live in gemm_test_lib (testGemmGpu*),
## launched over a 1×1 CTA grid.
##
## Requires an sm_80+ GPU. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_gpu_cuda.nim --nimcache:nimcache/tests/manual_gemm_gpu_cuda.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_gpu_cuda.nim

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

# Host derives the reference and launch tile/blockSize
# from the same policy gemm_gpu computes internally.
const atom = atom_selector(uint32, uint32, float32)
const tiled = make_tiled_mma(atom, threadLayoutOf(atom, 32, 32))

const kernelCode = cuda:
  proc gemmGpuKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_gpu(pC, pA, pB, initEpiAXPBY(alpha, beta, pC))

  proc gemmGpuK64Kernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    let pA = make_view(A, (32, 64), (1, 32))
    let pB = make_view(B, (32, 64), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_gpu(pC, pA, pB, initEpiAXPBY(alpha, beta, pC))

  proc gemmGpuIdentityKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_gpu(pC, pA, pB, EpiIdentity())

const kernelCodeBias = cuda:
  proc gemmGpuReLUKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_gpu(pC, pA, pB, EpiReLU())

  proc gemmGpuBiasKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      bias: ptr UncheckedArray[float32]) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    # Bias is a (N,) column vector: the bias view has stride-0 rows,
    # so every row of a column reads the same bias element.
    gemm_gpu(pC, pA, pB, initEpiAddBias(make_view(bias, (32, 32), (0, 1))))

proc runTest() =
  var engine = bkCuda.init(kernelCode)
  testGemmGpu(engine, tiled, "SM80")
  testGemmGpuBeta(engine, tiled, "SM80")
  testGemmGpuK64(engine, tiled, "SM80")
  testGemmGpuIdentity(engine, tiled, "SM80")
  var engineBias = bkCuda.init(kernelCodeBias)
  testGemmGpuReLU(engineBias, tiled, "SM80")
  testGemmGpuBias(engineBias, tiled, "SM80")

when isMainModule:
  runTest()
