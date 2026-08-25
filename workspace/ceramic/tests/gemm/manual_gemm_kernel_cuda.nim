## Manual GPU test: gemm_kernel via NVRTC/CUDA.
##
## gemm_kernel: the test kernels build input views over the raw buffers
## and hand them to gemm_kernel.
## gemm_kernel derives the tma policy (atom_selector + threadLayoutOf + tile_shape),
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
##   gemmKernelAXPBY        EpiAXPBY,    D = α·AB + β·C
##   gemmKernelIdentity     EpiIdentity, D = AB
##   gemmKernelReLU         EpiReLU,     D = max(0, AB)
##   gemmKernelBias         EpiAddBias,  D = AB + bias (column broadcast)
## C buffer is pre-filled with NaN: a dropped store leaves NaN != expected,
## and the β=0 branch must skip the C read. A spurious read also fails.
##
## Reference, trial loop and report live in gemm_test_lib (testGemmKernel*),
## launched over a 1×1 CTA grid.
##
## Requires an sm_80+ GPU. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_kernel_cuda.nim --nimcache:nimcache/tests/manual_gemm_kernel_cuda.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_kernel_cuda.nim

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry
import workspace/ceramic/src/hardware/h_properties

import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible

{.experimental: "callOperator".}

# Host derives the reference and launch tile/blockSize
# from the same policy gemm_kernel computes internally.
const atom = atom_selector(uint32, uint32, float32)
const tiled = make_tiled_mma(atom, threadLayoutOf(atom, 32, 32))

const kernelCode = cuda:
  proc gemmKernelAXPBY(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_kernel(pC, pA, pB, initEpiAXPBY(alpha, beta, pC))

  proc gemmKernelK64(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    let pA = make_view(A, (32, 64), (1, 32))
    let pB = make_view(B, (32, 64), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_kernel(pC, pA, pB, initEpiAXPBY(alpha, beta, pC))

  proc gemmKernelIdentity(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_kernel(pC, pA, pB, EpiIdentity())

const kernelCodeBias = cuda:
  proc gemmKernelReLU(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    gemm_kernel(pC, pA, pB, EpiReLU())

  proc gemmKernelBias(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      bias: ptr UncheckedArray[float32]) {.global.} =
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (32, 32), (1, 32))
    var pC = make_view(C, (32, 32), (1, 32))
    # Bias is a (N,) column vector: the bias view has stride-0 rows,
    # so every row of a column reads the same bias element.
    gemm_kernel(pC, pA, pB, initEpiAddBias(make_view(bias, (32, 32), (0, 1))))

proc runTest() =
  var engine = bkCuda.init(kernelCode)
  testGemmKernel(engine, tiled, "SM80")
  testGemmKernelBeta(engine, tiled, "SM80")
  testGemmKernelK64(engine, tiled, "SM80")
  testGemmKernelIdentity(engine, tiled, "SM80")
  var engineBias = bkCuda.init(kernelCodeBias)
  testGemmKernelReLU(engineBias, tiled, "SM80")
  testGemmKernelBias(engineBias, tiled, "SM80")

when isMainModule:
  runTest()
