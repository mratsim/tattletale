## Manual GPU test: gemm_grid (Level 4) via NVRTC/CUDA.
##
## The four shipped epilogues over a 2×2 CTA grid (tile 32×16, K =
## TILE_K = 32, four k_blocks through gemm_ukernel), 128 threads:
##   gemmGridKernel       EpiAXPBY,   D = α·AB + β·C
##   gemmGridIdentityKernel  EpiIdentity, D = AB
##   gemmGridReLUKernel   EpiReLU,    D = max(0, AB)
##   gemmGridBiasKernel   EpiAddBias, D = AB + bias (column broadcast)
## The C buffer is pre-filled with NaN: a dropped store leaves NaN !=
## expected, and the β=0 branch must skip the C read (a spurious read
## also fails).
##
## gemm_grid computes the tile origin from blockIdx (baking it into the
## A/B/C view pointers, the real global col-major strides stay), then
## delegates to gemm_tiled. The kernel below launches it directly. The
## oracle, trial loop and report live in gemm_test_lib (testGemmGrid*),
## shared with the OpenCL twin (manual_gemm_grid_opencl).
##
## The atom is the parameter, SM80_16x8x8_F32TF32TF32F32_TN tiled
## (2,2,1). Requires an sm_80+ GPU. Run with:
##
## TODO (file-level slop): M=64, N=32, tile (32,16), strides (1,64) and
## the m0/n0 tile origins are hardcoded in every kernel below — the dims
## must flow from the problem views and the origins from local_tile, per
## the gemm_grid view migration.
##   nim cpp -r workspace/ceramic/tests/gemm/manual_gemm_grid_cuda.nim

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
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

const kernelCode = cuda:
  # Each kernel wrapper builds the thread's C tile view (tile origin from
  # blockIdx), partitions it to the per-thread destination fragment (D),
  # constructs the epilogue op with its state, and hands both to
  # gemm_grid. The grid never sees alpha/beta/C/bias: they are op state.
  proc gemmGridKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

  proc gemmGridIdentityKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiIdentity()
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

  proc gemmGridReLUKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiReLU()
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

  proc gemmGridBiasKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      bias: ptr UncheckedArray[float32]) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    # The bias is a (N,) column vector: the tile's view has stride-0 rows
    # (every row of a column reads the same bias element) and the same
    # partition as C, which bakes each thread's column base into the
    # fragment's data pointer.
    # TODO: the tile origin (n0) and the hardcoded M/N/tile/strides in
    # this file are manual slop — the tile view must be auto-derived via
    # local_tile on the problem views (make_view(bias, (M, N), (0, 1)))
    # with the dims flowing from the views, per the gemm_grid view
    # migration.
    var biasView = tiled.partition_C(thr, make_view(bias +% n0, (32, 16), (0, 1)))
    var epi = initEpiAddBias(biasView)
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, int(threadIdx.x))

const kernelCodeSingle = cuda:
  proc gemmGridKernelSingle(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    # 1D grid linearization (engine LaunchConfig.grid is a single int):
    # blk = blockIdx.x, 2x2 CTA grid → mCTA = blk mod 2, nCTA = blk div 2
    # (same decomposition as the OpenCL twin's get_global_id path).
    let blk = int(blockIdx.x)
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 32), (32, 16), (1, 32))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_grid(tiled, tCv, A, 32, B, 16, epi, 32, 16, 32, (32, 16, 32), 0, 0, int(threadIdx.x))

proc main() =
  var engine = bkCuda.init(kernelCode)
  testGemmGrid(engine, tiled, "SM80")
  testGemmGridBeta(engine, tiled, "SM80")
  testGemmGridIdentity(engine, tiled, "SM80")
  testGemmGridReLU(engine, tiled, "SM80")
  testGemmGridBias(engine, tiled, "SM80")
  var engineSingle = bkCuda.init(kernelCodeSingle)
  testGemmGridSingle(engineSingle, tiled, "SM80")

when isMainModule:
  main()
