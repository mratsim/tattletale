## Manual GPU test: gemm_grid (Level 4) via the OpenCL backend.
##
## Same problems as manual_gemm_grid_cuda.nim: the four shipped epilogues
## (EpiAXPBY, EpiIdentity, EpiReLU, EpiAddBias) over a 2×2 CTA grid (tile
## 32×16, K = 32), 128 work-items per CTA. The kernel body is the CUDA
## twin's verbatim; the launch geometry is linearized: the engine launches
## grid.x·grid.y·blockSize work-items and the kernel decomposes the linear
## work-item id into (mCTA, nCTA, threadIdx). Work-groups are pinned to
## one warp: the mma.sync epilogue path miscomputes in multi-warp groups
## on NVIDIA's OpenCL. The oracle, trial loop and report live in
## gemm_test_lib (testGemmGrid*), shared with the CUDA twin.
##
## NVIDIA-OpenCL only: the mma.sync inline PTX travels inside the OpenCL C
## kernel as asm(...), which only NVIDIA's OpenCL compiler accepts. The
## host verifies the device vendor before launching.
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run with:
##   nim cpp -r workspace/ceramic/tests/gemm/manual_gemm_grid_opencl.nim

import std/[strformat, strutils]
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
import workspace/crucible/src/codegen/cl

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

# Work-group pinned to one warp multiple: mma.sync is warp-synchronous,
# each 32-lane warp must execute it convergently. 128 = 4 warps.
const blockSize = 128
static:
  doAssert blockSize == toIntVal(tiled.atom.threadCount(opA)) * 2 * 2 * 1

const kernelCode = opencl:
  # Each kernel wrapper decomposes the linear work-item id into
  # (mCTA, nCTA, threadIdx), builds the thread's C tile view (tile origin
  # from the grid coords), partitions it to the per-thread destination
  # fragment (D), constructs the epilogue op with its state, and hands
  # both to gemm_grid. The grid never sees alpha/beta/C/bias: they are
  # op state. Inputs-first (A, B, ...), output last (C): the OpenCL
  # engine's execOpenCL binding convention.
  proc gemmGridKernel(
      A, B: ptr UncheckedArray[uint32];
      alpha, beta: float32;
      C: ptr UncheckedArray[float32]) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, threadIdx)

  proc gemmGridIdentityKernel(
      A, B: ptr UncheckedArray[uint32];
      C: ptr UncheckedArray[float32]) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiIdentity()
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, threadIdx)

  proc gemmGridReLUKernel(
      A, B: ptr UncheckedArray[uint32];
      C: ptr UncheckedArray[float32]) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiReLU()
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, threadIdx)

  proc gemmGridBiasKernel(
      A, B: ptr UncheckedArray[uint32];
      bias: ptr UncheckedArray[float32];
      C: ptr UncheckedArray[float32]) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var tBias = make_view(bias +% (m0 + n0 * 64), (32, 16), (1, 64))
    let biasCv = tiled.partition_C(thr, tBias)
    var epi = initEpiAddBias(biasCv)
    gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), mCTA, nCTA, threadIdx)

  proc gemmGridKernelSingle(
      A, B: ptr UncheckedArray[uint32];
      alpha, beta: float32;
      C: ptr UncheckedArray[float32]) {.global.} =
    ## 1×1 CTA grid: only the thread id varies.
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    var tC = make_view(C +% 0, (32, 16), (1, 32))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_grid(tiled, tCv, A, 32, B, 16, epi, 32, 16, 32, (32, 16, 32), 0, 0, threadIdx)

proc main() =
  var engine = "opencl".getEngine(kernelCode)
  doAssert engine.ctx.device.vendor().contains("NVIDIA"),
    "gemm_grid OpenCL needs NVIDIA's OpenCL compiler for the mma.sync asm; got: " &
    engine.ctx.device.vendor()
  testGemmGrid(engine, tiled, "SM80")
  testGemmGridBeta(engine, tiled, "SM80")
  testGemmGridIdentity(engine, tiled, "SM80")
  testGemmGridReLU(engine, tiled, "SM80")
  testGemmGridBias(engine, tiled, "SM80")
  testGemmGridSingle(engine, tiled, "SM80")

when isMainModule:
  main()
