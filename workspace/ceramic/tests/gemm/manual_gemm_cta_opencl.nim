## Manual GPU test: gemm_cta (Level 4) via the OpenCL backend.
##
## Four shipped epilogues (EpiAXPBY, EpiIdentity, EpiReLU, EpiAddBias)
## over a 2×2 CTA grid (tile 32×16, K = 32), 128 work-items per CTA.
## Launch geometry is linearized: the engine launches
## grid.x·grid.y·blockSize work-items and the kernel decomposes the
## linear work-item id into (mCTA, nCTA, threadIdx).
## Reference, trial loop and report live in gemm_test_lib
## (testGemmCta*).
##
## NVIDIA-OpenCL only: the mma.sync inline PTX is embedded in the OpenCL C
## kernel as asm(...), which only NVIDIA's OpenCL compiler accepts. Host
## verifies the device vendor before launching.
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_cta_opencl.nim --nimcache:nimcache/tests/manual_gemm_cta_opencl.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_cta_opencl.nim

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
import workspace/crucible

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

# Work-group spans the full CTA: mma.sync is warp-synchronous, each
# 32-lane warp must execute it convergently, and 128 = 4 warps.
const blockSize = 128
static:
  doAssert blockSize == toIntVal(tiled.atom.threadCount(opA)) * 2 * 2 * 1

const kernelCode = opencl:
  # Epilogue-op state: alpha/beta/C/bias never appear as kernel arguments.
  # Output-first (C): the OpenCL engine's run binds the output at
  # binding 0, inputs at 1..N in signature order.
  proc gemmCtaKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, threadIdx)

  proc gemmCtaIdentityKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiIdentity()
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, threadIdx)

const kernelCodeBias = opencl:
  proc gemmCtaReLUKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    let epi = EpiReLU()
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, threadIdx)

  proc gemmCtaBiasKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      bias: ptr UncheckedArray[float32]) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 32), (1, 64))
    let pB = make_view(B, (32, 32), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    # Bias is a (N,) column vector: the input view has stride-0 rows, so
    # every row of a column reads the same bias element. The view is
    # partitioned like C, which bakes each thread's column base into the
    # fragment's data pointer. The (mCTA, nCTA) tile origin derives from
    # the layout.
    let pBias = make_view(bias, (64, 32), (0, 1))
    var biasView = tiled.partition_C(thr, local_tile(pBias, (32, 16), (mCTA, nCTA)))
    var epi = initEpiAddBias(biasView)
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 32, epi, (32, 16, 32), mCTA, nCTA, threadIdx)

# The single-tile kernel lives in its own opencl block: five view-heavy
# inlined kernels in one block exceed the OpenCL codegen's compile-time
# VM-loop budget (maxLoopIterationsVM).
const kernelCodeSingle = opencl:
  proc gemmCtaKernelSingle(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    # 1×1 CTA grid: only the thread id varies.
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let pA = make_view(A, (32, 32), (1, 32))
    let pB = make_view(B, (16, 32), (1, 16))
    let pC = make_view(C, (32, 16), (1, 32))
    let tC = local_tile(pC, (32, 16), (0, 0))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, 32, 16, 32, epi, (32, 16, 32), 0, 0, threadIdx)

proc runTest() =
  var engine = bkOpenCL.init(kernelCode)
  doAssert engine.deviceName().contains("NVIDIA"),
    "gemm_cta OpenCL needs NVIDIA's OpenCL compiler for the mma.sync asm; got: " &
    engine.deviceName()
  testGemmCta(engine, tiled, "SM80")
  testGemmCtaBeta(engine, tiled, "SM80")
  testGemmCtaIdentity(engine, tiled, "SM80")
  var engineBias = bkOpenCL.init(kernelCodeBias)
  testGemmCtaReLU(engineBias, tiled, "SM80")
  testGemmCtaBias(engineBias, tiled, "SM80")
  var engineSingle = bkOpenCL.init(kernelCodeSingle)
  testGemmCtaSingle(engineSingle, tiled, "SM80")

when isMainModule:
  runTest()
